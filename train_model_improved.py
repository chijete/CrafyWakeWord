# train_model_final.py
# Versión Completa: 
# - Timestamps recorte exacto
# - Integración Hard Negatives y TTS
# - Inyección de Ruido
# - CORRECCIÓN: Generación de ejemplos negativos con palabras cortadas (Partial Wake Words)

import os
import sys
import json
import re
import random
import math
import wave
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from pathlib import Path
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
from multiprocessing import freeze_support

# ---------------------------
# UTILIDADES
# ---------------------------
def list_files(mypath):
  return [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

def getWavAudioDuration(nombre_archivo):
  try:
    with wave.open(nombre_archivo, 'rb') as archivo_audio:
      frecuencia_muestreo = archivo_audio.getframerate()
      num_frames = archivo_audio.getnframes()
      return num_frames / frecuencia_muestreo
  except Exception:
    return 0.0

def read_csv_expect_path(path_csv, base_folder=None):
  if not os.path.exists(path_csv):
    return pd.DataFrame(columns=['path', 'sentence', 'timestamps', 'duration'])
    
  df = pd.read_csv(path_csv)
  
  # Normalizar columna path
  if 'path' not in df.columns:
    found = False
    for col in ['filename', 'file', 'wav', 'audio']:
      if col in df.columns:
        df['path'] = df[col].apply(lambda x: join(base_folder, str(x)) if base_folder else str(x))
        found = True
        break
    if not found:
      print(f"Advertencia: CSV {path_csv} ignorado por falta de columna path.")
      return pd.DataFrame(columns=['path', 'sentence', 'timestamps', 'duration'])
  
  if 'sentence' not in df.columns:
    df['sentence'] = ''
  if 'timestamps' not in df.columns:
    df['timestamps'] = ''
  if 'duration' not in df.columns:
    df['duration'] = 0.0
    
  # Rellenar NAs
  df['sentence'] = df['sentence'].fillna('')
  df['timestamps'] = df['timestamps'].fillna('')
  
  return df

# ---------------------------
# GENERADOR DE PARTIAL NEGATIVES (Nueva función)
# ---------------------------
def generate_partial_negatives(df_pos, wake_words):
    """
    Toma el dataset positivo y genera copias recortadas (inicio o final) de la wake word.
    Estas copias se etiquetan con nombres dummy para que el Collator las trate como negativas (OOV).
    """
    print("Generando negativos parciales (cortes de wake word)...")
    rows = []
    ww_set = set([w.lower() for w in wake_words])
    
    # Patrón para corregir JSONs si es necesario
    key_pattern = re.compile("\'(?P<k>[^ ]+)\'")

    for idx, row in df_pos.iterrows():
        ts_str = str(row['timestamps'])
        if not ts_str or ts_str.strip() == '':
            continue
            
        try:
            # Intentar parsear timestamps
            try:
                ts = json.loads(ts_str)
            except:
                ts = json.loads(key_pattern.sub(r'"\g<k>"', ts_str))
        except:
            continue
            
        # Buscar la wake word en los timestamps
        target_word = None
        for w in ww_set:
            if w in ts:
                target_word = w
                break
        
        if target_word:
            info = ts[target_word]
            start = float(info['start'])
            end = float(info['end'])
            duration = end - start
            
            # Solo si la palabra es suficientemente larga para cortar
            if duration > 0.15:
                
                # 1. Corte del INICIO (Head) - ej: "Alm" de "Almeja"
                # Nos quedamos con el primer 30-60% de la palabra
                ratio_head = random.uniform(0.3, 0.6)
                end_head = start + (duration * ratio_head)
                
                rows.append({
                    'path': row['path'],
                    'sentence': 'partial_head', # Etiqueta dummy negativa
                    'timestamps': json.dumps({'partial_head': {'start': start, 'end': end_head}}),
                    'duration': row['duration']
                })
                
                # 2. Corte del FINAL (Tail) - ej: "meja" de "Almeja"
                # Nos quedamos con el último 30-60% de la palabra
                ratio_tail = random.uniform(0.4, 0.7)
                start_tail = start + (duration * (1 - ratio_tail))
                
                rows.append({
                    'path': row['path'],
                    'sentence': 'partial_tail', # Etiqueta dummy negativa
                    'timestamps': json.dumps({'partial_tail': {'start': start_tail, 'end': end}}),
                    'duration': row['duration']
                })

    df_partials = pd.DataFrame(rows)
    print(f"  + Generados {len(df_partials)} ejemplos parciales negativos.")
    return df_partials

# ---------------------------
# MODELO (SpeechResModel)
# ---------------------------
class ResBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1, dropout=0.1):
    super().__init__()
    self.bn1 = nn.BatchNorm2d(in_channels)
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout)
    
    self.shortcut = nn.Sequential()
    should_downsample = (stride != 1) if isinstance(stride, int) else (stride != (1,1))
    
    if should_downsample or in_channels != out_channels:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels)
      )

  def forward(self, x):
    out = self.conv1(self.relu(self.bn1(x)))
    out = self.dropout(out)
    out = self.conv2(self.relu(self.bn2(out)))
    out += self.shortcut(x)
    return out

class AttentionPooling(nn.Module):
  def __init__(self, input_dim):
    super().__init__()
    self.attention = nn.Sequential(
      nn.Linear(input_dim, 64),
      nn.Tanh(),
      nn.Linear(64, 1),
      nn.Softmax(dim=1)
    )

  def forward(self, x):
    w = self.attention(x)
    return torch.sum(w * x, dim=1)

class SpeechResModel(nn.Module):
  def __init__(self, num_labels, n_maps=48):
    super().__init__()
    self.conv0 = nn.Conv2d(1, n_maps, kernel_size=(3, 3), padding=(1, 1), bias=False)
    self.n_maps = n_maps
    self.res1 = ResBlock(n_maps, n_maps, stride=1)
    self.res2 = ResBlock(n_maps, n_maps, stride=(2, 2)) 
    self.res3 = ResBlock(n_maps, n_maps, stride=1)
    self.res4 = ResBlock(n_maps, n_maps, stride=(2, 2))
    self.pool_freq = nn.AdaptiveAvgPool2d((1, None)) 
    self.attn = AttentionPooling(n_maps)
    self.fc = nn.Sequential(
        nn.Linear(n_maps, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, num_labels)
    )

  def forward(self, x):
    x = self.conv0(x)
    x = self.res1(x)
    x = self.res2(x)
    x = self.res3(x)
    x = self.res4(x)
    x = self.pool_freq(x) 
    x = x.squeeze(2)
    x = x.permute(0, 2, 1)
    x = self.attn(x)
    return self.fc(x)

# ---------------------------
# DATASET & COLLATOR
# ---------------------------
class AudioCollator(object):
  def __init__(self, wake_words, sr, window_size_ms, noise_set=None, augment_prob=0.5):
    self.wake_words = [w.lower() for w in wake_words]
    self.sr = sr
    self.window_size_ms = window_size_ms
    self.noise_set = noise_set or []
    self.augment_prob = augment_prob
    self.max_length = int(self.window_size_ms / 1000 * self.sr)
    self.key_pattern = re.compile("\'(?P<k>[^ ]+)\'")

  def parse_timestamps(self, ts_str):
    if not ts_str or not isinstance(ts_str, str) or ts_str.strip() == '':
      return {}
    try:
      return json.loads(ts_str)
    except:
      try:
        fixed = self.key_pattern.sub(r'"\g<k>"', ts_str)
        return json.loads(fixed)
      except:
        return {}

  def compute_labels_and_crop(self, metadata, audio_data):
    sentence = str(metadata.get('sentence', '')).lower()
    timestamps_str = metadata.get('timestamps', '')
    
    # 1. Determinar etiqueta base
    label = len(self.wake_words) # Default: OOV/Negativo
    target_word = None
    
    if sentence in self.wake_words:
      label = self.wake_words.index(sentence)
      target_word = sentence
    else:
      # Check contenido (por si la frase es más larga)
      for w in self.wake_words:
        if re.search(r'\b' + re.escape(w) + r'\b', sentence):
          label = self.wake_words.index(w)
          target_word = w
          break
    
    # 2. Recorte por Timestamps
    timestamps = self.parse_timestamps(timestamps_str)
    
    if timestamps and isinstance(timestamps, dict):
      
      # CASO POSITIVO: Recortar la wake word exacta
      if label < len(self.wake_words) and target_word in timestamps:
        ts = timestamps[target_word]
        start_idx = int(ts['start'] * self.sr)
        end_idx = int(ts['end'] * self.sr)
        # Margen 50ms
        margin = int(0.05 * self.sr)
        start_idx = max(0, start_idx - margin)
        end_idx = min(len(audio_data), end_idx + margin)
        
        if end_idx > start_idx:
          audio_data = audio_data[start_idx:end_idx]
          
      # CASO NEGATIVO (incluye Hard Negatives y Partials):
      elif label == len(self.wake_words):
        # Si la sentencia es "partial_head" o similar, la buscamos en timestamps
        # Si es una frase negativa normal, buscamos palabras que NO sean wake word
        
        # Palabras válidas para recortar (cualquiera que no sea wake word REAL)
        valid_words = [k for k in timestamps.keys() if k.lower() not in self.wake_words]
        
        if valid_words:
           # Si tenemos "partial_head" o "partial_tail", se seleccionarán aquí
           # porque no están en self.wake_words
           if sentence in valid_words:
               # Priorizar la palabra que coincide con la sentence (para partials)
               chosen_word = sentence
           else:
               chosen_word = random.choice(valid_words)

           ts = timestamps[chosen_word]
           start_idx = int(ts['start'] * self.sr)
           end_idx = int(ts['end'] * self.sr)
           if end_idx > start_idx:
               audio_data = audio_data[start_idx:end_idx]

    return label, audio_data

  def __call__(self, batch):
    audio_tensors = []
    labels = []
    metas = []
    
    for sample in batch:
      path = sample.get('path')
      
      if not path or not os.path.exists(path):
        audio_data = np.zeros(self.max_length, dtype=np.float32)
      else:
        try:
          audio_data, _ = librosa.load(path, sr=self.sr, mono=True)
        except:
          audio_data = np.zeros(self.max_length, dtype=np.float32)

      label, audio_data = self.compute_labels_and_crop(sample, audio_data)

      # Padding / Trimming
      if len(audio_data) > self.max_length:
        if label < len(self.wake_words):
            start = (len(audio_data) - self.max_length) // 2
        else:
            start = random.randint(0, len(audio_data) - self.max_length)
        audio_data = audio_data[start:start + self.max_length]
      elif len(audio_data) < self.max_length:
        pad = self.max_length - len(audio_data)
        pad_left = random.randint(0, pad)
        audio_data = np.pad(audio_data, (pad_left, pad - pad_left), 'constant')

      # Augmentations
      if self.augment_prob > 0 and random.random() < self.augment_prob:
        audio_data = audio_data * random.uniform(0.7, 1.3)
        
        if self.noise_set and random.random() < 0.7:
          noise_path = random.choice(self.noise_set)
          try:
            noise, _ = librosa.load(noise_path, sr=self.sr, mono=True)
            while len(noise) < len(audio_data):
                noise = np.concatenate([noise, noise])
            start_n = random.randint(0, len(noise) - len(audio_data))
            noise = noise[start_n:start_n + len(audio_data)]
            
            snr_db = random.uniform(5, 15)
            sig_rms = np.sqrt(np.mean(audio_data**2) + 1e-9)
            noise_rms = np.sqrt(np.mean(noise**2) + 1e-9)
            scale = sig_rms / (noise_rms * (10**(snr_db/20)) + 1e-9)
            
            audio_data = audio_data + (noise * scale)
          except Exception: pass
            
      audio_tensors.append(torch.from_numpy(audio_data).float())
      labels.append(label)
      metas.append(sample)

    return {
      'audio': torch.stack(audio_tensors),
      'labels': torch.tensor(labels, dtype=torch.long),
      'meta': metas
    }

# ---------------------------
# SAMPLER & HNM
# ---------------------------
def create_weighted_sampler_from_df(df, wake_words):
  labels = []
  ww_set = set(wake_words)
  
  print("Calculando pesos del Sampler...")
  for _, row in df.iterrows():
    s = str(row['sentence']).lower()
    lbl = len(wake_words) # OOV
    
    if s in ww_set:
        lbl = wake_words.index(s)
    else:
        for i, w in enumerate(wake_words):
            if w in s:
                lbl = i
                break
    labels.append(lbl)
    
  num_labels = len(wake_words) + 1
  counts = np.bincount(labels, minlength=num_labels)
  print(f"Distribución de clases (aprox): {counts}")
  
  weights_per_class = 1.0 / (counts + 1e-6)
  weights_per_sample = np.array([weights_per_class[l] for l in labels])
  
  return WeightedRandomSampler(
    weights=torch.DoubleTensor(weights_per_sample),
    num_samples=len(weights_per_sample),
    replacement=True
  )

def run_hard_negative_mining(model, pool_df, collator, mel_transform, zmuv, device, wake_words_len, threshold=0.6, max_collect=1000, batch_size=64):
  model.eval()
  found = []
  
  if len(pool_df) > 20000:
      pool_df = pool_df.sample(20000)
      
  records = pool_df.to_dict(orient='records')
  loader = torch.utils.data.DataLoader(records, batch_size=batch_size, collate_fn=collator, num_workers=0)
  
  print(f"HNM: Escaneando {len(records)} candidatos negativos...")
  
  with torch.no_grad():
    for batch in tqdm(loader, desc='HNM Inference', leave=False):
      aud = batch['audio'].to(device)
      mel = mel_transform(aud).unsqueeze(1)
      log_mel = (mel + 1e-6).log()
      normed = zmuv(log_mel)
      
      logits = model(normed)
      probs = F.softmax(logits, dim=1)
      max_probs, preds = torch.max(probs, dim=1)
      
      for i in range(len(preds)):
        pr = preds[i].item()
        p_score = max_probs[i].item()
        
        if pr < wake_words_len and p_score > threshold:
          path = batch['meta'][i]['path']
          found.append({'path': path, 'sentence': '', 'score': p_score})
          if len(found) >= max_collect: break
      if len(found) >= max_collect: break
  return pd.DataFrame(found)

# ---------------------------
# ZMUV
# ---------------------------
class ZmuvTransform(nn.Module):
  def __init__(self):
    super().__init__()
    self.register_buffer('mean', torch.zeros(1))
    self.register_buffer('std', torch.ones(1))
    self.initialized = False

  def fit(self, dataloader, mel_transform, device, limit_batches=50):
    if self.initialized: return
    print("Calculando media/std global (ZMUV)...")
    sum_x = 0.0; sum_sq = 0.0; count = 0.0
    
    for i, batch in enumerate(tqdm(dataloader, desc="ZMUV Stats")):
      if i >= limit_batches: break
      aud = batch['audio'].to(device)
      mel = mel_transform(aud).unsqueeze(1)
      log_mel = (mel + 1e-6).log()
      sum_x += log_mel.mean().item() * log_mel.numel()
      sum_sq += (log_mel ** 2).mean().item() * log_mel.numel()
      count += log_mel.numel()
      
    mean_val = sum_x / count
    std_val = math.sqrt((sum_sq / count) - mean_val**2)
    self.mean.fill_(mean_val)
    self.std.fill_(std_val)
    self.initialized = True
    print(f"ZMUV Init: Mean={mean_val:.4f}, Std={std_val:.4f}")

  def forward(self, x):
    return (x - self.mean) / (self.std + 1e-9)

# ---------------------------
# MAIN
# ---------------------------
def main():
  try:
    with open('your_config.json', 'r') as f:
      config = json.load(f)
  except:
    config = {}

  wake_words = [w.lower() for w in config.get('wake_words', ['marvin'])] 
  if not wake_words: wake_words = ['marvin']
  
  dataset_base = config.get('dataset_base', 'dataset')
  path_pos = join(dataset_base, 'positive')
  path_neg = join(dataset_base, 'negative')
  path_gen = join(dataset_base, 'generated')
  checkpoint_dir = join(dataset_base, 'checkpoints')
  os.makedirs(checkpoint_dir, exist_ok=True)
  
  sr = config.get('sample_rate', 16000)
  window_ms = config.get('window_size_ms', 1000)
  batch_size = config.get('batch_size', 32)
  num_workers = 0 
  lr = 1e-3
  max_epochs = config.get('train_epochs', 30)
  n_mels = 64
  n_fft = 512
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f'Device: {device}')

  # 1. CARGA DE DATASETS
  try:
    df_pos_train = read_csv_expect_path(join(path_pos, 'train.csv'), path_pos)
    df_neg_train = read_csv_expect_path(join(path_neg, 'train.csv'), path_neg)
    df_pos_dev = read_csv_expect_path(join(path_pos, 'dev.csv'), path_pos)
    df_neg_dev = read_csv_expect_path(join(path_neg, 'dev.csv'), path_neg)
  except Exception as e:
    print(f"Error cargando CSVs base: {e}")
    return

  # TTS Generado
  print("Cargando TTS...")
  df_gen_train = read_csv_expect_path(join(path_gen, 'train.csv'), path_gen)
  df_gen_dev = read_csv_expect_path(join(path_gen, 'dev.csv'), path_gen)
  if len(df_gen_train) > 0: df_pos_train = pd.concat([df_pos_train, df_gen_train], ignore_index=True)
  if len(df_gen_dev) > 0: df_pos_dev = pd.concat([df_pos_dev, df_gen_dev], ignore_index=True)

  # Ruidos Puros como Negativos
  noise_train_dir = 'noise/noise_train'
  if os.path.exists(noise_train_dir):
      noise_files = list_files(noise_train_dir)
      if noise_files:
          print(f"Agregando {len(noise_files)} archivos de ruido puro...")
          df_noise = pd.DataFrame({'path': noise_files, 'sentence': '', 'timestamps': '', 'duration': 0.0})
          df_neg_train = pd.concat([df_neg_train, df_noise], ignore_index=True)

  # --- NUEVA CORRECCIÓN: Agregar cortes parciales de wake words ---
  df_partials = generate_partial_negatives(df_pos_train, wake_words)
  if len(df_partials) > 0:
      df_neg_train = pd.concat([df_neg_train, df_partials], ignore_index=True)
  # ----------------------------------------------------------------

  train_df = pd.concat([df_pos_train, df_neg_train], ignore_index=True)
  dev_df = pd.concat([df_pos_dev, df_neg_dev], ignore_index=True)
  
  print(f"TOTAL Train: {len(train_df)} | Dev: {len(dev_df)}")

  # 2. PREPARACIÓN
  train_sampler = create_weighted_sampler_from_df(train_df, wake_words)
  noise_bg_files = list_files('noise/noise_train') if os.path.exists('noise/noise_train') else []
  
  collator_train = AudioCollator(wake_words, sr, window_ms, noise_set=noise_bg_files, augment_prob=0.6)
  collator_dev = AudioCollator(wake_words, sr, window_ms, noise_set=None, augment_prob=0.0)

  train_dl = torch.utils.data.DataLoader(train_df.to_dict('records'), batch_size=batch_size, sampler=train_sampler, collate_fn=collator_train, num_workers=num_workers)
  dev_dl = torch.utils.data.DataLoader(dev_df.to_dict('records'), batch_size=batch_size, shuffle=False, collate_fn=collator_dev, num_workers=num_workers)

  # 3. TRANSFORM & MODELO
  mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, n_mels=n_mels).to(device)
  zmuv = ZmuvTransform().to(device)
  
  batch_init_dl = torch.utils.data.DataLoader(train_df.sample(min(500, len(train_df))).to_dict('records'), batch_size=32, collate_fn=collator_dev)
  zmuv.fit(batch_init_dl, mel_transform, device)

  model = SpeechResModel(num_labels=len(wake_words) + 1).to(device)
  criterion = nn.CrossEntropyLoss(weight=None) 
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

  # 4. ENTRENAMIENTO
  best_val_loss = float('inf')
  patience_counter = 0
  
  for epoch in range(1, max_epochs + 1):
    model.train()
    train_loss = 0; train_correct = 0; train_total = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    
    pbar = tqdm(train_dl, desc=f"Ep {epoch} Train")
    for batch in pbar:
      aud = batch['audio'].to(device)
      lbl = batch['labels'].to(device)
      
      with torch.no_grad():
        mel = mel_transform(aud).unsqueeze(1) 
        log_mel = (mel + 1e-6).log()
        normed = zmuv(log_mel)
      
      if epoch < max_epochs - 5: 
        if random.random() < 0.5:
          f0 = random.randint(0, max(0, n_mels - 10))
          normed[:, :, f0:f0+10, :] = 0
        if random.random() < 0.5:
          t_dim = normed.shape[3]
          t0 = random.randint(0, max(0, t_dim - 20))
          normed[:, :, :, t0:t0+20] = 0

      optimizer.zero_grad()
      out = model(normed)
      loss = criterion(out, lbl)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      
      train_loss += loss.item() * lbl.size(0)
      _, pred = torch.max(out, 1)
      train_correct += (pred == lbl).sum().item()
      train_total += lbl.size(0)
      
      ww_limit = len(wake_words)
      is_ww = (lbl < ww_limit)
      pred_ww = (pred < ww_limit)
      
      tp += (is_ww & pred_ww).sum().item()
      fn += (is_ww & (~pred_ww)).sum().item()
      tn += ((~is_ww) & (~pred_ww)).sum().item()
      fp += ((~is_ww) & pred_ww).sum().item()
      
      recall = tp / (tp + fn + 1e-6)
      spec = tn / (tn + fp + 1e-6)
      pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Rec': f"{recall:.2f}", 'Spec': f"{spec:.2f}"})

    avg_train_loss = train_loss / train_total
    avg_train_acc = train_correct / train_total
    
    # Val
    model.eval()
    val_loss = 0; val_correct = 0; val_total = 0
    v_tp, v_fp, v_tn, v_fn = 0, 0, 0, 0
    
    with torch.no_grad():
      for batch in tqdm(dev_dl, desc=f"Ep {epoch} Val"):
        aud = batch['audio'].to(device)
        lbl = batch['labels'].to(device)
        
        mel = mel_transform(aud).unsqueeze(1)
        log_mel = (mel + 1e-6).log()
        normed = zmuv(log_mel)
        
        out = model(normed)
        loss = criterion(out, lbl)
        
        val_loss += loss.item() * lbl.size(0)
        _, pred = torch.max(out, 1)
        val_correct += (pred == lbl).sum().item()
        val_total += lbl.size(0)
        
        is_ww = (lbl < ww_limit)
        pred_ww = (pred < ww_limit)
        v_tp += (is_ww & pred_ww).sum().item()
        v_fn += (is_ww & (~pred_ww)).sum().item()
        v_tn += ((~is_ww) & (~pred_ww)).sum().item()
        v_fp += ((~is_ww) & pred_ww).sum().item()

    avg_val_loss = val_loss / val_total
    avg_val_acc = val_correct / val_total
    val_recall = v_tp / (v_tp + v_fn + 1e-6)
    val_spec = v_tn / (v_tn + v_fp + 1e-6)
    
    print(f"\nRESUMEN EP {epoch}:")
    print(f"Train Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc:.4f} | Recall: {recall:.4f} | Spec: {spec:.4f}")
    print(f"Val   Loss: {avg_val_loss:.4f} | Acc: {avg_val_acc:.4f} | Recall: {val_recall:.4f} | Spec: {val_spec:.4f}")
    
    scheduler.step(avg_val_loss)
    
    meta_info = {
        "epoch": epoch,
        "labels": {"wake_words": wake_words, "classes": wake_words + ["oov"]},
        "audio_config": {"sample_rate": sr, "window_ms": window_ms, "n_mels": n_mels, "n_fft": n_fft},
        "normalization": {"zmuv_mean": zmuv.mean.item(), "zmuv_std": zmuv.std.item()},
        "metrics": {"val_loss": avg_val_loss, "val_acc": avg_val_acc}
    }
    
    ckpt_path = join(checkpoint_dir, f"model_epoch_{epoch}.pt")
    json_path = join(checkpoint_dir, f"model_epoch_{epoch}.json")
    torch.save(model.state_dict(), ckpt_path)
    with open(json_path, 'w') as f: json.dump(meta_info, f, indent=2)

    if avg_val_loss < best_val_loss:
        print(f"--> MEJOR MODELO (loss {best_val_loss:.4f} -> {avg_val_loss:.4f})")
        best_val_loss = avg_val_loss
        best_ckpt_path = join(checkpoint_dir, 'model_best.pt')
        best_json_path = join(checkpoint_dir, 'model_best.json')
        torch.save(model.state_dict(), best_ckpt_path)
        with open(best_json_path, 'w') as f: json.dump(meta_info, f, indent=2)
        patience_counter = 0
    else:
        patience_counter += 1
      
    if epoch % 3 == 0 and epoch < max_epochs:
      print("Ejecutando HNM...")
      hnm_df = run_hard_negative_mining(model, df_neg_train, collator_dev, mel_transform, zmuv, device, len(wake_words))
      if len(hnm_df) > 0:
        print(f"HNM: Añadiendo {len(hnm_df)} negativos difíciles.")
        train_df = pd.concat([train_df, hnm_df], ignore_index=True)
        train_sampler = create_weighted_sampler_from_df(train_df, wake_words)
        train_dl = torch.utils.data.DataLoader(train_df.to_dict('records'), batch_size=batch_size, sampler=train_sampler, collate_fn=collator_train, num_workers=num_workers)

  print("Entrenamiento finalizado.")

if __name__ == '__main__':
  freeze_support()
  main()