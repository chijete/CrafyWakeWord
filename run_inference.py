# run_inference.py
# Script de inferencia en tiempo real para detección de Wake Word.
# Carga model_best.pt y model_best.json desde dataset/checkpoints/

import os
import json
import time
import struct
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import pyaudio
from collections import deque

# ----------------------------------------------------------
# 1. DEFINICIÓN DE LA ARQUITECTURA (Debe ser idéntica al training)
# ----------------------------------------------------------
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

# ----------------------------------------------------------
# 2. CLASE DE INFERENCIA EN TIEMPO REAL
# ----------------------------------------------------------
class WakeWordDetector:
    def __init__(self, checkpoint_dir="dataset/checkpoints", threshold=0.85):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.cooldown = 0
        
        # Rutas
        model_path = os.path.join(checkpoint_dir, "model_best.pt")
        json_path = os.path.join(checkpoint_dir, "model_best.json")
        
        if not os.path.exists(model_path) or not os.path.exists(json_path):
            raise FileNotFoundError("No se encontraron model_best.pt o model_best.json")
            
        # Cargar Configuración
        print(f"Cargando configuración desde {json_path}...")
        with open(json_path, 'r') as f:
            self.meta = json.load(f)
            
        # Extraer parámetros
        self.sr = self.meta['audio_config']['sample_rate']
        self.window_ms = self.meta['audio_config']['window_ms']
        self.n_mels = self.meta['audio_config']['n_mels']
        self.n_fft = self.meta['audio_config']['n_fft']
        self.labels = self.meta['labels']['classes']
        self.wake_words = self.meta['labels']['wake_words']
        self.zmuv_mean = self.meta['normalization']['zmuv_mean']
        self.zmuv_std = self.meta['normalization']['zmuv_std']
        
        # Calcular tamaño del buffer (muestras)
        self.buffer_size = int(self.sr * (self.window_ms / 1000))
        # Buffer circular lleno de ceros al inicio
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        
        # Cargar Modelo
        print(f"Cargando modelo en {self.device}...")
        num_classes = len(self.labels)
        self.model = SpeechResModel(num_labels=num_classes).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Transformación Mel (la misma que en training)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels
        ).to(self.device)
        
        print("Detector inicializado correctamente.")

    def process_frame(self, audio_frame):
        """
        Procesa un frame de audio (un tensor de floats).
        Retorna: (is_detected, label_detected, probability)
        """
        # Actualizar cooldown
        if self.cooldown > 0:
            self.cooldown -= 1
            return False, None, 0.0

        # Preprocesamiento
        with torch.no_grad():
            # 1. Mel Spectrogram
            # audio_frame shape: (1, samples)
            mel = self.mel_transform(audio_frame) # (1, n_mels, T)
            mel = mel.unsqueeze(1) # (1, 1, n_mels, T)
            
            # 2. Log + ZMUV
            log_mel = (mel + 1e-6).log()
            normed = (log_mel - self.zmuv_mean) / (self.zmuv_std + 1e-9)
            
            # 3. Inferencia
            outputs = self.model(normed)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # 4. Obtener predicción
            score, pred_idx = torch.max(probs, dim=1)
            score = score.item()
            pred_idx = pred_idx.item()
            
            label = self.labels[pred_idx]
            
            # Verificar si es Wake Word
            if label in self.wake_words and score > self.threshold:
                # Establecer cooldown para no detectar la misma palabra 20 veces seguidas
                # (aprox 1.5 segundos si procesamos chunks de 0.1s)
                self.cooldown = 15 
                return True, label, score
            
            return False, label, score

# ----------------------------------------------------------
# 3. BUCLE PRINCIPAL (Micrófono)
# ----------------------------------------------------------
def main():
    detector = WakeWordDetector(threshold=0.6) # Ajusta threshold si es muy sensible
    
    # Configurar PyAudio
    CHUNK_MS = 100 # Leer audio en trozos de 100ms
    CHUNK_SIZE = int(detector.sr * CHUNK_MS / 1000)
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    
    p = pyaudio.PyAudio()
    
    print("\n" + "="*40)
    print(f" ESCUCHANDO... (Wake Words: {detector.wake_words})")
    print(" Presiona Ctrl+C para salir.")
    print("="*40 + "\n")
    
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=detector.sr,
                        input=True,
                        frames_per_buffer=CHUNK_SIZE)
        
        while True:
            # 1. Leer datos crudos del micro
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            
            # 2. Convertir a numpy float32
            # PyAudio da int16 bytes -> convertir a int16 -> normalizar a float [-1, 1]
            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # 3. Actualizar Buffer Deslizante (Rolling Buffer)
            # Quitamos lo antiguo, metemos lo nuevo al final
            detector.audio_buffer = np.roll(detector.audio_buffer, -CHUNK_SIZE)
            detector.audio_buffer[-CHUNK_SIZE:] = audio_chunk
            
            # 4. Convertir buffer completo a Tensor
            tensor_buffer = torch.from_numpy(detector.audio_buffer).float().to(detector.device)
            # Añadir dimensión de batch (1, L)
            tensor_buffer = tensor_buffer.unsqueeze(0)
            
            # 5. Detectar
            is_detected, label, score = detector.process_frame(tensor_buffer)
            
            if is_detected:
                print(f"🔴 DETECTADO: ¡{label.upper()}! (Confianza: {score:.2f})")
            else:
                # Opcional: imprimir un punto para saber que está vivo
                # print(".", end="", flush=True)
                pass

    except KeyboardInterrupt:
        print("\nDeteniendo...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()