import pandas as pd
import numpy as np
import librosa
import librosa.display
import soundfile

import torch
import torch.utils.data as tud
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchaudio.transforms import MelSpectrogram, ComputeDeltas

from torch.optim.adamw import AdamW

import textgrid

import re
import json

import os
from os import listdir
from os.path import isfile, join

import math
import random

from pathlib import Path
from IPython.display import Audio

from tqdm import tqdm
tqdm.pandas()

import matplotlib.pyplot as plt

import pyaudio
import wave

from fastprogress import master_bar, progress_bar

from google.cloud import texttospeech

import warnings
warnings.simplefilter("ignore", UserWarning)

import sys
import time

with open('your_config.json', 'r') as archivo_json:
  config_datos = json.load(archivo_json)

wake_words = config_datos['wake_words']
wake_words_sequence = []
for indice, elemento in enumerate(wake_words):
  wake_words_sequence.append(str(indice))
wake_word_seq_map = dict(zip(wake_words, wake_words_sequence))

sr = 16000

dataset_language = config_datos['dataset_language']
add_vanilla_noise_to_negative_dataset = config_datos['add_vanilla_noise_to_negative_dataset']
generateVoicesWithGoogle = config_datos['voices_generation_with_google']
windowSizeFromConfig = config_datos['window_size_ms']

path_to_dataset = 'dataset'
path_to_dataset_w = path_to_dataset + '/'

# ------------

print("NOTE: Running this file may take several minutes.")

wake_words_withOOV = wake_words[:]
wake_words_withOOV.append("oov")

def list_files(mypath):
  return [mypath + f for f in listdir(mypath) if isfile(join(mypath, f))]

def getWavAudioDuration(nombre_archivo):
  with wave.open(nombre_archivo, 'rb') as archivo_audio:
    # Obtén la frecuencia de muestreo (número de muestras por segundo)
    frecuencia_muestreo = archivo_audio.getframerate()
    # Obtén el número total de frames (muestras)
    num_frames = archivo_audio.getnframes()
    # Calcula la duración en segundos
    duracion = num_frames / frecuencia_muestreo
  return duracion

noise_test = list_files('noise/noise_test/')
noise_train_complete = list_files('noise/noise_train/')

regex_pattern = r'\b(?:{})\b'.format('|'.join(map(re.escape, wake_words)))
pattern = re.compile(regex_pattern, flags=re.IGNORECASE)
def wake_words_search(pattern, word):
  try:
    return bool(pattern.search(word))
  except TypeError:
    return False

# Dataset checkpoint
positive_train_data = pd.read_csv(path_to_dataset_w+'positive/train.csv')
positive_dev_data = pd.read_csv(path_to_dataset_w+'positive/dev.csv')
positive_test_data = pd.read_csv(path_to_dataset_w+'positive/test.csv')

negative_train_data = pd.read_csv(path_to_dataset_w+'negative/train.csv')
negative_dev_data = pd.read_csv(path_to_dataset_w+'negative/dev.csv')
negative_test_data = pd.read_csv(path_to_dataset_w+'negative/test.csv')

# Add vanilla noise to negative dataset

max_noise_duration = 90000

if add_vanilla_noise_to_negative_dataset:

  for noiseItemPath in noise_train_complete:
    noiseItemDuration = round(getWavAudioDuration(noiseItemPath) * 1000, 1)
    if noiseItemDuration <= max_noise_duration:
      negative_train_data = pd.concat([negative_train_data, pd.DataFrame([{
        'path': noiseItemPath,
        'sentence': 'Hsdflkjhsdf lhskldhfañsljf sñdlkfjñsdf',
        'timestamps': {},
        'duration': noiseItemDuration
      }])], ignore_index=True)

  for noiseItemPath in noise_test:
    noiseItemDuration = round(getWavAudioDuration(noiseItemPath) * 1000, 1)
    if noiseItemDuration <= max_noise_duration:
      negative_test_data = pd.concat([negative_test_data, pd.DataFrame([{
        'path': noiseItemPath,
        'sentence': 'Hsdflkjhsdf lhskldhfañsljf sñdlkfjñsdf',
        'timestamps': {},
        'duration': noiseItemDuration
      }])], ignore_index=True)

# max duration in positive dataset
print(f"Max duration in positive train {positive_train_data['duration'].max()}")
print(f"Min duration in positive train {positive_train_data['duration'].min()}")
print(f"Max duration in positive dev {positive_dev_data['duration'].max()}")
print(f"Min duration in positive dev {positive_dev_data['duration'].min()}")
print(f"Max duration in positive test {positive_test_data['duration'].max()}")
print(f"Min duration in positive test {positive_test_data['duration'].min()}")

# max duration in negative dataset
print(f"Max duration in negative train {negative_train_data['duration'].max()}")
print(f"Min duration in negative train {negative_train_data['duration'].min()}")
print(f"Max duration in negative dev {negative_dev_data['duration'].max()}")
print(f"Min duration in negative dev {negative_dev_data['duration'].min()}")
print(f"Max duration in negative test {negative_test_data['duration'].max()}")
print(f"Min duration in negative test {negative_test_data['duration'].min()}")

train_ds = pd.concat([positive_train_data , negative_train_data]).sample(frac=1).reset_index(drop=True)
dev_ds = pd.concat([positive_dev_data , negative_dev_data]).sample(frac=1).reset_index(drop=True)
test_ds = pd.concat([positive_test_data , negative_test_data]).sample(frac=1).reset_index(drop=True)

print(f"Training dataset size {train_ds.shape}")
print(f"Validation dataset size {dev_ds.shape}")
print(f"Test dataset size {test_ds.shape}")

# checking pattern spread on train_ds
for word in wake_words:
  word_pattern = re.compile(r'\b'+word+r'\b', flags=re.IGNORECASE)
  print(word + f" Total word {(train_ds[[wake_words_search(word_pattern, sentence) for sentence in train_ds['sentence']]].size/train_ds.size) * 100} %")

generated_data = path_to_dataset_w + 'generated'
Path(f"{generated_data}").mkdir(parents=True, exist_ok=True)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=config_datos['google_credentials_file']

client = texttospeech.TextToSpeechClient()

def generate_voices(word):
  Path(f"{generated_data}/{word}").mkdir(parents=True, exist_ok=True)
  # Set the text input to be synthesized
  synthesis_input = texttospeech.SynthesisInput(text=word)
  # Performs the list voices request
  voices = client.list_voices()
  # Get english voices
  en_voices =  [voice.name for voice in voices.voices if voice.name.split("-")[0] == dataset_language]
  speaking_rates = np.arange(0.25, 4.25, 0.25).tolist()
  pitches = np.arange(-10.0, 10.0, 2).tolist()
  file_count = 0
  start = time.time()

  for voi in en_voices:
    for sp_rate in speaking_rates:
      for pit in pitches:
        file_name = f'{generated_data}/{word}/{voi}_{sp_rate}_{pit}.wav'
        voice = texttospeech.VoiceSelectionParams(language_code=voi[:5], name=voi)
        # Select the type of audio file you want returned
        audio_config = texttospeech.AudioConfig(
          # format of the audio byte stream.
          audio_encoding=texttospeech.AudioEncoding.LINEAR16,

          #Speaking rate/speed, in the range [0.25, 4.0]. 1.0 is the normal native speed
          speaking_rate=sp_rate,

          #Speaking pitch, in the range [-20.0, 20.0]. 20 means increase 20 semitones from the original pitch. -20 means decrease 20 semitones from the original pitch.
          pitch=pit # [-10, -5, 0, 5, 10]
        )
        response = client.synthesize_speech(
          request={"input": synthesis_input, "voice": voice, "audio_config": audio_config}
        )
        # The response's audio_content is binary.
        with open(file_name, "wb") as out:
          out.write(response.audio_content)
          file_count+=1
        if file_count%100 == 0:
          end = time.time()
          print(f"generated {file_count} files in {end-start} seconds")

# Voices generation with Google Cloud text-to-speech API
if generateVoicesWithGoogle:
  print("Generating audios with Google Cloud text-to-speech API:")
  for word in wake_words:
    generate_voices(word)

for word in wake_words:
  d = {}
  d['path'] = [f"{generated_data}/{word}/{file_name}" for file_name in os.listdir(f"{generated_data}/{word}")]
  d['sentence'] = [word] * len(d['path'])
  pd.DataFrame(data=d).to_csv(f"{generated_data}/{word}.csv", index=False)

word_cols = {'path' : [], 'sentence': []}
train, dev, test = pd.DataFrame(word_cols), pd.DataFrame(word_cols), pd.DataFrame(word_cols)
for word in wake_words:
  word_df = pd.read_csv(f"{generated_data}/{word}.csv")
  tra, val, te =  np.split(word_df.sample(frac=1, random_state=42),  [int(.6*len(word_df)), int(.8*len(word_df))])
  train = pd.concat([train , tra]).sample(frac=1).reset_index(drop=True)
  dev = pd.concat([dev , val]).sample(frac=1).reset_index(drop=True)
  test = pd.concat([test , te]).sample(frac=1).reset_index(drop=True)

# Checkpoint save
train.to_csv(f"{generated_data}/train.csv", index=False)
dev.to_csv(f"{generated_data}/dev.csv", index=False)
test.to_csv(f"{generated_data}/test.csv", index=False)

# add dummy values for these columns for generated data
train['timestamps'] = ''
train['duration'] = ''

dev['timestamps'] = ''
dev['duration'] = ''

test['timestamps'] = ''
test['duration'] = ''

train_ds = pd.concat([train_ds , train]).sample(frac=1).reset_index(drop=True)
dev_ds = pd.concat([dev_ds , dev]).sample(frac=1).reset_index(drop=True)
test_ds = pd.concat([test_ds , test]).sample(frac=1).reset_index(drop=True)

print('train_ds.shape', train_ds.shape)

print(f"Training dataset size {train_ds.shape}")
print(f"Validation dataset size {dev_ds.shape}")
print(f"Test dataset size {test_ds.shape}")

# now verify how much data we have for train set
for word in wake_words:
  word_pattern = re.compile(r'\b'+word+r'\b', flags=re.IGNORECASE)
  print(word + f" (2) Total word {(train_ds[[wake_words_search(word_pattern, sentence) for sentence in train_ds['sentence']]].size/train_ds.size) * 100} %")

# --- Add noise

noise_train = noise_train_complete[:int(len(noise_train_complete) * 0.8)]
noise_dev = noise_train_complete[int(len(noise_train_complete) * 0.8):]

# random.randint(0,len(noise_dev))
# print noise data stats
print(f"Train noise dataset {len(noise_train)}")
print(f"Train noise dataset {len(noise_dev)}")
print(f"Train noise dataset {len(noise_test)}")

key_pattern = re.compile("\'(?P<k>[^ ]+)\'")
def compute_labels(metadata, audio_data):
  label = len(wake_words) # by default negative label

  # if it is generated data then
  if metadata['sentence'].lower() in wake_words:
    label = int(wake_word_seq_map[metadata['sentence'].lower()])
  else:
    # if the sentence has one wakeword get label and end timestamp
    for word in metadata['sentence'].lower().split():
      wake_word_found = False
      word = re.sub('\W+', '', word)
      if word in wake_words:
        wake_word_found = True
        break

    if wake_word_found:
      label = int(wake_word_seq_map[word])
      if word in  metadata['timestamps']:
        timestamps = metadata['timestamps']
        if type(timestamps) == str:
          timestamps = json.loads(key_pattern.sub(r'"\g<k>"', timestamps))
        word_ts = timestamps[word]
        audio_start_idx = int((word_ts['start'] * 1000) * sr / 1000)
        audio_end_idx = int((word_ts['end'] * 1000) * sr / 1000)
        audio_data = audio_data[audio_start_idx:audio_end_idx]
      else: # if there are issues with word alignment, we might not get ts
        label = len(wake_words)  # mark them for negative

  return label, audio_data

class AudioCollator(object):
  def __init__(self, noise_set=None):
    self.noise_set = noise_set

  def __call__(self, batch):
    batch_tensor = {}
    window_size_ms = windowSizeFromConfig
    max_length = int(window_size_ms/1000 * sr)
    audio_tensors = []
    labels = []
    for sample in batch:
      # get audio_data in tensor format
      audio_data = librosa.core.load(sample['path'], sr=sr, mono=True)[0]
      # get the label and its audio
      label, audio_data = compute_labels(sample, audio_data)
      audio_data_length = audio_data.size / sr * 1000 #ms

      # below is to make sure that we always got length of 12000
      # i.e 750 ms with sr 16000
      # trim to max_length
      if audio_data_length > window_size_ms:
        # randomly trim either at start and end
        if random.random() < 0.5:
          audio_data = audio_data[:max_length]
        else:
          audio_data = audio_data[audio_data.size-max_length:]

      # pad with zeros
      if audio_data_length < window_size_ms:
        # randomly either append or prepend
        if random.random() < 0.5:
          audio_data = np.append(audio_data, np.zeros(int(max_length - audio_data.size)))
        else:
          audio_data = np.append(np.zeros(int(max_length - audio_data.size)), audio_data)

      # Add noise
      if self.noise_set:
        noise_level =  random.randint(5, 30)/100 # 5 to 30%
        noise_sample = librosa.core.load(self.noise_set[random.randint(0,len(self.noise_set)-1)], sr=sr, mono=True)[0]
        # randomly select first or last seq of noise
        if random.random() < 0.5:
          audio_data = (1 - noise_level) * audio_data +  noise_level * noise_sample[:max_length]
        else:
          audio_data = (1 - noise_level) * audio_data +  noise_level * noise_sample[-max_length:]

      audio_tensors.append(torch.from_numpy(audio_data))
      labels.append(label)

    batch_tensor = {
        'audio': torch.stack(audio_tensors),
        'labels': torch.tensor(labels)
    }

    return batch_tensor

# --- Prepare for train

batch_size = 16
num_workers = 0

train_audio_collator = AudioCollator(noise_set=noise_train)
train_dl = tud.DataLoader(train_ds.to_dict(orient='records'),
                  batch_size=batch_size,
                  drop_last=True,
                  shuffle=True,
                  num_workers=num_workers,
                  collate_fn=train_audio_collator)

dev_audio_collator = AudioCollator(noise_set=noise_dev)
dev_dl = tud.DataLoader(dev_ds.to_dict(orient='records'),
                  batch_size=batch_size,
                  num_workers=num_workers,
                  collate_fn=dev_audio_collator)

test_audio_collator = AudioCollator(noise_set=noise_test)
test_dl = tud.DataLoader(test_ds.to_dict(orient='records'),
                  batch_size=batch_size,
                  num_workers=num_workers,
                  collate_fn=test_audio_collator)

zmuv_audio_collator = AudioCollator()
zmuv_dl = tud.DataLoader(train_ds.to_dict(orient='records'),
                  batch_size=1,
                  num_workers=num_workers,
                  collate_fn=zmuv_audio_collator)

class CNN(nn.Module):
  def __init__(self, num_labels, num_maps1, num_maps2, num_hidden_input, hidden_size):
    super(CNN, self).__init__()
    conv0 = nn.Conv2d(1, num_maps1, (8, 16), padding=(4, 0), stride=(2, 2), bias=True)
    pool = nn.MaxPool2d(2)
    conv1 = nn.Conv2d(num_maps1, num_maps2, (5, 5), padding=2, stride=(2, 1), bias=True)
    self.num_hidden_input = num_hidden_input
    self.encoder1 = nn.Sequential(conv0,
                                  nn.ReLU(),
                                  pool,
                                  nn.BatchNorm2d(num_maps1, affine=True))
    self.encoder2 = nn.Sequential(conv1,
                                  nn.ReLU(),
                                  pool,
                                  nn.BatchNorm2d(num_maps2, affine=True))
    self.output = nn.Sequential(nn.Linear(num_hidden_input, hidden_size),
                                nn.ReLU(),
                                nn.Dropout(0.1),
                                nn.Linear(hidden_size, num_labels))

  def forward(self, x):
    x = x[:, :1] # log_mels only
    x = x.permute(0, 1, 3, 2)  # (time, n_mels)
    # pass through first conv layer
    x1 = self.encoder1(x)
    # pass through second conv layer
    x2 = self.encoder2(x1)
    # flattening - keep first dim batch same, flatten last 3 dims
    x = x2.view(-1, self.num_hidden_input)
    return self.output(x)

num_labels = len(wake_words) + 1 # oov
num_maps1  = 48
num_maps2  = 64
num_hidden_input =  768
hidden_size = 128
model = CNN(num_labels, num_maps1, num_maps2, num_hidden_input, hidden_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print(model)

class CNN_Cal(nn.Module):
  def __init__(self, num_labels, num_maps1, num_maps2, num_hidden_input, hidden_size):
    super(CNN_Cal, self).__init__()
    conv0 = nn.Conv2d(1, num_maps1, (8, 16), padding=(4, 0), stride=(2, 2), bias=True)
    pool = nn.MaxPool2d(2)
    conv1 = nn.Conv2d(num_maps1, num_maps2, (5, 5), padding=2, stride=(2, 1), bias=True)
    self.encoder1 = nn.Sequential(conv0,
                                  nn.ReLU(),
                                  pool,
                                  nn.BatchNorm2d(num_maps1, affine=True))
    self.encoder2 = nn.Sequential(conv1,
                                  nn.ReLU(),
                                  pool,
                                  nn.BatchNorm2d(num_maps2, affine=True))

  def forward(self, x):
    x = x[:, :1] # log_mels only
    x = x.permute(0, 1, 3, 2)  # change to (time, n_mels)
    # pass through first conv layer
    x1 = self.encoder1(x)
    # pass through second conv layer
    x2 = self.encoder2(x1)
    # flattening - keep first dim batch same, flatten last 3 dims
    x = x2.view(x2.size(0), -1)
    return x

num_labels = len(wake_words) + 1 # oov
num_maps1  = 48
num_maps2  = 64
num_hidden_input =  768
hidden_size = 128
model_calc = CNN_Cal(num_labels, num_maps1, num_maps2, num_hidden_input, hidden_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_calc.to(device)
print(model_calc)

print(summary(model_calc, input_size=(1,80,61)))
print(summary(model, input_size=(1,40,61)))

from typing import Iterable
class ZmuvTransform(nn.Module):
  def __init__(self):
    super().__init__()
    self.register_buffer('total', torch.zeros(1))
    self.register_buffer('mean', torch.zeros(1))
    self.register_buffer('mean2', torch.zeros(1))

  def update(self, data, mask=None):
    with torch.no_grad():
      if mask is not None:
        data = data * mask
        mask_size = mask.sum().item()
      else:
        mask_size = data.numel()
      self.mean = (data.sum() + self.mean * self.total) / (self.total + mask_size)
      self.mean2 = ((data ** 2).sum() + self.mean2 * self.total) / (self.total + mask_size)
      self.total += mask_size

  def initialize(self, iterable: Iterable[torch.Tensor]):
    for ex in iterable:
      self.update(ex)

  @property
  def std(self):
    return (self.mean2 - self.mean ** 2).sqrt()

  def forward(self, x):
    return (x - self.mean) / self.std

zmuv_transform = ZmuvTransform().to(device)
if Path(path_to_dataset_w + "zmuv.pt.bin").exists():
  zmuv_transform.load_state_dict(torch.load(str(path_to_dataset_w + "zmuv.pt.bin")))
else:
  for idx, batch in enumerate(tqdm(zmuv_dl, desc="Constructing ZMUV")):
    zmuv_transform.update(batch['audio'].to(device))
  print(dict(zmuv_mean=zmuv_transform.mean, zmuv_std=zmuv_transform.std))
  torch.save(zmuv_transform.state_dict(), str(path_to_dataset_w + "zmuv.pt.bin"))

print(f"Mean is {zmuv_transform.mean.item():0.6f}")
print(f"Standard Deviation is {zmuv_transform.std.item():0.6f}")

zmuv_mean = zmuv_transform.mean.item()
zmuv_std = zmuv_transform.std.item()

learning_rate = 0.001
weight_decay = 0.0001 # Weight regularization
lr_decay = 0.95

criterion = nn.CrossEntropyLoss()
params = list(filter(lambda x: x.requires_grad, model.parameters()))
optimizer = AdamW(params, learning_rate, weight_decay=weight_decay)

log_offset = 1e-7
num_mels = 40 # https://en.wikipedia.org/wiki/Mel_scale
num_fft = 512 # window length - Fast Fourier Transform
hop_length = 200  # making hops of size hop_length each time to sample the next window
def audio_transform(audio_data):
  # Transformations
  # Mel-scale spectrogram is a combination of Spectrogram and mel scale conversion
  # 1. compute FFT - for each window to transform from time domain to frequency domain
  # 2. Generate Mel Scale - Take entire freq spectrum & seperate to n_mels evenly spaced
  #    frequencies. (not by distance on freq domain but distance as it is heard by human ear)
  # 3. Generate Spectrogram - For each window, decompose the magnitude of the signal
  #    into its components, corresponding to the frequencies in the mel scale.
  mel_spectrogram  = MelSpectrogram(n_mels=num_mels,
                                    sample_rate=sr,
                                    n_fft=num_fft,
                                    hop_length=hop_length,
                                    norm='slaney')
  mel_spectrogram.to(device)
  log_mels = mel_spectrogram(audio_data.float()).add_(log_offset).log_().contiguous()
  # returns (channel, n_mels, time)
  return log_mels.to(device)

# --- Training

# epochs = 20
epochs = config_datos['train_epochs']

# config for progress bar
mb = master_bar(range(epochs))
mb.names = ['Training loss', 'Validation loss']
x = []

training_losses = []
validation_losses = []

valid_mean_min = np.Inf

for epoch in mb:
  x.append(epoch)
  # Evaluate
  model.train()
  total_loss = torch.Tensor([0.0]).to(device)
  #pbar = tqdm(train_dl, total=len(train_dl), position=0, desc="Training", leave=True)
  for batch in progress_bar(train_dl, parent=mb):
    audio_data = batch['audio'].to(device)
    labels = batch['labels'].to(device)
    # get mel spectograms
    mel_audio_data = audio_transform(audio_data)
    # do zmuv transform
    mel_audio_data = zmuv_transform(mel_audio_data)
    predicted_scores = model(mel_audio_data.unsqueeze(1))
    # get loss
    loss = criterion(predicted_scores, labels)

    optimizer.zero_grad()
    model.zero_grad()

    # backward propagation
    loss.backward()
    optimizer.step()

    with torch.no_grad():
      total_loss += loss

  for group in optimizer.param_groups:
    group["lr"] *= lr_decay

  mean = total_loss / len(train_dl)
  training_losses.append(mean.cpu())

  # Evaluate
  model.eval()
  validation_loss = torch.Tensor([0.0]).to(device)
  with torch.no_grad():
    #pbar = tqdm(dev_dl, total=len(dev_dl), position=0, desc="Evaluating", leave=True)
    for batch in progress_bar(dev_dl, parent=mb):
      audio_data = batch['audio'].to(device)
      labels = batch['labels'].to(device)
      # get mel spectograms
      mel_audio_data = audio_transform(audio_data)
      # do zmuv transform
      mel_audio_data = zmuv_transform(mel_audio_data)
      predicted_scores = model(mel_audio_data.unsqueeze(1))
      # get loss
      loss = criterion(predicted_scores, labels)
      validation_loss += loss

  val_mean = validation_loss / len(dev_dl)
  validation_losses.append(val_mean.cpu())

  # Update training chart
  mb.update_graph([[x, training_losses], [x, validation_losses]], [0,epochs])
  mb.write(f"\nEpoch {epoch}: Training loss {mean.item():.6f} validation loss {val_mean.item():.6f} with lr {group['lr']:.6f}")

  # save model if validation loss has decreased
  if val_mean.item() <= valid_mean_min:
    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
    valid_mean_min,
    val_mean.item()))
    torch.save(model.state_dict(), path_to_dataset_w + 'model_trained.pt')
    valid_mean_min = val_mean.item()

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
  print('CUDA is not available.  Training on CPU ...')
else:
  print('CUDA is available!  Training on GPU ...')

path_to_dataset_w

torch.save(model.state_dict(), path_to_dataset_w + 'model_trained.pt')

with open(path_to_dataset_w + 'model_data.json', 'w') as archivo:
  archivo.write(json.dumps({
    "zmuv_mean": zmuv_mean,
    "zmuv_std": zmuv_std,
    "window_size": windowSizeFromConfig,
    "hop_length": hop_length,
    "num_mels": num_mels,
    "num_fft": num_fft,
    "sample_rate": sr,
    "log_offset": log_offset,
    "train_epochs": epochs,
    "original_path": path_to_dataset_w + 'model_trained.pt',
    "final_validation_loss": valid_mean_min,
    "train_epochs": epochs,
    "classes": wake_words_withOOV,
    "classes_base": wake_words,
    "vanilla_noise_in_negative_dataset": add_vanilla_noise_to_negative_dataset
  }))