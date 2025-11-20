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

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

with open('your_config.json', 'r') as archivo_json:
  config_datos = json.load(archivo_json)

wake_words = config_datos['wake_words']
wake_words_sequence = []
for indice, elemento in enumerate(wake_words):
  wake_words_sequence.append(str(indice))
wake_word_seq_map = dict(zip(wake_words, wake_words_sequence))

sr = 16000

path_to_dataset = 'dataset'
path_to_dataset_w = path_to_dataset + '/'

# ------------

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
  log_mels = mel_spectrogram(audio_data.float()).add_(1e-7).log_().contiguous()
  # returns (channel, n_mels, time)
  return log_mels.to(device)

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

model.load_state_dict(torch.load(path_to_dataset_w + 'model_trained.pt'))

model.eval()

classes = wake_words[:]
# oov
classes.append("oov")

audio_float_size = 32767
p = pyaudio.PyAudio()

CHUNK = 500
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = sr
RECORD_MILLI_SECONDS = config_datos['window_size_ms']

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* listening .. ")

print('Say this words in order:')
for word in wake_words:
  print(word)

inference_track = []
target_state = 0

while True:
  no_of_frames = 4

  #import pdb;pdb.set_trace()
  batch = []
  for frame in range(no_of_frames):
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_MILLI_SECONDS/1000)):
      data = stream.read(CHUNK)
      frames.append(data)
    audio_data = np.frombuffer( b''.join(frames), dtype=np.int16).astype(np.float64) / audio_float_size
    inp = torch.from_numpy(audio_data).float().to(device)
    batch.append(inp)

  audio_tensors = torch.stack(batch)

  #sav_temp_wav(frames)
  mel_audio_data = audio_transform(audio_tensors)
  mel_audio_data = zmuv_transform(mel_audio_data)
  scores = model(mel_audio_data.unsqueeze(1))
  scores = F.softmax(scores, -1).squeeze(1)  # [no_of_frames x num_labels]
  #import pdb;pdb.set_trace()
  for score in scores:
    preds = score.cpu().detach().numpy()
    preds = preds / preds.sum()
    # print([f"{x:.3f}" for x in preds.tolist()])
    pred_idx = np.argmax(preds)
    pred_word = classes[pred_idx]
    #print(f"predicted label {pred_idx} - {pred_word}")
    label = wake_words[target_state]
    if pred_word == label:
      target_state += 1 # go to next label
      inference_track.append(pred_word)
      print(inference_track)
      if inference_track == wake_words:
        print(f"Wake word {' '.join(inference_track)} detected")
        target_state = 0
        inference_track = []
        break