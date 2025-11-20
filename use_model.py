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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

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
log_offset = 1e-7

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CREATE MEL SPECTROGRAM TRANSFORM ONCE AND KEEP IN GPU
mel_spectrogram_transform = MelSpectrogram(n_mels=num_mels,
                                          sample_rate=sr,
                                          n_fft=num_fft,
                                          hop_length=hop_length,
                                          norm='slaney').to(device)

def audio_transform(audio_data):
  # Ensure audio_data is on GPU
  if not audio_data.is_cuda:
    audio_data = audio_data.to(device)
  
  # Use pre-initialized transform that's already on GPU
  log_mels = mel_spectrogram_transform(audio_data.float()).add_(log_offset).log_().contiguous()
  # returns (channel, n_mels, time) - already on GPU
  return log_mels

# IMPROVED MODEL WITH DEPTHWISE SEPARABLE CONVOLUTIONS AND RESIDUAL CONNECTIONS
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                  padding=padding, stride=stride, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, use_depthwise=True):
        super(ResidualBlock, self).__init__()
        
        if use_depthwise:
            self.conv = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size, padding, stride)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=True)
        
        self.bn = nn.BatchNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.3)  # More aggressive dropout
        
        # Skip connection adjustment if dimensions change
        self.skip_connection = None
        if in_channels != out_channels or stride != 1:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Adjust skip connection if needed
        if self.skip_connection is not None:
            identity = self.skip_connection(identity)
        elif identity.shape != out.shape:
            # Simple padding/cropping for shape mismatch
            if identity.shape[2] > out.shape[2]:  # Height
                diff = identity.shape[2] - out.shape[2]
                identity = identity[:, :, diff//2:diff//2+out.shape[2], :]
            if identity.shape[3] > out.shape[3]:  # Width
                diff = identity.shape[3] - out.shape[3]
                identity = identity[:, :, :, diff//2:diff//2+out.shape[3]]
        
        # Only add if shapes match exactly
        if identity.shape == out.shape:
            out = out + identity
        
        return out

class ImprovedCNN(nn.Module):
    def __init__(self, num_labels, num_maps1, num_maps2, num_hidden_input, hidden_size):
        super(ImprovedCNN, self).__init__()
        self.num_hidden_input = num_hidden_input
        
        # First residual block
        self.encoder1 = nn.Sequential(
            ResidualBlock(1, num_maps1, (8, 16), padding=(4, 0), stride=(2, 2)),
            nn.MaxPool2d(2)
        )
        
        # Second residual block  
        self.encoder2 = nn.Sequential(
            ResidualBlock(num_maps1, num_maps2, (5, 5), padding=2, stride=(2, 1)),
            nn.MaxPool2d(2)
        )
        
        # Output layers with more aggressive dropout
        self.output = nn.Sequential(
            nn.Linear(num_hidden_input, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased dropout
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_labels)
        )
    
    def forward(self, x):
        x = x[:, :1]  # log_mels only
        x = x.permute(0, 1, 3, 2)  # (time, n_mels)
        
        # Pass through residual blocks
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        
        # Flattening
        x = x2.view(-1, self.num_hidden_input)
        return self.output(x)

# Original CNN for backward compatibility
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

# Try to load improved model first, fallback to original if it doesn't exist
try:
    model = ImprovedCNN(num_labels, num_maps1, num_maps2, num_hidden_input, hidden_size)
    model.to(device)
    model.load_state_dict(torch.load(path_to_dataset_w + 'model_trained_improved.pt', map_location=device))
    print("Loaded improved model successfully!")
except:
    print("Improved model not found, loading original model...")
    model = CNN(num_labels, num_maps1, num_maps2, num_hidden_input, hidden_size)
    model.to(device)
    model.load_state_dict(torch.load(path_to_dataset_w + 'model_trained.pt', map_location=device))
    print("Loaded original model successfully!")

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

# Per-utterance normalization transform
class PerUtteranceNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        # x shape: (batch, channels, n_mels, time)
        # Normalize over n_mels and time dimensions for each utterance
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True)
        return (x - mean) / (std + self.eps)

zmuv_transform = ZmuvTransform().to(device)
per_utterance_norm = PerUtteranceNorm().to(device)

if Path(path_to_dataset_w + "zmuv.pt.bin").exists():
  zmuv_transform.load_state_dict(torch.load(str(path_to_dataset_w + "zmuv.pt.bin"), map_location=device))
  print("Loaded ZMUV transform successfully!")
else:
  print("Warning: ZMUV transform not found! Model may not work properly.")

model.eval()

classes = wake_words[:]
# oov
classes.append("oov")

print(f"Model loaded with classes: {classes}")
print(f"Model is on GPU: {next(model.parameters()).is_cuda}")

# Enhanced inference with confidence thresholding and smoothing
class KeywordDetector:
    def __init__(self, wake_words, confidence_threshold=0.7, smoothing_window=3):
        self.wake_words = wake_words
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.prediction_buffer = []
        self.target_state = 0
        self.inference_track = []
        
    def reset(self):
        """Reset detector state"""
        self.target_state = 0
        self.inference_track = []
        self.prediction_buffer = []
        
    def smooth_predictions(self, current_pred, confidence):
        """Apply temporal smoothing to reduce false positives"""
        self.prediction_buffer.append((current_pred, confidence))
        
        # Keep only recent predictions
        if len(self.prediction_buffer) > self.smoothing_window:
            self.prediction_buffer.pop(0)
            
        # Get most common prediction with high confidence
        if len(self.prediction_buffer) >= self.smoothing_window:
            recent_preds = [pred for pred, conf in self.prediction_buffer if conf > self.confidence_threshold]
            if recent_preds and len(recent_preds) >= self.smoothing_window // 2:
                # Return most common prediction
                return max(set(recent_preds), key=recent_preds.count)
        
        return None
    
    def process_prediction(self, pred_word, confidence, verbose=False):
        """Process a single prediction"""
        # Apply smoothing
        smoothed_pred = self.smooth_predictions(pred_word, confidence)
        
        if smoothed_pred is None:
            return False, None
            
        if verbose:
            print(f"Detected: {smoothed_pred} (conf: {confidence:.3f})")
            
        # Check if it's the expected wake word
        expected_label = self.wake_words[self.target_state]
        
        if smoothed_pred == expected_label:
            self.target_state += 1
            self.inference_track.append(smoothed_pred)
            
            if verbose:
                print(f"Progress: {self.inference_track}")
                
            # Check if sequence is complete
            if self.inference_track == self.wake_words:
                return True, self.inference_track.copy()
        else:
            # Reset if wrong word detected (but not if it's just background noise/oov)
            if smoothed_pred != "oov" and smoothed_pred in self.wake_words:
                # Check if it's the first word in sequence
                if smoothed_pred == self.wake_words[0]:
                    self.target_state = 1
                    self.inference_track = [smoothed_pred]
                    if verbose:
                        print(f"Restarted sequence: {self.inference_track}")
                else:
                    self.reset()
                    if verbose:
                        print("Sequence reset due to unexpected word")
                        
        return False, None

# Initialize detector
detector = KeywordDetector(wake_words, confidence_threshold=0.6, smoothing_window=2)

audio_float_size = 32767
p = pyaudio.PyAudio()

CHUNK = 500
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = sr
RECORD_MILLI_SECONDS = config_datos['window_size_ms']

try:
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
except Exception as e:
    print(f"Error opening audio stream: {e}")
    print("Make sure your microphone is connected and accessible.")
    sys.exit(1)

print("* Listening for wake words...")
print('Say these words in order:')
for i, word in enumerate(wake_words):
  print(f"{i+1}. {word}")
print()
print("Press Ctrl+C to stop")
print("-" * 50)

try:
    while True:
        no_of_frames = 4

        batch = []
        for frame in range(no_of_frames):
            frames = []
            for i in range(0, int(RATE / CHUNK * RECORD_MILLI_SECONDS/1000)):
                data = stream.read(CHUNK)
                frames.append(data)
            
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float64) / audio_float_size
            inp = torch.from_numpy(audio_data).float().to(device)
            batch.append(inp)

        audio_tensors = torch.stack(batch)

        # Process audio through improved pipeline
        with torch.no_grad():
            mel_audio_data = audio_transform(audio_tensors)
            mel_audio_data = zmuv_transform(mel_audio_data)
            mel_audio_data = per_utterance_norm(mel_audio_data.unsqueeze(1))
            
            scores = model(mel_audio_data)
            scores = F.softmax(scores, -1)  # [no_of_frames x num_labels]
            
            for score in scores:
                preds = score.cpu().detach().numpy()
                preds = preds / preds.sum()
                
                pred_idx = np.argmax(preds)
                pred_word = classes[pred_idx]
                confidence = preds[pred_idx]
                
                # Process through enhanced detector
                sequence_complete, detected_sequence = detector.process_prediction(
                    pred_word, confidence, verbose=False
                )
                
                if sequence_complete:
                    print(f"🎉 WAKE WORD SEQUENCE DETECTED: {' '.join(detected_sequence)}")
                    print("=" * 60)
                    detector.reset()
                    
                    # Optional: Add a short pause to avoid immediate re-triggering
                    time.sleep(1)

except KeyboardInterrupt:
    print("\nStopping keyword detection...")
except Exception as e:
    print(f"Error during inference: {e}")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Audio stream closed.")