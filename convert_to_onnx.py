# from: https://github.com/rajashekar/WakeWordDetector/blob/main/server/utils/convert_to_onnx.py
# Adapted for improved model with depthwise separable convolutions and residual connections

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
sr = 16000

path_to_dataset = 'dataset'
path_to_dataset_w = path_to_dataset + '/'

# IMPROVED MODEL CLASSES - Same as in training code
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

# Legacy CNN class for backwards compatibility (if needed)
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

num_labels = len(wake_words) + 1  # oov
num_maps1 = 48
num_maps2 = 64
num_hidden_input = 768
hidden_size = 128
batch_size = 1

# get available device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Try to load improved model first, fallback to legacy model
model_paths = [
    (path_to_dataset_w + 'model_trained_improved.pt', ImprovedCNN),
    (path_to_dataset_w + 'model_trained.pt', CNN)
]

pytorch_model = None
model_type = None

for model_path, ModelClass in model_paths:
    if os.path.exists(model_path):
        try:
            pytorch_model = ModelClass(num_labels, num_maps1, num_maps2, num_hidden_input, hidden_size)
            pytorch_model.load_state_dict(
                torch.load(model_path, map_location=device)
            )
            model_type = ModelClass.__name__
            print(f"Successfully loaded {model_type} from {model_path}")
            break
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            continue

if pytorch_model is None:
    raise FileNotFoundError("Could not load any model. Please ensure either 'model_trained_improved.pt' or 'model_trained.pt' exists in the dataset folder.")

# put in eval mode
pytorch_model.eval()

# define the input size
input_size = (1, 40, 61)

# generate dummy data
dummy_input = torch.rand(batch_size, *input_size).type(torch.FloatTensor).to(device=device)

if torch.cuda.is_available():
    dummy_input = dummy_input.to('cuda')
    pytorch_model = pytorch_model.to('cuda')

# Determine output filename based on model type
output_filename = f"onnx_model_trained{'_improved' if model_type == 'ImprovedCNN' else ''}.onnx"

print(f"Converting {model_type} to ONNX format...")
print(f"Input shape: {dummy_input.shape}")
print(f"Output file: {output_filename}")

# generate onnx file
torch.onnx.export(
    pytorch_model,
    dummy_input,
    path_to_dataset_w + output_filename,
    export_params=True,  # store the trained parameter weights inside the model file
    verbose=True,
    input_names=["input"],  # the model's input names
    output_names=["output"],  # the model's output names
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # variable length axes
    opset_version=11,  # Use a stable ONNX opset version
)

print(f"Successfully converted {model_type} to ONNX format: {output_filename}")
print("Conversion finished!")

# Optional: Verify the ONNX model can be loaded
try:
    import onnx
    onnx_model = onnx.load(path_to_dataset_w + output_filename)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification: PASSED")
    
    # Print model info
    print(f"ONNX model inputs: {[input.name for input in onnx_model.graph.input]}")
    print(f"ONNX model outputs: {[output.name for output in onnx_model.graph.output]}")
    
except ImportError:
    print("ONNX package not available for verification. Install with: pip install onnx")
except Exception as e:
    print(f"ONNX model verification failed: {e}")