# from: https://github.com/rajashekar/WakeWordDetector/blob/main/server/utils/convert_to_onnx.py

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
# load model
pytorch_model = CNN(num_labels, num_maps1, num_maps2, num_hidden_input, hidden_size)
pytorch_model.load_state_dict(
  torch.load(path_to_dataset_w + 'model_trained.pt', map_location=device)
)
# put in eval mode
pytorch_model.eval()
# define the input size
input_size = (1, 40, 61)
# generate dummy data
dummy_input = torch.rand(batch_size, *input_size).type(torch.FloatTensor).to(device=device)
if torch.cuda.is_available():
  dummy_input.to('cuda')
  pytorch_model.to('cuda')
# generate onnx file
torch.onnx.export(
  pytorch_model,
  dummy_input,
  path_to_dataset_w + "onnx_model_trained.onnx",
  export_params=True,  # store the trained parameter weights inside the model file
  verbose=True,
  input_names=["input"],  # the model's input names
  output_names=["output"],  # the model's output names
  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # variable length axes
)

print("finished")