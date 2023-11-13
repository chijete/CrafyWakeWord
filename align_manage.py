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

positive_train_data = pd.read_json(path_to_dataset+'/json/' + 'positive_train_data.json')
positive_dev_data = pd.read_json(path_to_dataset+'/json/' + 'positive_dev_data.json')
positive_test_data = pd.read_json(path_to_dataset+'/json/' + 'positive_test_data.json')

negative_train_data = pd.read_json(path_to_dataset+'/json/' + 'negative_train_data.json')
negative_dev_data = pd.read_json(path_to_dataset+'/json/' + 'negative_dev_data.json')
negative_test_data = pd.read_json(path_to_dataset+'/json/' + 'negative_test_data.json')

wake_word_datapath = path_to_dataset
positive_data = "/positive/audio"
negative_data = "/negative/audio"

print(positive_train_data.head())

def get_timestamps(path):
  filename = path.split('/')[-1].split('.')[0]
  filepath = path_to_dataset_w + f'aligned_data/{filename}.TextGrid'
  words_timestamps = {}
  if os.path.exists(filepath):
    tg = textgrid.TextGrid.fromFile(filepath)
    for tg_intvl in range(len(tg[0])):
      word = tg[0][tg_intvl].mark
      if word:
        words_timestamps[word] = {'start': tg[0][tg_intvl].minTime, 'end':  tg[0][tg_intvl].maxTime}
  return words_timestamps

def get_duration(path):
  sounddata = librosa.core.load(path, sr=sr, mono=True)[0]
  return sounddata.size / sr * 1000 # ms

positive_train_data = pd.read_csv(path_to_dataset_w+'positive/train.csv')
positive_dev_data = pd.read_csv(path_to_dataset_w+'positive/dev.csv')
positive_test_data = pd.read_csv(path_to_dataset_w+'positive/test.csv')

positive_train_data['path'] = positive_train_data['path'].apply(lambda x: path_to_dataset_w+'positive/audio/'+x.split('.')[0]+'.wav')
positive_dev_data['path'] = positive_dev_data['path'].apply(lambda x: path_to_dataset_w+'positive/audio/'+x.split('.')[0]+'.wav')
positive_test_data['path'] = positive_test_data['path'].apply(lambda x: path_to_dataset_w+'positive/audio/'+x.split('.')[0]+'.wav')

positive_train_data['timestamps'] = positive_train_data['path'].apply(get_timestamps)
positive_dev_data['timestamps'] = positive_dev_data['path'].apply(get_timestamps)
positive_test_data['timestamps'] = positive_test_data['path'].apply(get_timestamps)

positive_train_data['duration'] = positive_train_data['path'].apply(get_duration)
positive_dev_data['duration'] = positive_dev_data['path'].apply(get_duration)
positive_test_data['duration'] = positive_test_data['path'].apply(get_duration)

print(positive_train_data['timestamps'][:5])
print(positive_train_data['duration'][:5])

# sys.exit()

negative_train_data = pd.read_csv(path_to_dataset_w+'negative/train.csv')
negative_dev_data = pd.read_csv(path_to_dataset_w+'negative/dev.csv')
negative_test_data = pd.read_csv(path_to_dataset_w+'negative/test.csv')

negative_train_data['path'] = negative_train_data['path'].apply(lambda x: path_to_dataset_w+'negative/audio/'+x.split('.')[0]+'.wav')
negative_dev_data['path'] = negative_dev_data['path'].apply(lambda x: path_to_dataset_w+'negative/audio/'+x.split('.')[0]+'.wav')
negative_test_data['path'] = negative_test_data['path'].apply(lambda x: path_to_dataset_w+'negative/audio/'+x.split('.')[0]+'.wav')

negative_train_data['timestamps'] = negative_train_data['path'].apply(get_timestamps)
negative_dev_data['timestamps'] = negative_dev_data['path'].apply(get_timestamps)
negative_test_data['timestamps'] = negative_test_data['path'].apply(get_timestamps)

negative_train_data['duration'] = negative_train_data['path'].apply(get_duration)
negative_dev_data['duration'] = negative_dev_data['path'].apply(get_duration)
negative_test_data['duration'] = negative_test_data['path'].apply(get_duration)

print(negative_train_data['timestamps'][:5])
print(negative_train_data['duration'][:5])

# save above data
positive_train_data.to_csv(wake_word_datapath + "/positive/train.csv", index=False)
positive_dev_data.to_csv(wake_word_datapath + "/positive/dev.csv", index=False)
positive_test_data.to_csv(wake_word_datapath + "/positive/test.csv", index=False)

negative_train_data.to_csv(wake_word_datapath  + "/negative/train.csv", index=False)
negative_dev_data.to_csv(wake_word_datapath  + "/negative/dev.csv", index=False)
negative_test_data.to_csv(wake_word_datapath  + "/negative/test.csv", index=False)

print('finished')