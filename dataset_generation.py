import pandas as pd
import numpy as np
from IPython.display import Audio
import librosa
import soundfile

import json
import math
import os
import shutil
from os import listdir
from os.path import isfile, join
import json
import re
import random
from pathlib import Path
import sys

with open('your_config.json', 'r') as archivo_json:
  config_datos = json.load(archivo_json)

common_voice_datapath = config_datos['common_voice_datapath']

wake_words = config_datos['wake_words']
wake_words_sequence = []
for indice, elemento in enumerate(wake_words):
  wake_words_sequence.append(str(indice))
wake_word_seq_map = dict(zip(wake_words, wake_words_sequence))

sr = 16000

train_data = pd.read_csv(common_voice_datapath + 'train.tsv', sep='\t')
dev_data = pd.read_csv(common_voice_datapath + 'dev.tsv', sep='\t')
test_data = pd.read_csv(common_voice_datapath + 'test.tsv', sep='\t')

if len(config_datos['custom_dataset_path']) > 0:
  if os.path.isdir(config_datos['custom_dataset_path'] + 'clips'):
    if os.path.exists(config_datos['custom_dataset_path'] + 'train.csv'):
      custom_train_data = pd.read_csv(config_datos['custom_dataset_path'] + 'train.csv')
      for customAudioPath in custom_train_data['path']:
        shutil.copy(config_datos['custom_dataset_path'] + 'clips/' + customAudioPath, common_voice_datapath + '/clips/')
      train_data = pd.concat([train_data, custom_train_data], ignore_index=True)
    if os.path.exists(config_datos['custom_dataset_path'] + 'dev.csv'):
      custom_dev_data = pd.read_csv(config_datos['custom_dataset_path'] + 'dev.csv')
      for customAudioPath in custom_dev_data['path']:
        shutil.copy(config_datos['custom_dataset_path'] + 'clips/' + customAudioPath, common_voice_datapath + '/clips/')
      dev_data = pd.concat([dev_data, custom_dev_data], ignore_index=True)
    if os.path.exists(config_datos['custom_dataset_path'] + 'test.csv'):
      custom_test_data = pd.read_csv(config_datos['custom_dataset_path'] + 'test.csv')
      for customAudioPath in custom_test_data['path']:
        shutil.copy(config_datos['custom_dataset_path'] + 'clips/' + customAudioPath, common_voice_datapath + '/clips/')
      test_data = pd.concat([test_data, custom_test_data], ignore_index=True)

print(f"Total clips available in Train {train_data.shape[0]}")
print(f"Total clips available in Dev {dev_data.shape[0]}")
print(f"Total clips available in Test {test_data.shape[0]}")

# print(train_data.columns)

# sys.exit()

regex_pattern = r'\b(?:{})\b'.format('|'.join(map(re.escape, wake_words)))
pattern = re.compile(regex_pattern, flags=re.IGNORECASE)
def wake_words_search(pattern, word):
  try:
    return bool(pattern.search(word))
  except TypeError:
    return False

positive_train_data = train_data[[wake_words_search(pattern, sentence) for sentence in train_data['sentence']]]
positive_dev_data = dev_data[[wake_words_search(pattern, sentence) for sentence in dev_data['sentence']]]
positive_test_data = test_data[[wake_words_search(pattern, sentence) for sentence in test_data['sentence']]]

print(f"Total clips available in Train with wake words {positive_train_data.shape[0]}")
print(f"Total clips available in Dev with wake words {positive_dev_data.shape[0]}")
print(f"Total clips available in Test with wake words {positive_test_data.shape[0]}")

positive_train_data.head()

negative_train_data = train_data[[not wake_words_search(pattern, sentence) for sentence in train_data['sentence']]]
negative_dev_data = dev_data[[not wake_words_search(pattern, sentence) for sentence in dev_data['sentence']]]
negative_test_data = test_data[[not wake_words_search(pattern, sentence) for sentence in test_data['sentence']]]

# negative data size
print(f"Total clips available in Train without wake words {negative_train_data.shape[0]}")
print(f"Total clips available in Dev without wake words {negative_dev_data.shape[0]}")
print(f"Total clips available in Test without wake words {negative_test_data.shape[0]}")

negative_data_percent = 1

negative_train_data = negative_train_data.sample(math.floor(negative_train_data.shape[0] * (negative_data_percent/100)))
negative_dev_data = negative_dev_data.sample(math.floor(negative_dev_data.shape[0] * (negative_data_percent/100)))
negative_test_data = negative_test_data.sample(math.floor(negative_test_data.shape[0] * (negative_data_percent/100)))

# trimmed negative data sizes
print(f"Total clips available in Train without wake words {negative_train_data.shape[0]}")
print(f"Total clips available in Dev without wake words {negative_dev_data.shape[0]}")
print(f"Total clips available in Test without wake words {negative_test_data.shape[0]}")

# save dateframes to json files
jsons_container_folder = 'dataset/json/'

if not os.path.isdir('dataset'):
  os.mkdir('dataset')

if os.path.isdir(jsons_container_folder):
  shutil.rmtree(jsons_container_folder)
os.mkdir(jsons_container_folder)

positive_train_data.to_json(jsons_container_folder + 'positive_train_data.json')
positive_dev_data.to_json(jsons_container_folder + 'positive_dev_data.json')
positive_test_data.to_json(jsons_container_folder + 'positive_test_data.json')

negative_train_data.to_json(jsons_container_folder + 'negative_train_data.json')
negative_dev_data.to_json(jsons_container_folder + 'negative_dev_data.json')
negative_test_data.to_json(jsons_container_folder + 'negative_test_data.json')

# copy audios dataset
wake_word_datapath = 'dataset'

positive_audios_container_folder = 'dataset/positive/audio/'
if os.path.isdir('dataset/positive'):
  shutil.rmtree('dataset/positive')
os.mkdir('dataset/positive')
os.mkdir(positive_audios_container_folder)

negative_audios_container_folder = 'dataset/negative/audio/'
if os.path.isdir('dataset/negative'):
  shutil.rmtree('dataset/negative')
os.mkdir('dataset/negative')
os.mkdir(negative_audios_container_folder)

# save the dataframes we got from above in each dataset
positive_train_data[['path', 'sentence']].to_csv(wake_word_datapath + "/positive/train.csv", index=False)
positive_dev_data[['path', 'sentence']].to_csv(wake_word_datapath + "/positive/dev.csv", index=False)
positive_test_data[['path', 'sentence']].to_csv(wake_word_datapath + "/positive/test.csv", index=False)

negative_train_data[['path', 'sentence']].to_csv(wake_word_datapath  + "/negative/train.csv", index=False)
negative_dev_data[['path', 'sentence']].to_csv(wake_word_datapath  + "/negative/dev.csv", index=False)
negative_test_data[['path', 'sentence']].to_csv(wake_word_datapath  + "/negative/test.csv", index=False)

def save_wav_lab(path, filename, sentence, decibels=40):
  # load file
  sounddata = librosa.core.load(f"{common_voice_datapath}/clips/{filename}", sr=sr, mono=True)[0]
  # trim
  sounddata = librosa.effects.trim(sounddata, top_db=decibels)[0]
  # save as wav file
  soundfile.write(f"{wake_word_datapath}{path}/{filename.split('.')[0]}.wav", sounddata, sr)
  # write lab file
  with open(f"{wake_word_datapath}{path}/{filename.split('.')[0]}.lab", "w", encoding="utf-8") as f:
    f.write(sentence)

positive_train_data.apply(lambda x: save_wav_lab('/positive/audio', x['path'], x['sentence']), axis=1)
positive_dev_data.apply(lambda x: save_wav_lab('/positive/audio', x['path'], x['sentence']), axis=1)
positive_test_data.apply(lambda x: save_wav_lab('/positive/audio', x['path'], x['sentence']), axis=1)

negative_train_data.apply(lambda x: save_wav_lab('/negative/audio', x['path'], x['sentence']), axis=1)
negative_dev_data.apply(lambda x: save_wav_lab('/negative/audio', x['path'], x['sentence']), axis=1)
negative_test_data.apply(lambda x: save_wav_lab('/negative/audio', x['path'], x['sentence']), axis=1)