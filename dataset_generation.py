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

train_data_woCustom = train_data
dev_data_woCustom = dev_data
test_data_woCustom = test_data

custom_train_data_only = False
custom_dev_data_only = False
custom_test_data_only = False

if len(config_datos['custom_dataset_path']) > 0:
  if os.path.isdir(config_datos['custom_dataset_path'] + 'clips'):
    if os.path.exists(config_datos['custom_dataset_path'] + 'train.csv'):
      custom_train_data = pd.read_csv(config_datos['custom_dataset_path'] + 'train.csv')
      for customAudioPath in custom_train_data['path']:
        shutil.copy(config_datos['custom_dataset_path'] + 'clips/' + customAudioPath, common_voice_datapath + '/clips/')
      custom_train_data_only = custom_train_data
      train_data = pd.concat([custom_train_data, train_data], ignore_index=True)
    if os.path.exists(config_datos['custom_dataset_path'] + 'dev.csv'):
      custom_dev_data = pd.read_csv(config_datos['custom_dataset_path'] + 'dev.csv')
      for customAudioPath in custom_dev_data['path']:
        shutil.copy(config_datos['custom_dataset_path'] + 'clips/' + customAudioPath, common_voice_datapath + '/clips/')
      custom_dev_data_only = custom_dev_data
      dev_data = pd.concat([custom_dev_data, dev_data], ignore_index=True)
    if os.path.exists(config_datos['custom_dataset_path'] + 'test.csv'):
      custom_test_data = pd.read_csv(config_datos['custom_dataset_path'] + 'test.csv')
      for customAudioPath in custom_test_data['path']:
        shutil.copy(config_datos['custom_dataset_path'] + 'clips/' + customAudioPath, common_voice_datapath + '/clips/')
      custom_test_data_only = custom_test_data
      test_data = pd.concat([custom_test_data, test_data], ignore_index=True)

print(f"Total clips available in Train {train_data.shape[0]}")
print(f"Total clips available in Dev {dev_data.shape[0]}")
print(f"Total clips available in Test {test_data.shape[0]}")

# print(train_data.columns)

# print(train_data.head(20))
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

positive_train_data['sentence'] = positive_train_data['sentence'].apply(
  lambda x: x if isinstance(x, str) else ""
)
positive_dev_data['sentence'] = positive_dev_data['sentence'].apply(
  lambda x: x if isinstance(x, str) else ""
)
positive_test_data['sentence'] = positive_test_data['sentence'].apply(
  lambda x: x if isinstance(x, str) else ""
)

print(f"Total clips available in Train with wake words {positive_train_data.shape[0]}")
print(f"Total clips available in Dev with wake words {positive_dev_data.shape[0]}")
print(f"Total clips available in Test with wake words {positive_test_data.shape[0]}")

print(positive_train_data.head(20))
print(positive_train_data.tail(20))

negative_train_data = train_data_woCustom[[not wake_words_search(pattern, sentence) for sentence in train_data_woCustom['sentence']]]
negative_dev_data = dev_data_woCustom[[not wake_words_search(pattern, sentence) for sentence in dev_data_woCustom['sentence']]]
negative_test_data = test_data_woCustom[[not wake_words_search(pattern, sentence) for sentence in test_data_woCustom['sentence']]]

# negative data size
print(f"Total clips available in Train without wake words {negative_train_data.shape[0]}")
print(f"Total clips available in Dev without wake words {negative_dev_data.shape[0]}")
print(f"Total clips available in Test without wake words {negative_test_data.shape[0]}")

# negative_data_percent = 0.012
# 
# negative_train_data = negative_train_data.sample(frac=negative_data_percent)
# negative_dev_data = negative_dev_data.sample(frac=negative_data_percent)
# negative_test_data = negative_test_data.sample(frac=negative_data_percent)

positive_train_amount = positive_train_data.shape[0] + config_datos['tts_generated_amounts']['train']
positive_dev_amount = positive_dev_data.shape[0] + config_datos['tts_generated_amounts']['dev']
positive_test_amount = positive_test_data.shape[0] + config_datos['tts_generated_amounts']['test']

negative_train_data = negative_train_data.sample(int(positive_train_amount / 100 * config_datos['negative_data_proportion']) + config_datos['hnm_pool_size'])
negative_dev_data = negative_dev_data.sample(int(positive_dev_amount / 100 * config_datos['negative_data_proportion']))
negative_test_data = negative_test_data.sample(int(positive_test_amount / 100 * config_datos['negative_data_proportion']))

negative_pool_data = negative_train_data.tail(config_datos['hnm_pool_size']).copy()
negative_train_data = negative_train_data[:-config_datos['hnm_pool_size']]

if isinstance(custom_train_data_only, pd.DataFrame) and not custom_train_data_only.empty:
  custom_negative_train_data = custom_train_data_only[[not wake_words_search(pattern, sentence) for sentence in custom_train_data_only['sentence']]]
  rowsToDeleteAmount = custom_negative_train_data.shape[0]
  negative_train_data = negative_train_data[:-rowsToDeleteAmount]
  negative_train_data = pd.concat([custom_negative_train_data, negative_train_data], ignore_index=True)

if isinstance(custom_dev_data_only, pd.DataFrame) and not custom_dev_data_only.empty:
  custom_negative_dev_data = custom_dev_data_only[[not wake_words_search(pattern, sentence) for sentence in custom_dev_data_only['sentence']]]
  rowsToDeleteAmount = custom_negative_dev_data.shape[0]
  negative_dev_data = negative_dev_data[:-rowsToDeleteAmount]
  negative_dev_data = pd.concat([custom_negative_dev_data, negative_dev_data], ignore_index=True)

if isinstance(custom_test_data_only, pd.DataFrame) and not custom_test_data_only.empty:
  custom_negative_test_data = custom_test_data_only[[not wake_words_search(pattern, sentence) for sentence in custom_test_data_only['sentence']]]
  rowsToDeleteAmount = custom_negative_test_data.shape[0]
  negative_test_data = negative_test_data[:-rowsToDeleteAmount]
  negative_test_data = pd.concat([custom_negative_test_data, negative_test_data], ignore_index=True)

negative_train_data['sentence'] = negative_train_data['sentence'].apply(
  lambda x: x if isinstance(x, str) else ""
)
negative_dev_data['sentence'] = negative_dev_data['sentence'].apply(
  lambda x: x if isinstance(x, str) else ""
)
negative_test_data['sentence'] = negative_test_data['sentence'].apply(
  lambda x: x if isinstance(x, str) else ""
)
negative_pool_data['sentence'] = negative_pool_data['sentence'].apply(
  lambda x: x if isinstance(x, str) else ""
)

# trimmed negative data sizes
print(f"Total clips available in Train without wake words {negative_train_data.shape[0]}")
print(f"Total clips available in Dev without wake words {negative_dev_data.shape[0]}")
print(f"Total clips available in Test without wake words {negative_test_data.shape[0]}")

print('negative_train_data:')
print(negative_train_data.head(20))
print(negative_train_data.tail(20))

print('negative_dev_data:')
print(negative_dev_data.head(10))
print('negative_test_data:')
print(negative_test_data.head(10))

# Filtrar donde 'sentence' no está vacío después de quitar espacios
negative_train_data_with_sentence = negative_train_data[negative_train_data['sentence'].str.strip() != '']
negative_dev_data_with_sentence = negative_dev_data[negative_dev_data['sentence'].str.strip() != '']
negative_test_data_with_sentence = negative_test_data[negative_test_data['sentence'].str.strip() != '']

# Tomar un X% aleatorio de los with_sentence
negative_train_data_with_sentence_piece = negative_train_data_with_sentence.sample(frac=config_datos['negative_with_sentence_percent'])
negative_train_data_with_sentence_rest = negative_train_data_with_sentence.drop(negative_train_data_with_sentence_piece.index)
negative_train_data_with_sentence = negative_train_data_with_sentence_piece

negative_dev_data_with_sentence_piece = negative_dev_data_with_sentence.sample(frac=config_datos['negative_with_sentence_percent'])
negative_dev_data_with_sentence_rest = negative_dev_data_with_sentence.drop(negative_dev_data_with_sentence_piece.index)
negative_dev_data_with_sentence = negative_dev_data_with_sentence_piece

negative_test_data_with_sentence_piece = negative_test_data_with_sentence.sample(frac=config_datos['negative_with_sentence_percent'])
negative_test_data_with_sentence_rest = negative_test_data_with_sentence.drop(negative_test_data_with_sentence_piece.index)
negative_test_data_with_sentence = negative_test_data_with_sentence_piece

# Obtener los without_sentence
negative_train_data_without_sentence = negative_train_data[negative_train_data['sentence'].str.strip() == '']
negative_dev_data_without_sentence = negative_dev_data[negative_dev_data['sentence'].str.strip() == '']
negative_test_data_without_sentence = negative_test_data[negative_test_data['sentence'].str.strip() == '']

# El 50% de los with_sentence que no se tomó los concateno a los without_sentence, porque no los voy a alinear, van como ruido/palabras rotas
negative_train_data_without_sentence = pd.concat([negative_train_data_without_sentence, negative_train_data_with_sentence_rest], ignore_index=True)
negative_dev_data_without_sentence = pd.concat([negative_dev_data_without_sentence, negative_dev_data_with_sentence_rest], ignore_index=True)
negative_test_data_without_sentence = pd.concat([negative_test_data_without_sentence, negative_test_data_with_sentence_rest], ignore_index=True)

print(f"negative_train_data_with_sentence count: {negative_train_data_with_sentence.shape[0]}")
print(f"negative_dev_data_with_sentence count: {negative_dev_data_with_sentence.shape[0]}")
print(f"negative_test_data_with_sentence count: {negative_test_data_with_sentence.shape[0]}")

print(f"negative_train_data_without_sentence count: {negative_train_data_without_sentence.shape[0]}")
print(f"negative_dev_data_without_sentence count: {negative_dev_data_without_sentence.shape[0]}")
print(f"negative_test_data_without_sentence count: {negative_test_data_without_sentence.shape[0]}")

jsonFileData = {
  'positive_train_amount': positive_train_amount,
  'positive_dev_amount': positive_dev_amount,
  'positive_test_amount': positive_test_amount,
  'negative_train_amount': negative_train_data.shape[0],
  'negative_dev_amount': negative_dev_data.shape[0],
  'negative_test_amount': negative_test_data.shape[0],
  'negative_pool_data': negative_pool_data.shape[0],
  'negative_train_data_with_sentence': negative_train_data_with_sentence.shape[0],
  'negative_dev_data_with_sentence': negative_dev_data_with_sentence.shape[0],
  'negative_test_data_with_sentence': negative_test_data_with_sentence.shape[0],
  'negative_train_data_without_sentence': negative_train_data_without_sentence.shape[0],
  'negative_dev_data_without_sentence': negative_dev_data_without_sentence.shape[0],
  'negative_test_data_without_sentence': negative_test_data_without_sentence.shape[0]
}

with open("dataset/dataset_conformation.json", "w", encoding="utf-8") as jsonFile:
  json.dump(jsonFileData, jsonFile, ensure_ascii=False, indent=2)

# sys.exit()

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
negative_pool_data.to_json(jsons_container_folder + 'negative_pool_data.json')

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

negative_audios_align_container_folder = 'dataset/negative/audio_align/'
os.mkdir(negative_audios_align_container_folder)

# save the dataframes we got from above in each dataset
positive_train_data[['path', 'sentence']].to_csv(wake_word_datapath + "/positive/train.csv", index=False)
positive_dev_data[['path', 'sentence']].to_csv(wake_word_datapath + "/positive/dev.csv", index=False)
positive_test_data[['path', 'sentence']].to_csv(wake_word_datapath + "/positive/test.csv", index=False)

negative_train_data[['path', 'sentence']].to_csv(wake_word_datapath  + "/negative/train.csv", index=False)
negative_dev_data[['path', 'sentence']].to_csv(wake_word_datapath  + "/negative/dev.csv", index=False)
negative_test_data[['path', 'sentence']].to_csv(wake_word_datapath  + "/negative/test.csv", index=False)
negative_pool_data[['path', 'sentence']].to_csv(wake_word_datapath  + "/negative/pool.csv", index=False)

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

negative_train_data_without_sentence.apply(lambda x: save_wav_lab('/negative/audio', x['path'], x['sentence']), axis=1)
negative_dev_data_without_sentence.apply(lambda x: save_wav_lab('/negative/audio', x['path'], x['sentence']), axis=1)
negative_test_data_without_sentence.apply(lambda x: save_wav_lab('/negative/audio', x['path'], x['sentence']), axis=1)

negative_train_data_with_sentence.apply(lambda x: save_wav_lab('/negative/audio_align', x['path'], x['sentence']), axis=1)
negative_dev_data_with_sentence.apply(lambda x: save_wav_lab('/negative/audio_align', x['path'], x['sentence']), axis=1)
negative_test_data_with_sentence.apply(lambda x: save_wav_lab('/negative/audio_align', x['path'], x['sentence']), axis=1)