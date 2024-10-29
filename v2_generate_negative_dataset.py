import pandas as pd
import numpy as np
import json
import sys
import librosa
import soundfile as sf
import os
import shutil
import argparse

parser = argparse.ArgumentParser(description='V2 negative dataset gen')
parser.add_argument('-limit', type=int, required=False, help='Limit of audio clips')
args = parser.parse_args()

generated_dataset_limit = 5000
if args.limit:
  generated_dataset_limit = args.limit

with open('your_config.json', 'r') as archivo_json:
  config_datos = json.load(archivo_json)

target_dataset_path = 'v2_dataset/negative/'

if os.path.isdir('v2_dataset/'):
  shutil.rmtree('v2_dataset/')
os.mkdir('v2_dataset/')

if os.path.isdir(target_dataset_path):
  shutil.rmtree(target_dataset_path)
os.mkdir(target_dataset_path)

os.mkdir(target_dataset_path + 'clips/')

common_voice_datapath = config_datos['common_voice_datapath']

train_data = pd.read_csv(common_voice_datapath + 'train.tsv', sep='\t')

my_train_data = train_data.head(generated_dataset_limit)

dataframe_columns = ['path', 'sentence', 'timestamps', 'duration']
train_negative_df = pd.DataFrame(columns=dataframe_columns)

for dfIndex, trainElement in my_train_data.iterrows():

  final_clip_pathname = trainElement['path'].replace('.mp3', '.wav')

  audio, sr = librosa.load(common_voice_datapath + 'clips/' + trainElement['path'], sr=None)
  duration_ms = len(audio) / sr * 1000
  sf.write(target_dataset_path + 'clips/' + final_clip_pathname, audio, sr)

  train_negative_df = pd.concat([train_negative_df, pd.DataFrame([{
    'path': 'clips/' + final_clip_pathname,
    'sentence': trainElement['sentence'],
    'timestamps': '{}',
    'duration': duration_ms,
  }])], ignore_index=True)

train_negative_df.to_csv(target_dataset_path + 'dataset.csv', index=False)

print('Dataset generated in: ' + target_dataset_path + 'dataset.csv')