import pandas as pd
import numpy as np
import json
import os
import shutil
import sys

path_to_positive_dataset_v1 = 'dataset/positive/'
csv_files = ['train.csv', 'test.csv', 'dev.csv']
target_dataset_path = 'v2_dataset/positive/'

if os.path.isdir('v2_dataset/') == False:
  print('Execute v2_generate_negative_dataset.py first!')
  sys.exit()

if os.path.isdir(target_dataset_path):
  shutil.rmtree(target_dataset_path)
os.mkdir(target_dataset_path)

os.mkdir(target_dataset_path + 'clips/')

dataframe_columns = ['path', 'sentence', 'timestamps', 'duration']
dataset_positive_df = pd.DataFrame(columns=dataframe_columns)

for csvFileName in csv_files:
  train_data = pd.read_csv(path_to_positive_dataset_v1 + csvFileName)
  dataset_positive_df = pd.concat([dataset_positive_df, train_data], ignore_index=True)

  for dfIndex, trainElement in train_data.iterrows():
    audioFilename = os.path.basename(trainElement['path'])
    newAudioPath = target_dataset_path + 'clips/' + audioFilename
    shutil.copy(trainElement['path'], newAudioPath)

dataset_positive_df.to_csv(target_dataset_path + 'dataset.csv', index=False)