from pydub import AudioSegment
import pandas as pd
import sys
import json
import re

keyword = 'cerebro'
path_to_dataset = 'dataset'
path_to_dataset_w = path_to_dataset + '/'

key_pattern = re.compile("\'(?P<k>[^ ]+)\'")

positive_train_data = pd.read_csv(path_to_dataset_w+'positive/train.csv')

trainElement = positive_train_data.iloc[0]
trainElementTimestamps = json.loads(key_pattern.sub(r'"\g<k>"', trainElement['timestamps']))

# Cargar el archivo de audio
audio = AudioSegment.from_file(trainElement['path'])

# Calcular los tiempos en milisegundos
start_time = trainElementTimestamps[keyword]['start'] * 1000
end_time = trainElementTimestamps[keyword]['end'] * 1000

# Cortar el segmento de audio
segmento_cortado = audio[start_time:end_time]

# Guardar el segmento cortado en un nuevo archivo
segmento_cortado.export("cut_audio_test.wav", format="wav")