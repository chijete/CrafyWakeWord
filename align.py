import subprocess
import json
import os
import shutil
from pathlib import Path

with open('your_config.json', 'r') as archivo_json:
  config_datos = json.load(archivo_json)

if os.path.isdir('dataset/aligned_data'):
  shutil.rmtree('dataset/aligned_data')
os.mkdir('dataset/aligned_data')

# positive_data

# Define el comando que deseas ejecutar
comando = "mfa align dataset/positive/audio "+config_datos['mfa_DICTIONARY_PATH']+" "+config_datos['mfa_ACOUSTIC_MODEL_PATH']+" dataset/aligned_data --clean --single_speaker"

# Ejecuta el comando
resultado = subprocess.run(comando, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Imprime la salida estándar y la salida de error
print("Output:")
print(resultado.stdout)

print("Output error:")
print(resultado.stderr)

# negative nada

# Define el comando que deseas ejecutar
neg_comando = "mfa align dataset/negative/audio_align "+config_datos['mfa_DICTIONARY_PATH']+" "+config_datos['mfa_ACOUSTIC_MODEL_PATH']+" dataset/aligned_data --clean --single_speaker"

# Ejecuta el comando
neg_resultado = subprocess.run(neg_comando, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Imprime la salida estándar y la salida de error
print("Output:")
print(neg_resultado.stdout)

print("Output error:")
print(neg_resultado.stderr)

# Mover archivos desde audio_align/ a audio/

origen = Path("dataset/negative/audio_align")
destino = Path("dataset/negative/audio")

for archivo in origen.iterdir():
  if archivo.is_file():
    shutil.move(str(archivo), str(destino))


print("Task finished!")