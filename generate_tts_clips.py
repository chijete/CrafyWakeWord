import pandas as pd
import numpy as np
import re
import json
import os
from os import listdir
from os.path import isfile, join
from google.cloud import texttospeech
import sys
from pathlib import Path
import time
import warnings
warnings.simplefilter("ignore", UserWarning)
import argparse

parser = argparse.ArgumentParser(description='Generate TextToSpeech audio clips')
parser.add_argument('-word', type=str, required=True, help='Word to TTS')
args = parser.parse_args()

wake_word = args.word

with open('your_config.json', 'r') as archivo_json:
  config_datos = json.load(archivo_json)

dataset_language = config_datos['dataset_language']
ttsConfig = config_datos['tts_generated_clips']

if (ttsConfig['rate']['start'] >= 0.25 and ttsConfig['rate']['start'] <= 4.0 and ttsConfig['rate']['stop'] >= 0.25 and ttsConfig['rate']['stop'] <= 4.0 and ttsConfig['rate']['start'] <= ttsConfig['rate']['stop']) and (ttsConfig['pitch']['start'] >= -20.0 and ttsConfig['pitch']['start'] <= 20.0 and ttsConfig['pitch']['stop'] >= -20.0 and ttsConfig['pitch']['stop'] <= 20.0 and ttsConfig['pitch']['start'] <= ttsConfig['pitch']['stop']):
  i7512 = 1
else:
  print('your_config.json > tts_generated_clips invalid values. rate must be in the range [0.25, 4.0] and pitch must be in the range [-20.0, 20.0], and start must be lower than stop.')
  sys.exit()

path_to_dataset = 'dataset'
path_to_dataset_w = path_to_dataset + '/'

generated_data = path_to_dataset_w + 'generated'
Path(f"{generated_data}").mkdir(parents=True, exist_ok=True)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=config_datos['google_credentials_file']

client = texttospeech.TextToSpeechClient()

def generate_voices(word):
  Path(f"{generated_data}/{word}").mkdir(parents=True, exist_ok=True)
  # Set the text input to be synthesized
  synthesis_input = texttospeech.SynthesisInput(text=word)
  # Performs the list voices request
  voices = client.list_voices()
  # Get english voices

  en_voices =  [voice.name for voice in voices.voices if voice.name.split("-")[0] == dataset_language]
  speaking_rates = np.arange(ttsConfig['rate']['start'], ttsConfig['rate']['stop'], ttsConfig['rate']['step']).tolist()
  pitches = np.arange(ttsConfig['pitch']['start'], ttsConfig['pitch']['stop'], ttsConfig['pitch']['step']).tolist()

  file_count = 0
  start = time.time()

  for voi in en_voices:
    for sp_rate in speaking_rates:
      for pit in pitches:
        try:
          file_name = f'{generated_data}/{word}/{voi}_{sp_rate}_{pit}.wav'
          voice = texttospeech.VoiceSelectionParams(language_code=voi[:5], name=voi)
          # Select the type of audio file you want returned
          audio_config = texttospeech.AudioConfig(
            # format of the audio byte stream.
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,

            #Speaking rate/speed, in the range [0.25, 4.0]. 1.0 is the normal native speed
            speaking_rate=sp_rate,

            #Speaking pitch, in the range [-20.0, 20.0]. 20 means increase 20 semitones from the original pitch. -20 means decrease 20 semitones from the original pitch.
            pitch=pit # [-10, -5, 0, 5, 10]
          )
          response = client.synthesize_speech(
            request={"input": synthesis_input, "voice": voice, "audio_config": audio_config}
          )
          # The response's audio_content is binary.
          with open(file_name, "wb") as out:
            out.write(response.audio_content)
            file_count+=1
          if file_count%100 == 0:
            end = time.time()
            print(f"generated {file_count} files in {end-start} seconds")
        except Exception as e:
          print("In-generation error, you can ignore it.")

print("Generating audios with Google Cloud text-to-speech API:")
generate_voices(wake_word)

d = {}
d['path'] = [f"{generated_data}/{wake_word}/{file_name}" for file_name in os.listdir(f"{generated_data}/{wake_word}")]
d['sentence'] = [wake_word] * len(d['path'])
pd.DataFrame(data=d).to_csv(f"{generated_data}/{wake_word}.csv", index=False)

print("Finished! Saved in: " + generated_data)