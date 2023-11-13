# This file it's not part of the process (optional execution).

# from: https://github.com/rajashekar/WakeWordDetector/blob/main/onnx_to_tf/test_tf.py

import numpy as np
import tensorflow as tf

import os
import json

with open('your_config.json', 'r') as archivo_json:
  config_datos = json.load(archivo_json)

wake_words = config_datos['wake_words']
sr = 16000

path_to_dataset = 'dataset'
path_to_dataset_w = path_to_dataset + '/'

# -------

physical_devices = tf.config.list_physical_devices("GPU")
for device in physical_devices:
  tf.config.experimental.set_memory_growth(device, True)

model = tf.saved_model.load(path_to_dataset_w + "tf_model_trained")

print(list(model.signatures.keys()))

infer = model.signatures["serving_default"]
print(infer.structured_outputs)

input_shape = [1, 1, 40, 61]
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

output = infer(tf.constant(input_data))["output"]
print(output)
print(f"model predicted - {output.numpy().argmax()}")