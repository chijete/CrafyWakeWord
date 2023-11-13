# from: https://github.com/rajashekar/WakeWordDetector/blob/main/onnx_to_tf/convert_onnx_to_tf.py

import onnx

from onnx_tf.backend import prepare
import tensorflow as tf

import numpy as np

import os
import json

with open('your_config.json', 'r') as archivo_json:
  config_datos = json.load(archivo_json)

wake_words = config_datos['wake_words']
sr = 16000

path_to_dataset = 'dataset'
path_to_dataset_w = path_to_dataset + '/'

onnx_model = onnx.load(path_to_dataset_w + "onnx_model_trained.onnx")  # load onnx model

tf_rep = prepare(onnx_model)  # prepare tf representation

# Input nodes to the model
print("inputs:", tf_rep.inputs)

# Output nodes from the model
print("outputs:", tf_rep.outputs)

# All nodes in the model
print("tensor_dict:")
print(tf_rep.tensor_dict)

tf_rep.export_graph(path_to_dataset_w + "tf_model_trained")  # export the model

# so used below method
model = tf.saved_model.load(path_to_dataset_w + "tf_model_trained")
input_shape = [1, 1, 40, 61]
func = tf.function(model).get_concrete_function(input=tf.TensorSpec(shape=input_shape, dtype=np.float32, name="input"))
converter = tf.lite.TFLiteConverter.from_concrete_functions([func])

tflite_model = converter.convert()
open(path_to_dataset_w + "tf_model_trained.tflite", "wb").write(tflite_model)