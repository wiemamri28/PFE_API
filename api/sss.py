import tensorflow as tf
import tensorflow_hub as hub


def save_module(url, save_path):
  module = hub.KerasLayer(url)
  model = tf.keras.Sequential(module)
  tf.saved_model.save(model, save_path)


save_module("https://kaggle.com/models/rishitdagli/plant-disease/frameworks/TensorFlow2/variations/plant-disease/versions/1", "./saved-module")

# https://www.kaggle.com/models/rishitdagli/plant-disease/TensorFlow2/plant-disease/1/code

# https://www.kaggle.com/code/vikash013/project

# https://www.kaggle.com/models/rishitdagli/plant-disease/TensorFlow2/plant-disease/1#:~:text=New%20Notebook-,About%20Variation,-FINE%2DTUNABLE