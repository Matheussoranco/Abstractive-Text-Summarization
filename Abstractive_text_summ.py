#Install !pip install git+https://github.com/keras-team/keras-hub.git py7zr -q

import os

os.environ["KERAS_BACKEND"] = "jax"

import py7zr
import time

import keras_hub
import keras
import tensorflow as tf
import tensorflow_datasets as tfds
