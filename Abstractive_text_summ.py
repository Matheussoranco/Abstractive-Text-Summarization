#Install !pip install git+https://github.com/keras-team/keras-hub.git py7zr -q

import os

os.environ["KERAS_BACKEND"] = "jax"

import py7zr
import time

import keras_hub
import keras
import tensorflow as tf
import tensorflow_datasets as tfds

BATCH_SIZE = 8
NUM_BATCHES = 600
EPOCHS = 10
MAX_ENCODER_SEQUENCE_LENGTH = 512
MAX_DECODER_SEQUENCE_LENGTH = 128
MAX_GENERATION_LENGTH = 40

filename = keras.utils.get_file(
    "corpus.7z",
    origin="https://huggingface.co/datasets/samsum/resolve/main/data/corpus.7z",
)

with py7zr.SevenZipFile(filename, mode="r") as z:
    z.extractall(path="/root/tensorflow_datasets/downloads/manual")

samsum_ds = tfds.load("samsum", split="train", as_supervised=True)

for dialogue, summary in samsum_ds:
    print(dialogue.numpy())
    print(summary.numpy())
    break

train_ds = (
    samsum_ds.map(
        lambda dialogue, summary: {"encoder_text": dialogue, "decoder_text": summary}
    )
    .batch(BATCH_SIZE)
    .cache()
)
train_ds = train_ds.take(NUM_BATCHES)
