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
EPOCHS = 1
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

preprocessor = keras_hub.models.BartSeq2SeqLMPreprocessor.from_preset(
    "bart_base_en",
    encoder_sequence_length=MAX_ENCODER_SEQUENCE_LENGTH,
    decoder_sequence_length=MAX_DECODER_SEQUENCE_LENGTH,
)
bart_lm = keras_hub.models.BartSeq2SeqLM.from_preset(
    "bart_base_en", preprocessor=preprocessor
)

bart_lm.summary()

bart_lm.fit(train_ds, epochs=EPOCHS)

def generate_text(model, input_text, max_length=200, print_time_taken=False):
    start = time.time()
    output = model.generate(input_text, max_length=max_length)
    end = time.time()
    print(f"Total Time Elapsed: {end - start:.2f}s")
    return output

val_ds = tfds.load("samsum", split="validation", as_supervised=True)
val_ds = val_ds.take(100)

dialogues = []
ground_truth_summaries = []
for dialogue, summary in val_ds:
    dialogues.append(dialogue.numpy())
    ground_truth_summaries.append(summary.numpy())

_ = generate_text(bart_lm, "sample text", max_length=MAX_GENERATION_LENGTH)

generated_summaries = generate_text(
    bart_lm,
    val_ds.map(lambda dialogue, _: dialogue).batch(8),
    max_length=MAX_GENERATION_LENGTH,
    print_time_taken=True,
)

for dialogue, generated_summary, ground_truth_summary in zip(
    dialogues[:5], generated_summaries[:5], ground_truth_summaries[:5]
):
    print("Diálogo:", dialogue)
    print("Sumário:", generated_summary)
    print("Sumário de verdade:", ground_truth_summary)
    print("=============================")
