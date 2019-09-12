import os
from functools import partial

import tensorflow as tf

res = [500, 500]


def _parse(filename, channels):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=channels)
    image_resized = tf.image.resize(image_decoded, res)
    return image_resized


def iterator(path, dataset_size, batch_size, channels):
    filenames = list(map(lambda p: os.path.join(path, p), os.listdir(path)))[:dataset_size]
    files = tf.constant(filenames)
    dataset = tf.data.Dataset.from_tensor_slices(files)\
        .map(partial(_parse, channels=channels)) \
        .map(lambda x: (x / 128) - 1) \
        .shuffle(buffer_size=1000) \
        .batch(batch_size) \
        .repeat()

    return dataset.make_initializable_iterator()
