import os
import tensorflow as tf

res = [500, 500]


def _parse(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string)
    image_resized = tf.image.resize_images(image_decoded, res)
    return image_resized


def get_iterator(path, dataset_size, batch_size):
    filenames = list(map(lambda p: os.path.join(path, p), os.listdir(path)))[:dataset_size]
    files = tf.constant(filenames)
    dataset = tf.data.Dataset.from_tensor_slices(files)\
        .map(_parse) \
        .map(lambda x: (x / 128) - 1) \
        .repeat() \
        .shuffle(buffer_size=10000) \
        .batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    return iterator
