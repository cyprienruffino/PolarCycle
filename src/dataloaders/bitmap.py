import os
from functools import partial

import tensorflow as tf

from deeplauncher.dataloaders.abstractdataloader import AbstractDataloader


class BitmapDataloader(AbstractDataloader):

    @staticmethod
    def _parse(filename, channels, resolution):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_png(image_string, channels=channels)
        image_resized = tf.image.resize(image_decoded, resolution)
        return image_resized

    @staticmethod
    def _flip(x):
        # x = tf.compat.v1.image.random_flip_left_right(x)
        return x

    @staticmethod
    def _crop(x, npx):
        shape = x.shape
        topleft_x = tf.random.uniform((1,), minval=0, maxval=(shape[0] - npx), dtype=tf.int32)
        topleft_y = tf.random.uniform((1,), minval=npx, maxval=(shape[1] - npx), dtype=tf.int32)
        return tf.image.crop_to_bounding_box(x, topleft_y[0], topleft_x[0], npx, npx)

    @staticmethod
    def _add_noise(x):
        return x + tf.random.normal(tf.shape(x), 0, 1e-8)

    def __init__(self, dataset_path, batch_size, channels, image_size, crop_size):
        super(BitmapDataloader, self).__init__()
        self.crop_size = crop_size
        self.image_size = image_size
        self.channels = channels
        self.batch_size = batch_size
        self.dataset_path = dataset_path

    def train(self):
        super(BitmapDataloader, self).train()
        filenames = list(map(lambda p: os.path.join(self.dataset_path, p), os.listdir(self.dataset_path)))
        files = tf.constant(filenames)

        dataset = tf.compat.v1.data.Dataset.from_tensor_slices(files) \
            .map(partial(self._parse, channels=self.channels, resolution=(self.image_size,self.image_size))) \
            .map(lambda x: (x / 127.5) - 1) \
            .map(self._flip) \
            .map(partial(self._crop, npx=self.crop_size)) \
            .map(self._add_noise) \
            .shuffle(buffer_size=500) \
            .batch(self.batch_size) \

        return dataset

    def test(self):
        return None

    def valid(self):
        return None

    """
    def iterator_gen(self, path, channels):
        filenames = list(map(lambda p: os.path.join(path, p), os.listdir(path)))
        files = tf.constant(filenames)

        dataset = tf.compat.v1.data.Dataset.from_tensor_slices(files) \
            .map(partial(self._parse, channels=channels)) \
            .map(lambda x: (x / 127.5) - 1) \

        return tf.compat.v1.data.make_one_shot_iterator(dataset)
    """
