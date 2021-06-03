import numpy as np
import tensorflow as tf

import imageio

X = imageio.imread('imageio:chelsea.png')

A = 0.5 * np.array([[1, 1, 0],
                     [1, 0, 1],
                     [1, -1, 0],
                     [1, 0, -1]])

Ad = np.array([[1, 0, 1, 0],
                [1, 0, -1, 0],
                [0, 1, 0, -1]])

tf.cast()