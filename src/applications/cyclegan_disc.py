import tensorflow as tf
from tensorflow import keras as k
from tensorflow.python.keras import layers as kl

from layers import InstanceNormalization


def create_network(inp, channels, name):
    with tf.name_scope("Disc"):
        inp_layer = kl.Input((None, None, channels), tensor=inp)

        # Discriminator
        layer = kl.Conv2D(64, 4, padding="same", strides=2)(inp_layer)
        layer = InstanceNormalization()(layer)
        layer = kl.LeakyReLU()(layer)

        layer = kl.Conv2D(128, 4, padding="same", strides=2)(layer)
        layer = InstanceNormalization()(layer)
        layer = kl.LeakyReLU()(layer)

        layer = kl.Conv2D(256, 4, padding="same", strides=2)(layer)
        layer = InstanceNormalization()(layer)
        layer = kl.LeakyReLU()(layer)

        layer = kl.Conv2D(512, 4, padding="same", strides=2)(layer)
        layer = InstanceNormalization()(layer)
        layer = kl.LeakyReLU()(layer)

        D_out = kl.Conv2D(1, 4, activation="sigmoid", padding="same")(layer)

        model = k.Model(inputs=inp_layer, outputs=D_out)
    return model
