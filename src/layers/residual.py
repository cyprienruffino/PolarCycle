from tensorflow.python.keras import layers as kl

from layers import InstanceNormalization


def ResidualBlock(filters, nb_layers=2, kernel_size=3, normalization="batchnorm"):
    def _resblock(inp):
        layer = inp
        if normalization is "batchnorm":
            normalizer = kl.BatchNormalization
        elif normalization is "instancenorm":
            normalizer = InstanceNormalization
        else:
            raise Exception(normalization + "is not a valid normalization type. These are 'batchnorm' and "
                                            "'instancenorm'")

        for i in range(nb_layers - 1):
            layer = kl.Conv2D(filters, kernel_size=kernel_size, padding="same")(layer)
            layer = normalizer()(layer)

        layer = kl.Conv2D(filters, kernel_size=kernel_size, padding="same")(layer)
        layer = normalizer()(layer)
        layer = kl.add([layer, inp])
        return kl.Activation("relu")(layer)
    return _resblock



