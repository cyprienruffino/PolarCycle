import tensorflow as tf


def create_weight_histograms(model, name):
    __weightsumms = []
    for layer in model.layers:
        i = 0
        for vect in layer.trainable_weights:
            __weightsumms.append(tf.summary.histogram(name + '_' + layer.name + str(i), vect))
    return __weightsumms
