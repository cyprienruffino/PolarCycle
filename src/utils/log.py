import progressbar
import tensorflow as tf


def create_weight_histograms(model, name):
    __weightsumms = []
    for layer in model.layers:
        i = 0
        for vect in layer.trainable_weights:
            __weightsumms.append(tf.summary.histogram(name + '_' + layer.name + str(i), vect))
    return __weightsumms


def custom_bar(epoch, epoch_iters):
    return progressbar.ProgressBar(widgets=[
        "Epoch " + str(epoch), ' ',
        progressbar.Percentage(), ' ',
        progressbar.SimpleProgress(format='(%s)' % progressbar.SimpleProgress.DEFAULT_FORMAT),
        progressbar.Bar(), ' ',
        progressbar.Timer(), ' ',
        progressbar.AdaptiveETA()
    ], maxvalue=epoch_iters, redirect_stdout=True)


def detect_nan(tensor, name):
    return tf.cond(tf.math.reduce_any((tf.math.is_nan(tensor))), lambda: tf.print("NaN in disc tensor", name),
                   lambda: tf.constant(False))
