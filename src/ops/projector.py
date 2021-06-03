import tensorflow as tf

from ops.norm import norm


@tf.custom_gradient
def projector(Sd, var):
    S = tf.squeeze(Sd, axis=-1)

    norm2 = norm(S[:, :, :, 1:3], axis=-1, ord=2)
    Sstack = tf.stack([norm2, S[:, :, :, 1], S[:, :, :, 2]], axis=-1)

    sbool = tf.expand_dims(tf.cast((norm2 > S[:, :, :, 0]), tf.float32), axis=-1)

    Sproj = 0.5 * (1 + (S[:, :, :, 0:1] / (tf.expand_dims(norm2, axis=-1) + 1e-8))) * Sstack

    def grad(Sd):
        S_grad = tf.gradients(Sd, var) - tf.gradients(0.5 * (1 + (S[:, :, :, 0:1] / (tf.expand_dims(norm2, axis=-1) + 1e-8))) * Sstack, var)

        return sbool * (Sd - Sproj) * S_grad

    S_new = sbool * S + (1 - sbool) * Sproj

    return tf.reshape(S_new, tf.shape(Sd)), grad
