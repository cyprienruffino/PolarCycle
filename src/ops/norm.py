import tensorflow as tf


@tf.custom_gradient
def custom_norm(x, axis=-1, ord=2):
    y = tf.norm(x, axis=axis, ord=ord, keepdims=False)

    def grad(dy):
        return tf.expand_dims(dy, axis=-1) * x / (tf.expand_dims(y, axis=-1) + 1e-8)

    return y, grad


def norm(X, axis=-1, ord=2):
    return tf.map_fn(lambda x: custom_norm(x), X)
