import tensorflow as tf


def from_pool(pool, real, pool_size, batch_size=1):
    p = tf.random.uniform((1,), 0, 2, dtype=tf.int32)
    num = tf.random.uniform((1,), 0, pool_size, dtype=tf.int32)[0]
    return tf.cond(tf.equal(p, 0)[0],
                   lambda: real,
                   lambda: pool[num:num + batch_size])


def update_pool(pool, real, pool_size, batch_size=1):
    num = tf.random.uniform((1,), 0, pool_size, dtype=tf.int32)[0]
    return tf.compat.v1.assign(pool[num:num + batch_size], real)