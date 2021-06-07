import numpy as np
import tensorflow as tf

import imageio

from ops.projector import projector


class Projectortest(tf.test.TestCase):
    def setUp(self):
        super(Projectortest, self).setUp()
        self.X = tf.Variable(imageio.imread('imageio:chelsea.png'))
        self.A = tf.Variable(0.5 * np.array([[1, 1, 0],
                            [1, 0, 1],
                            [1, -1, 0],
                            [1, 0, -1]]))

        self.Ad = tf.Variable(np.array([[1, 0, 1, 0],
                       [1, 0, -1, 0],
                       [0, 1, 0, -1]]))

    def tearDown(self):
        pass

    def test_projector(self):
        S = self.Ad @ self.X
        
        with tf.GradientTape as tape:
            Sproj = projector(S, self.genB.variables)

        self.assertEqual(tf.reduce_sum(tf.math.is_nan(Sproj)), 0)

        grads = tape.gradient(tf.reduce_sum(tf.square(Sproj - S)), self.X)
        self.assertEqual(tf.reduce_sum(tf.math.is_nan(grads)), 0)


if __name__ == '__main__':
    tf.test.main()

