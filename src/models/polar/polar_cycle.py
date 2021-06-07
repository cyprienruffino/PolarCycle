import tensorflow as tf

from models.cyclegan_base import CycleGANBase
from ops.tools import to_256

from tensorflow.python.framework import function


@function.Defun(tf.float32, tf.float32)
def norm_grad(x, dy):
    return dy * (x / tf.norm(x))


@function.Defun(tf.float32, grad_func=norm_grad)
def norm(x):
    return tf.norm(x)


class PolarCycle(CycleGANBase):

    def build_S_I(self):
        self.x = to_256(self.out_gB)

        self.img_I0 = self.x[:, :, :, 0]
        self.img_I45 = self.x[:, :, :, 1]
        self.img_I90 = self.x[:, :, :, 2]
        self.img_I135 = self.x[:, :, :, 3]

        self.S0 = self.img_I0 + self.img_I90
        self.S1 = self.img_I0 - self.img_I90
        self.S2 = self.img_I45 - self.img_I135

        self.S = tf.cast(
            tf.reshape(tf.stack([self.S0, self.S1, self.S2]), shape=[self.cfg.batch_size, 3, -1]),
            tf.float32
        )

        self.A = tf.cast(self.cfg.calibration_matrix, tf.float32)
        self.A_dagger = tf.cast(self.cfg.inverse_calibration_matrix, tf.float32)

    def create_objectives(self):
        super(PolarCycle, self).create_objectives()
        self.build_S_I()

        I = tf.cast(tf.reshape(self.x, shape=[self.cfg.batch_size, self.cfg.dataB_channels, -1]), tf.float32)

        AS = tf.matmul(self.A, self.S)
        delta1 = I - AS
        norm_AS = norm(delta1) / ((self.cfg.image_size ** 2) + 1e-8)

        self.norm_AS_obj = self.cfg.lmbda * norm_AS
        self.gB_obj += self.norm_AS_obj

        self.__dbggt(I, "## I")
        self.__dbggt(AS, "## AS")
        self.__dbggt(delta1, "## delta1")
        self.__dbggt(norm_AS, "normAS")
        self.__dbggt(self.S, "##S")

    def setup_logging(self, logs_dir):
        super(PolarCycle, self).setup_logging(logs_dir)
        self.summaries = tf.compat.v1.summary.merge([self.summaries, tf.compat.v1.summary.scalar("IminAS_norm", self.norm_AS_obj)])
