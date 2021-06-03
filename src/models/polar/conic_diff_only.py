import tensorflow as tf

from models.cyclegan_base import CycleGANBase


class ConicDiff(CycleGANBase):
    def build_S_I(self):
        self.x = (self.out_gB + 1) * 127.5

        self.img_I0 = self.x[:, :, :, 0]
        self.img_I45 = self.x[:, :, :, 1]
        self.img_I90 = self.x[:, :, :, 2]
        self.img_I135 = self.x[:, :, :, 3]

        self.S0 = self.img_I0 + self.img_I90
        self.S1 = self.img_I0 - self.img_I90
        self.S2 = self.img_I45 - self.img_I135

    def create_objectives(self):
        super(ConicDiff, self).create_objectives()
        self.build_S_I()
        reg = tf.nn.relu(self.S1**2 + self.S2**2 - self.S0**2)

        conic_dist = tf.norm(reg) / (self.cfg.image_size ** 2)

        self.conic_dist = self.cfg.mu * conic_dist
        self.gB_obj += self.conic_dist

    def setup_logging(self, logs_dir):
        super(ConicDiff, self).setup_logging(logs_dir)
        self.summaries = tf.summary.merge([self.summaries, tf.summary.scalar("Conic_dist", self.conic_dist)])
