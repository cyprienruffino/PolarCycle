import tensorflow as tf

from filesystem import CycleGANPaths
from models import CycleGANBase
from base_configs import PolarCycleConfig


class ConicDiff(CycleGANBase):
    def __init__(self, cfg: PolarCycleConfig, paths: CycleGANPaths, resume=None, epoch=0):
        super(ConicDiff, self).__init__(cfg, paths, resume, epoch)
        self.cfg = cfg
        self.paths = paths

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

    def setup_logging(self):
        super(ConicDiff, self).setup_logging()
        self.summaries = tf.compat.v1.summary.merge([self.summaries, tf.summary.scalar("Conic_dist", self.conic_dist)])
