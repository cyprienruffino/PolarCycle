import tensorflow as tf

from filesystem import CycleGANPaths
from src.models import PolarCycle
from src.base_configs import PolarCycleConfig


class PolarCycleConicExp(PolarCycle):
    def __init__(self, cfg: PolarCycleConfig, paths: CycleGANPaths, resume=None, epoch=0):
        super(PolarCycleConicExp, self).__init__(cfg, paths, resume, epoch)
        self.cfg = cfg
        self.paths = paths

    def create_objectives(self):
        super(PolarCycleConicExp, self).create_objectives()

        phi = ((tf.sqrt(self.S1 ** 2 + self.S2 ** 2) / self.S0) - 1)
        reg = tf.nn.relu(tf.exp(1 - phi))

        conic_dist = tf.norm(reg) / (self.cfg.image_size ** 2)

        self.conic_dist = self.cfg.mu * conic_dist
        self.gB_obj += self.conic_dist

    def setup_logging(self, logs_dir):
        super(PolarCycleConicExp, self).setup_logging(logs_dir)
        self.summaries = tf.summary.merge([self.summaries, tf.summary.scalar("Conic_dist", self.conic_dist)])
