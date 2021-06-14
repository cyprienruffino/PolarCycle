import tensorflow as tf

from filesystem import CycleGANPaths
from models.polar_cycle import PolarCycle
from ops import projector
from base_configs import PolarCycleConfig

DEBUG = True


class PolarCycleProjectorDiff(PolarCycle):

    def __init__(self, cfg: PolarCycleConfig, paths: CycleGANPaths, resume=None, epoch=0):
        super(PolarCycleProjectorDiff, self).__init__(cfg, paths, resume, epoch)
        self.cfg = cfg
        self.paths = paths

    def create_objectives(self):
        super(PolarCycleProjectorDiff, self).create_objectives()

        Ad = tf.stack([tf.stack([self.A_dagger] * 200, axis=1)] * 200, axis=1)

        S = Ad @ tf.expand_dims(self.x, axis=-1)

        prox_dist = tf.reduce_mean((S - projector(S, self.genB.variables)) ** 2)

        self.prox_dist = self.cfg.mu * prox_dist
        self.gB_obj += self.prox_dist

        if self.cfg.debug:
            self.dbggt(self.x, "## x")
            self.dbggt(Ad, "## Ad matrix")
            self.dbggt(S, "## S matrix")
            self.dbggt(prox_dist, "prox_dist")

    def setup_logging(self):
        super(PolarCycleProjectorDiff, self).setup_logging()
        self.summaries = tf.compat.v1.summary.merge([self.summaries, tf.compat.v1.summary.scalar("Proximal_dist", self.prox_dist)])
