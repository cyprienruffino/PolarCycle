import tensorflow as tf

from models.polar.polar_cycle import PolarCycle
from ops import projector

DEBUG = True


class PolarCycleProjectorDiff(PolarCycle):
    def create_objectives(self):
        super(PolarCycleProjectorDiff, self).create_objectives()

        Ad = tf.stack([tf.stack([self.A_dagger] * 200, axis=1)] * 200, axis=1)

        S = Ad @ tf.expand_dims(self.x, axis=-1)

        prox_dist = tf.reduce_mean((S - projector(S, self.genB.variables)) ** 2)

        self.prox_dist = self.cfg.mu * prox_dist
        self.gB_obj += self.prox_dist

        if DEBUG:
            self.dbggt(self.x, "## x")
            self.dbggt(Ad, "## Ad matrix")
            self.dbggt(S, "## S matrix")
            self.dbggt(prox_dist, "prox_dist")

    def setup_logging(self, logs_dir):
        super(PolarCycleProjectorDiff, self).setup_logging(logs_dir)
        self.summaries = tf.summary.merge([self.summaries, tf.summary.scalar("Proximal_dist", self.prox_dist)])
