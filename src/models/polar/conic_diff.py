import tensorflow as tf

from models.polar.polar_cycle import PolarCycle


class PolarCycleConicDiff(PolarCycle):
    def create_objectives(self):
        super(PolarCycleConicDiff, self).create_objectives()

        reg = tf.nn.relu(self.S1**2 + self.S2**2 - self.S0**2)

        conic_dist = tf.reduce_mean(reg)

        self.conic_dist = self.cfg.mu * conic_dist
        self.gB_obj += self.conic_dist

    def setup_logging(self, logs_dir):
        super(PolarCycleConicDiff, self).setup_logging(logs_dir)
        self.summaries = tf.compat.v1.summary.merge([self.summaries, tf.compat.v1.summary.scalar("Conic_dist", self.conic_dist)])
