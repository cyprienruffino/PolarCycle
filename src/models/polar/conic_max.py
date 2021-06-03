import tensorflow as tf

from models.polar.polar_cycle import PolarCycle


class PolarCycleConicMax(PolarCycle):
    def create_objectives(self):
        super(PolarCycleConicMax, self).create_objectives()

        phi = ((tf.sqrt(self.S1 ** 2 + self.S2 ** 2) / self.S0) - 1)
        phi = tf.nn.relu(phi)

        conic_dist = tf.norm(phi) / (self.cfg.image_size ** 2)

        self.conic_dist = self.cfg.mu * conic_dist
        self.gB_obj += self.conic_dist

    def setup_logging(self, logs_dir):
        super(PolarCycleConicMax, self).setup_logging(logs_dir)
        self.summaries = tf.summary.merge([self.summaries, tf.summary.scalar("Conic_dist", self.conic_dist)])
