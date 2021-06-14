from base_configs.cyclegan_config import CycleGANConfig


class PolarCycleConfig(CycleGANConfig):

    def __init__(self, name):

        super(PolarCycleConfig, self).__init__(name)

        # Polar costs
        self.norm_AS = None
        self.conic_dist = None
        self.calibration_matrix = None
        self.inverse_calibration_matrix = None
        self.lmbda = None
        self.mu = None

