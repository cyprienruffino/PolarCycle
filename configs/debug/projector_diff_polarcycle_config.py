import hashlib

import numpy as np

from src.networks import cyclegan_gen_9, cyclegan_disc
from src.base_configs import CycleGANConfig
from src.models.conic_proximal import PolarCycleProjectorDiff


class CustomConfig(CycleGANConfig):

    def __init__(self, name):
        super().__init__(name)

        # Run metadata
        self.num_gpu = 2
        self.name = name
        self.seed = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % (
                10 ** 8)

        # Training settings
        self.model = PolarCycleProjectorDiff
        self.batch_size = 1
        self.epochs = 400
        self.dataA_channels = 3
        self.dataB_channels = 4
        self.dataset_size = 2480
        self.image_size = 200
        self.cyc_factor = 10
        self.pool_size = 50
        self.calibration_matrix = 0.5 * np.array([[[1, 1, 0],
                                                   [1, 0, 1],
                                                   [1, -1, 0],
                                                   [1, 0, -1]]] * self.batch_size)

        self.inverse_calibration_matrix = np.array([[[1, 0, 1, 0],
                                                     [1, 0, -1, 0],
                                                     [0, 1, 0, -1]]] * self.batch_size)

        self.lmbda = 1
        self.mu = 0.01

        self.learning_rate = 0.0002

        # Network setup
        self.genA = cyclegan_gen_9.create_network
        self.genA_args = {"channels_out": 3, "name": "GenA"}

        self.genB = cyclegan_gen_9.create_network
        self.genB_args = {"channels_out": 4, "name": "GenB"}

        self.discA = cyclegan_disc.create_network
        self.discA_args = {"channels": 3, "name": "DiscA"}

        self.discB = cyclegan_disc.create_network
        self.discB_args = {"channels": 4, "name": "DiscB"}
