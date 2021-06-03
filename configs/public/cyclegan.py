import hashlib

from networks import cyclegan_disc, cyclegan_gen_9
from utils.config import AbstractConfig
from models.cyclegan_base import CycleGANBase


class CustomConfig(AbstractConfig):

    def __init__(self, name):
        super().__init__(name)

        # Run metadata
        self.num_gpu = 1
        self.name = name
        self.seed = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % (
            10 ** 8)

        # Training settings
        self.model = CycleGANBase
        self.batch_size = 1
        self.epochs = 400
        self.rgb_channels = 3
        self.polar_channels = 4
        self.dataset_size = 2485
        self.image_size = 200
        self.cyc_factor = 10
        self.pool_size = 50
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
