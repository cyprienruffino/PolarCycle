import hashlib

from dataloaders.bitmap import BitmapDataloader
from deeplauncher.executors.gpu_v1 import GPUExecutorV1

from base_configs.cyclegan_config import CycleGANConfig
from models.cyclegan_base import CycleGANBase
from networks import cyclegan_disc
from networks import cyclegan_gen_9


class CustomConfig(CycleGANConfig):

    def __init__(self, name):
        super().__init__(name)

        # Run metadata
        self.num_gpu = 1
        self.name = name
        self.seed = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % (
                10 ** 8)

        # Training settings
        self.model = CycleGANBase
        self.executor = GPUExecutorV1
        self.dataloader = BitmapDataloader
        self.batch_size = 1
        self.epochs = 400
        self.dataA_channels = 3
        self.dataB_channels = 4
        self.dataset_size = 2485
        self.image_size = 500
        self.crop_size = 200
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
