import hashlib
import tensorflow as tf

from applications import cyclegan_disc, cyclegan_gen_9
from configs.abstract_config import AbstractConfig


class CustomConfig(AbstractConfig):

    def __init__(self, name):
        super().__init__(name)

        # Run metadata
        self.num_gpu = 2
        self.name = name
        self.seed = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % (
            10 ** 8)

        # Training settings
        self.batch_size = 2
        self.epochs = 400
        self.dataset_size = 100
        self.image_size = 500
        self.cyc_factor = 10

        self.norm_AS = True
        self.conic_dist = True

        self.lmbda = 1
        self.mu = 1

        # Optimizers
        self.optimizer = tf.train.AdamOptimizer
        self.optimizer_args = {
            "learning_rate": 0.0002
        }

        # Network setup
        self.genA = cyclegan_gen_9.create_network
        self.genA_args = {"channels_out": 3, "name": "GenA"}

        self.genB = cyclegan_gen_9.create_network
        self.genB_args = {"channels_out": 4, "name": "GenB"}

        self.discA = cyclegan_disc.create_network
        self.discA_args = {"channels": 3, "name": "DiscA"}

        self.discB = cyclegan_disc.create_network
        self.discB_args = {"channels": 4, "name": "DiscB"}
