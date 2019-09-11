import hashlib


class AbstractConfig:

    def __init__(self, name):

        # Run metadata
        self.num_gpu = None
        self.name = name
        self.seed = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % (
            10 ** 8)

        # Training settings
        self.batch_size = None
        self.epochs = None
        self.dataset_size = None
        self.image_size = None
        self.cyc_factor = None

        # Polar costs
        self.norm_AS = None
        self.conic_dist = None
        self.lmbda = None
        self.mu = None

        # Optimizers
        self.optimizer = None
        self.optimizer_args = {}

        # Network setup
        self.genA = None
        self.genA_args = {}

        self.genB = None
        self.genB_args = {}

        self.discA = None
        self.discA_args = {}

        self.discB = None
        self.discB_args = {}

