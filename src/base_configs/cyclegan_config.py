from deeplauncher.base_configs.gpu_v1_config import GPUConfig


class CycleGANConfig(GPUConfig):

    def __init__(self, name):

        super(CycleGANConfig, self).__init__(name)

        # Run metadata
        self.debug = False

        # Training settings
        self.batch_size = None
        self.epochs = None
        self.dataA_channels = None
        self.dataB_channels = None
        self.dataset_size = None
        self.image_size = None
        self.crop_size = None
        self.pool_size = None
        self.cyc_factor = None
        self.learning_rate = None

        # Network setup
        self.genA = None
        self.genA_args = {}

        self.genB = None
        self.genB_args = {}

        self.discA = None
        self.discA_args = {}

        self.discB = None
        self.discB_args = {}