import hashlib
import importlib.util


def loadconfig(filepath, name=""):
    spec = importlib.util.spec_from_file_location("configs", filepath)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    return config_module.CustomConfig(name)


class AbstractConfig:

    def __init__(self, name):

        # Run metadata
        self.num_gpu = None
        self.name = name
        self.seed = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % (
            10 ** 8)

        # Training settings
        self.model = None
        self.batch_size = None
        self.epochs = None
        self.rgb_channels = None
        self.polar_channels = None
        self.dataset_size = None
        self.image_size = None
        self.pool_size = None
        self.cyc_factor = None

        # Polar costs
        self.norm_AS = None
        self.conic_dist = None
        self.calibration_matrix = None
        self.inverse_calibration_matrix = None

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