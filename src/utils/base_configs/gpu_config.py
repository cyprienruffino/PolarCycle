from utils.config import AbstractConfig


class GPUConfig(AbstractConfig):

    def __init__(self, name):

        super(GPUConfig, self).__init__(name)

        # Run metadata
        self.num_gpu = None

