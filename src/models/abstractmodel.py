from utils.base_configs.cyclegan_config import CycleGANConfig
from filesystem.path import Paths


class AbstractModel:

    def __init__(self, cfg: CycleGANConfig, paths: Paths, resume=None, epoch=0):
        raise NotImplementedError

    def setup_datasets(self):
        raise NotImplementedError

    def build_models(self):
        raise NotImplementedError

    def create_objectives(self):
        raise NotImplementedError

    def create_optimizers(self):
        raise NotImplementedError

    def setup_logging(self):
        raise NotImplementedError

    def checkpoint_models(self, epoch):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def reset_session(self):
        raise NotImplementedError

    def resume_models(self, resume, epoch):
        raise NotImplementedError

