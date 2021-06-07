from utils.config import AbstractConfig


class AbstractModel:

    def __init__(self, cfg: AbstractConfig, rgb_path, polar_path, logs_dir, checkpoints_dir, resume=None, epoch=0):
        raise NotImplementedError

    def setup_datasets(self, rgb_path, polar_path):
        raise NotImplementedError

    def build_models(self):
        raise NotImplementedError

    def create_objectives(self):
        raise NotImplementedError

    def create_optimizers(self):
        raise NotImplementedError

    def setup_logging(self, logs_dir):
        raise NotImplementedError

    def setup_checkpoints(self, checkpoints_dir):
        raise NotImplementedError

    def checkpoint_models(self, epoch):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def reset_session(self):
        raise NotImplementedError
