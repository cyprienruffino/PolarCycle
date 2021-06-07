import tensorflow as tf

from filesystem.path import Paths
from models.abstractmodel import AbstractModel
from utils.base_configs import GPUConfig


class GPUModel(AbstractModel):
    def __init__(self, cfg: GPUConfig, paths: Paths, resume=None, epoch=0):
        self.cfg = cfg
        self.paths = paths
        self.start_epoch = epoch
        self.checkpoints_dir = self.paths.checkpoints_dir

        if cfg.num_gpu == 0:
            self.device_0 = "/cpu:0"
            self.device_1 = "/cpu:0"
        elif cfg.num_gpu == 1:
            self.device_0 = "/gpu:0"
            self.device_1 = "/gpu:0"
        else:
            self.device_0 = "/gpu:0"
            self.device_1 = "/gpu:1"

        tf.compat.v1.disable_eager_execution()
        proto = tf.compat.v1.ConfigProto()
        proto.gpu_options.allow_growth = True
        # Uncomment this line if you're using GTX 2080 Ti
        # proto.gpu_options.per_process_gpu_memory_fraction = 0.95
        proto.allow_soft_placement = True
        tf.compat.v1.experimental.output_all_intermediates(True)
        self.sess = tf.compat.v1.Session(config=proto)
        tf.compat.v1.keras.backend.set_session(self.sess)

        self.tape = tf.GradientTape()

        self.setup_datasets()
        self.build_models()

        self.global_step = tf.Variable(epoch, trainable=False)
        self.sess.run(tf.compat.v1.variables_initializer([self.global_step]))
        if resume is not None:
            self.resume_models(resume, epoch)

        self.create_objectives()

        self.create_optimizers()
        self.setup_logging()
        self.train()

    def reset_session(self):
        tf.compat.v1.reset_default_graph()