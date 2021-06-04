# warnings.filterwarnings("ignore")

import os
import sys

from models.abstractmodel import AbstractModel
from utils.config import loadconfig, AbstractConfig
from utils.paths import Paths


def main():
    if len(sys.argv) < 4:
        print("Usage : python train.py config_file rgb_path polar_path [path_to_checkpoints] [epoch]")
        exit(1)

    configpath = sys.argv[1]
    dataA_path = sys.argv[2]
    dataB_path = sys.argv[3]

    if len(sys.argv) == 5:
        print("Usage : python train.py config_file rgb_path polar_path [path_to_checkpoints] [epoch]")
        exit(1)

    resume_path = None
    epoch = 0

    if len(sys.argv) == 6:
        resume_path = sys.argv[4]
        epoch = int(sys.argv[5])

    paths = Paths(configpath, dataA_path, dataB_path, resume_path)

    for run in paths.list_configs():
        cfg: AbstractConfig = loadconfig(run)
        model: AbstractModel = cfg.model(cfg, paths, resume_path, epoch)

        model.train()
        model.reset_session()


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
