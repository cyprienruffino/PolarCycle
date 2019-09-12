import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os
import shutil
import sys

from utils.config import loadconfig
from PolarCycle import PolarCycle


def do_run(filepath, rgb_path, polar_path, extension=""):
    name = filepath.split('/')[-1].replace('.py', '') + extension
    config = loadconfig(filepath)

    print("\nRunning " + name + "\n")

    if not os.path.exists("runs"):
        os.mkdir("runs")
    os.mkdir(os.path.join("runs", name))
    shutil.copy2(filepath, os.path.join("runs", name, "config.py"))
    checkpoints_dir = os.path.join("runs", name, "checkpoints")
    logs_dir = os.path.join("runs", name, "logs")

    model = PolarCycle(config, rgb_path, polar_path, logs_dir, checkpoints_dir)
    model.train()
    model.reset_session()


def main():
    if len(sys.argv) < 4:
        print("Usage : python train_polarcycle.py config_file rgb_path polar_path")
        exit(1)

    filepath = sys.argv[1]
    rgb_path = sys.argv[2]
    polar_path = sys.argv[3]

    if os.path.isfile(filepath):
        do_run(filepath, rgb_path, polar_path)
    else:
        for run in os.listdir(filepath):
            if ".py" in run:
                do_run(filepath + "/" + run, rgb_path, polar_path)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
