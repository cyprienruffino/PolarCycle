import os
import shutil


class Paths:
    def __init__(self, configpath: str, dataA_path: str, dataB_path: str, resume_path: str = None):
        self.resume_path: str = resume_path
        self.dataB_path: str = dataB_path
        self.dataA_path: str = dataA_path
        self.configpath: str = configpath

        self.name: str = self.configpath.split('/')[-1].replace('.py', '')

        self.checkpoints_dir: str = os.path.join("runs", self.name, "checkpoints")
        self.logs_dir = os.path.join("runs", self.name, "logs")

    def setup_paths(self):
        if not os.path.exists("runs"):
            os.mkdir("runs")

        if self.resume_path is None:
            os.mkdir(os.path.join("runs", self.name))
            os.mkdir(self.checkpoints_dir)
            shutil.copy2(self.configpath, os.path.join("runs", self.name, "config.py"))

    def list_configs(self):
        if os.path.isfile(self.configpath):
            return [self.configpath]
        else:
            return [
                self.configpath + os.path.sep + cfg for cfg in
                filter(lambda x: ".py" in x, os.listdir(self.configpath))
            ]
