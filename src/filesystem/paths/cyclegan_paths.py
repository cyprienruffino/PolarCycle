from filesystem.path import Paths


class CycleGANPaths(Paths):
    def __init__(self, configpath: str, dataA_path: str, dataB_path: str, resume_path: str = None):
        super(CycleGANPaths, self).__init__(configpath, resume_path)

        self.dataB_path: str = dataB_path
        self.dataA_path: str = dataA_path
