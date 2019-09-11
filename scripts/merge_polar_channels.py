import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import cv2
import sys
import fnmatch
import os
import progressbar
import numpy as np

res = (500, 500)


def preprocess_dataset(polar_path):
    print("Preprocessing polarimetric images")

    os.mkdir("polar_merged")
    ch = fnmatch.filter(os.listdir(polar_path), '*' + '*I0.png')
    ch = np.array(ch)
    print("len(ch) : ", len(ch))

    i = 0
    for _ in progressbar.progressbar(range(len(ch))):
        index, _ = ch[i].split("_")
        I0 = fnmatch.filter(os.listdir(polar_path), str(index) + '*I0.png')
        I45 = fnmatch.filter(os.listdir(polar_path), str(index) + '*I45.png')
        I90 = fnmatch.filter(os.listdir(polar_path), str(index) + '*I90.png')
        I135 = fnmatch.filter(os.listdir(polar_path), str(index) + '*I135.png')

        if I0 is not [] and I45 is not [] and I90 is not [] and I135 is not []:
            i += 1
            img_I0 = cv2.imread(os.path.join(polar_path, I0[0]))
            img_I45 = cv2.imread(os.path.join(polar_path, I45[0]))
            img_I90 = cv2.imread(os.path.join(polar_path, I90[0]))
            img_I135 = cv2.imread(os.path.join(polar_path, I135[0]))

            merged = cv2.merge((img_I0[:,:,0], img_I45[:,:,0], img_I90[:,:,0], img_I135[:,:,0]))
            cv2.imwrite(os.path.join("polar_merged", str(i) + ".png"), merged)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        preprocess_dataset(sys.argv[1])
    else:
        print("Usage : merge_polar_channels.py polar_data_path")
