import os
import sys

import cv2
import progressbar
import tensorflow as tf

from data_processing.from_image_files import iterator_gen
from layers import *
from utils.postprocessing import revert_normalization


def main(checkpoint_path, files_path, output_path, channels):
    mod = tf.compat.v1.keras.models.load_model(checkpoint_path, custom_objects={CUSTOM_OBJECTS})

    inp = iterator_gen(files_path, channels)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    with tf.compat.v1.Session as sess:
        for name in progressbar.ProgressBar()(os.listdir(files_path)):
            out = sess.run(mod(inp))
            cv2.imwrite(os.path.join(output_path, name), revert_normalization(out[0]))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python generate.py checkpoint_path files_path output_path, nb_channels")
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
