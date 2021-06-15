import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os
import sys

import cv2
import progressbar
import tensorflow as tf

from deeplauncher.data_processing import from_image_files


def main(input_path, output_path):
    size = len(list(filter(lambda x: ".png" in x, os.listdir(input_path))))
    dataset = from_image_files.oneshot_iterator(input_path, size, 1, 4)

    with tf.Session() as sess:
        image_iter = dataset.get_next()
        projector = projector(image_iter)
        sess.run(tf.compat.v1.initialize_all_variables())
        for i in progressbar.progressbar(range(size)):
            projected = sess.run(projector)
            cv2.imwrite(os.path.join(output_path, str(i) + ".png"), projected)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python project_images.py input_path output_path")
        exit()
    main(sys.argv[1], sys.argv[2])
