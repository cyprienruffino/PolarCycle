import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os
import shutil
import sys

import progressbar

from configs.abstract_config import AbstractConfig
from datasets import from_images
from utils.config import loadconfig
from utils.log import create_weight_histograms

import numpy as np
import tensorflow as tf


def custom_bar(epoch, epoch_iters):
    return progressbar.ProgressBar(widgets=[
        "Epoch " + str(epoch), ' ',
        progressbar.Percentage(), ' ',
        progressbar.SimpleProgress(format='(%s)' % progressbar.SimpleProgress.DEFAULT_FORMAT),
        progressbar.Bar(), ' ',
        progressbar.Timer(), ' ',
        progressbar.AdaptiveETA()
    ], maxvalue=epoch_iters, redirect_stdout=True)


class PolarCycle:

    def __init__(self, cfg: AbstractConfig, rgb_path, polar_path, logs_dir, checkpoints_dir, epoch=0):
        self.cfg = cfg
        self.epoch = epoch

        if cfg.num_gpu == 0:
            self.device_0 = "/cpu:0"
            self.device_1 = "/cpu:0"
        elif cfg.num_gpu == 1:
            self.device_0 = "/gpu:0"
            self.device_1 = "/gpu:0"
        else:
            self.device_0 = "/gpu:0"
            self.device_1 = "/gpu:1"

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        self.__setup_datasets(rgb_path, polar_path)
        self.__build_models()
        self.__create_objectives()
        self.__create_optimizers()
        self.__setup_logging(logs_dir)
        self.__setup_checkpoints(checkpoints_dir)

        self.sess.run(tf.global_variables_initializer())

    def __setup_datasets(self, rgb_path, polar_path):
        iterA = from_images.get_iterator(rgb_path, self.cfg.dataset_size, self.cfg.batch_size, self.cfg.rgb_channels)
        iterB = from_images.get_iterator(polar_path, self.cfg.dataset_size, self.cfg.batch_size,
                                         self.cfg.polar_channels)

        self.sess.run(iterA.initializer)
        self.sess.run(iterB.initializer)

        self.inputA = iterA.get_next()
        self.inputB = iterB.get_next()

    def __build_models(self):
        # A: RGB, B: Pol
        with tf.device(self.device_0):
            self.genA = self.cfg.genA(self.inputB, **self.cfg.genA_args)
            self.discA = self.cfg.discA(self.inputA, **self.cfg.discA_args)
        self.out_gA = self.genA.output
        self.out_dA = self.discA.output

        with tf.device(self.device_1):
            self.genB = self.cfg.genB(self.inputA, **self.cfg.genB_args)
            self.discB = self.cfg.discB(self.inputB, **self.cfg.discB_args)
        self.out_gB = self.genB.output
        self.out_dB = self.discB.output

        with tf.device(self.device_0):
            with tf.name_scope("cycA"):
                self.cycA = self.genA(self.out_gB)
            with tf.name_scope("ganA"):
                self.ganA = self.discA(self.out_gA)

        with tf.device(self.device_1):
            with tf.name_scope("cycB"):
                self.cycB = self.genB(self.out_gA)
            with tf.name_scope("ganB"):
                self.ganB = self.discB(self.out_gB)

    def __create_objectives(self):
        # Objectives
        with tf.device(self.device_0):
            with tf.name_scope("gA_objective"):
                self.ganA_obj = tf.reduce_mean(tf.squared_difference(self.ganA, 1))
                self.cycA_obj = tf.reduce_mean(tf.abs(self.inputA - self.cycA))

                self.gA_obj = self.ganA_obj + self.cfg.cyc_factor * self.cycA_obj

            with tf.name_scope("dA_objective"):
                self.dA_obj = (tf.reduce_mean(tf.square(self.ganA)) + tf.reduce_mean(
                    tf.math.squared_difference(self.discA.output, 1))) / 2

        with tf.device(self.device_1):
            with tf.name_scope("gB_objective"):
                self.ganB_obj = tf.reduce_mean(tf.squared_difference(self.ganB, 1))
                self.cycB_obj = tf.reduce_mean(tf.abs(self.inputB - self.cycB))

                self.gB_obj = self.ganB_obj + self.cfg.cyc_factor * self.cycB_obj

                x = (self.out_gB + 1) * 128

                img_I0 = x[:, :, :, 0]
                img_I45 = x[:, :, :, 1]
                img_I90 = x[:, :, :, 2]
                img_I135 = x[:, :, :, 3]

                S0 = img_I0 + img_I90
                S1 = img_I0 - img_I90
                S2 = img_I45 - img_I135

                if self.cfg.norm_AS:
                    A = tf.cast(0.5 * np.array([[[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1]]] * self.cfg.batch_size),
                                tf.float32)
                    S = tf.cast(
                        tf.reshape(tf.stack([S0, S1, S2]), shape=[self.cfg.batch_size, self.cfg.rgb_channels, -1]),
                        tf.float32)
                    I = tf.cast(tf.reshape(x, shape=[self.cfg.batch_size, self.cfg.polar_channels, -1]), tf.float32)

                    AS = tf.matmul(A, S)
                    delta1 = I - AS
                    norm_AS = tf.norm(delta1) / (tf.norm(I) + tf.norm(AS))

                    self.norm_AS_obj = self.cfg.lmbda * norm_AS
                    self.gB_obj += self.norm_AS_obj
                else:
                    self.norm_AS_obj = tf.constant(0)

                if self.cfg.conic_dist:
                    phi = ((tf.sqrt(S1 ** 2 + S2 ** 2) / S0) - 1)
                    phi = tf.maximum(phi, 0)

                    conic_dist = tf.norm(phi) / (self.cfg.image_size ** 2)

                    self.conic_dist = self.cfg.mu * conic_dist
                    self.gB_obj += self.conic_dist
                else:
                    self.conic_dist = tf.constant(0)

            with tf.name_scope("dB_objective"):
                self.dB_obj = (tf.reduce_mean(tf.square(self.ganB)) + tf.reduce_mean(
                    tf.math.squared_difference(self.discB.output, 1))) / 2

    def __create_optimizers(self):
        optimizer = self.cfg.optimizer(**self.cfg.optimizer_args)

        with tf.device(self.device_0):
            self.gA_cost = optimizer.minimize(self.gA_obj, var_list=self.genA.trainable_weights)
            self.dA_cost = optimizer.minimize(self.dA_obj, var_list=self.discA.trainable_weights)

        with tf.device(self.device_1):
            self.gB_cost = optimizer.minimize(self.gB_obj, var_list=self.genB.trainable_weights)
            self.dB_cost = optimizer.minimize(self.dB_obj, var_list=self.discB.trainable_weights)

    def __setup_logging(self, logs_dir):
        # Logging costs
        self.gen_costs = tf.summary.merge([
            tf.summary.scalar("Cyclic_recoA", self.cycA_obj),
            tf.summary.scalar("Cyclic_recoB", self.cycB_obj),
            tf.summary.scalar("GenA_gan", self.ganA_obj),
            tf.summary.scalar("GenB_gan", self.ganB_obj),
            tf.summary.scalar("GenA_sum", self.gA_obj),
            tf.summary.scalar("GenB_sum", self.gB_obj),
            tf.summary.scalar("IminAS_norm", self.norm_AS_obj),
            tf.summary.scalar("Conic_dist", self.conic_dist)
        ])

        self.disc_costs = tf.summary.merge([
            tf.summary.scalar("DiscA_gan", self.dA_obj),
            tf.summary.scalar("DiscB_gan", self.dB_obj)
        ])

        # Logging images and weight histograms
        self.epoch_end = tf.summary.merge([
            tf.summary.image("GT_A", self.inputA),
            tf.summary.image("GT_B", self.inputB),
            tf.summary.image("Gen_A", self.genA(self.inputB)),
            tf.summary.image("Gen_B", self.genB(self.inputA)),
            tf.summary.image("Rec_A", self.genA(self.genB(self.inputA))),
            tf.summary.image("Rec_B", self.genB(self.genA(self.inputB))),
            create_weight_histograms(self.genA, "GenA"),
            create_weight_histograms(self.genB, "GenB"),
            create_weight_histograms(self.discA, "DiscA"),
            create_weight_histograms(self.discB, "DiscB")
        ])

        self.writer = tf.summary.FileWriter(
            logs_dir + os.sep + self.cfg.name, tf.get_default_graph())

        self.writer.flush()

    def __setup_checkpoints(self, checkpoints_dir):
        self.checkpoints_dir = checkpoints_dir
        os.mkdir(checkpoints_dir)

    def __checkpoint_models(self):
        self.genA.save(self.checkpoints_dir + os.sep + "genA_" + str(self.epoch) + ".hdf5", include_optimizer=False)
        self.genB.save(self.checkpoints_dir + os.sep + "genB_" + str(self.epoch) + ".hdf5", include_optimizer=False)

        self.discA.save(self.checkpoints_dir + os.sep + "discA_" + str(self.epoch) + ".hdf5", include_optimizer=False)
        self.discB.save(self.checkpoints_dir + os.sep + "discB_" + str(self.epoch) + ".hdf5", include_optimizer=False)

    def train(self):
        epoch_iters = int(self.cfg.dataset_size / self.cfg.batch_size)

        # Do the actual training
        for epoch in range(self.cfg.epochs):
            self.epoch = epoch

            for it in custom_bar(epoch, epoch_iters)(range(epoch_iters)):
                _, _, disc_costs = self.sess.run([self.dA_cost, self.dB_cost, self.disc_costs])
                _, _, gen_costs = self.sess.run([self.gA_cost, self.gB_cost, self.gen_costs])

                t = epoch_iters * epoch + it
                self.writer.add_summary(disc_costs, t)
                self.writer.add_summary(gen_costs, t)

            self.writer.add_summary(self.sess.run(self.epoch_end), self.epoch)
            self.writer.flush()
            self.__checkpoint_models()

        # Run end
        self.writer.close()
        self.sess.close()


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

    trainer = PolarCycle(config, rgb_path, polar_path, logs_dir, checkpoints_dir)
    trainer.train()

    tf.reset_default_graph()


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
