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


class Trainer:

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


        self.__setup_datasets(rgb_path, polar_path)
        self.__build_models()
        self.__create_objectives()
        self.__create_optimizers()
        self.__setup_logging(logs_dir)
        self.__setup_checkpoints(checkpoints_dir)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    def __setup_datasets(self, rgb_path, polar_path):
        self.dataA = from_images.get_iterator(rgb_path, self.cfg.dataset_size, self.cfg.batch_size, 3)
        self.dataB = from_images.get_iterator(polar_path, self.cfg.dataset_size, self.cfg.batch_size, 4)

        self.iterA = self.dataA.get_next()
        self.iterB = self.dataB.get_next()

    def __build_models(self):
        # A: RGB
        # B: Pol
        with tf.device(self.device_0):
            self.genA = self.cfg.genA(self.iterB, **self.cfg.genA_args)
            self.discA = self.cfg.discA(self.iterA, **self.cfg.discA_args)
        self.out_gA = self.genA.output
        self.out_dA = self.discA.output

        with tf.device(self.device_1):
            self.genB = self.cfg.genB(self.iterA, **self.cfg.genB_args)
            self.discB = self.cfg.discB(self.iterB, **self.cfg.discB_args)
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
                self.cycA_obj = tf.reduce_mean(tf.abs(self.iterA - self.cycA))

                self.gA_obj = self.ganA_obj + self.cfg.cyc_factor * self.cycA_obj

            with tf.name_scope("dA_objective"):
                self.dA_obj = (tf.reduce_mean(tf.square(self.ganA)) + tf.reduce_mean(
                    tf.math.squared_difference(self.discA.output, 1))) / 2

        with tf.device(self.device_1):
            with tf.name_scope("gB_objective"):
                self.ganB_obj = tf.reduce_mean(tf.squared_difference(self.ganB, 1))
                self.cycB_obj = tf.reduce_mean(tf.abs(self.iterB - self.cycB))

                self.gB_obj = self.ganB_obj + self.cfg.cyc_factor * self.cycB_obj

                x = (self.out_gB + 1) * 128

                img_I0 = x[:, :, :, 0]
                img_I45 = x[:, :, :, 1]
                img_I90 = x[:, :, :, 2]
                img_I135 = x[:, :, :, 3]

                S0 = tf.add(img_I0, img_I90)
                S1 = tf.subtract(img_I0, img_I90)
                S2 = tf.subtract(img_I45, img_I135)

                if self.cfg.norm_AS:
                    A = tf.cast(0.5 * np.array([[[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1]]] * self.cfg.batch_size),
                                tf.float32)
                    S = tf.cast(tf.reshape(tf.stack([S0, S1, S2]), shape=[self.cfg.batch_size, 3, -1]), tf.float32)
                    I = tf.cast(tf.reshape(x, shape=[self.cfg.batch_size, 4, -1]), tf.float32)

                    AS = tf.matmul(A, S)
                    delta1 = I - AS
                    norm_AS = tf.norm(delta1) / tf.add(tf.norm(I), tf.norm(AS))

                    self.norm_AS_obj = self.cfg.lmbda * norm_AS
                    self.gB_obj += self.norm_AS_obj
                else:
                    self.norm_AS_obj = tf.constant(0)

                if self.cfg.conic_dist:
                    phi = tf.subtract(tf.div(tf.sqrt(S1 ** 2 + S2 ** 2), S0), 1)
                    phi = tf.maximum(phi, 0)

                    conic_dist = tf.div(tf.norm(phi), 500 * 500)

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
        self.cycA_summ = tf.summary.scalar("Cyclic_recoA", self.cycA_obj)
        self.ganA_summ = tf.summary.scalar("GenA_gan", self.ganA_obj)
        self.totalA_summ = tf.summary.scalar("GenA_sum", self.gA_obj)
        self.discA_summ = tf.summary.scalar("DiscA_gan", self.dA_obj)

        self.cycB_summ = tf.summary.scalar("Cyclic_recoB", self.cycB_obj)
        self.ganB_summ = tf.summary.scalar("GenB_gan", self.ganB_obj)
        self.totalB_summ = tf.summary.scalar("GenB_sum", self.gB_obj)
        self.discB_summ = tf.summary.scalar("DiscB_gan", self.dB_obj)

        self.norm_AS_summ = tf.summary.scalar("IminAS_norm", self.norm_AS_obj)
        self.conic_dist_summ = tf.summary.scalar("Conic_dist", self.conic_dist)

        # Logging images
        self.trueApl = tf.placeholder(tf.float32, shape=(1, self.cfg.image_size, self.cfg.image_size, 3))
        self.trueBpl = tf.placeholder(tf.float32, shape=(1, self.cfg.image_size, self.cfg.image_size, 4))

        self.imgsumm = tf.summary.merge([
            tf.summary.image("GT_A", self.trueApl),
            tf.summary.image("GT_B", self.trueBpl),
            tf.summary.image("Gen_A", self.genA(self.trueBpl)),
            tf.summary.image("Gen_B", self.genB(self.trueApl)),
            tf.summary.image("Rec_A", self.genA(self.genB(self.trueApl))),
            tf.summary.image("Rec_B", self.genB(self.genA(self.trueBpl)))
        ])

        # Logging weights histograms
        self.weightsum = tf.summary.merge([
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

    def __log_weights_histograms(self):
        weightout = self.sess.run(self.weightsum)
        self.writer.add_summary(weightout, self.epoch)

    def __log_images(self):
        true_A = self.sess.run(self.iterA)[:1]
        true_B = self.sess.run(self.iterB)[:1]

        imgout = self.sess.run(
            self.imgsumm, feed_dict={
                self.trueApl: true_A,
                self.trueBpl: true_B
            })
        self.writer.add_summary(imgout, self.epoch)

    def __checkpoint_models(self):
        self.genA.save(self.checkpoints_dir + os.sep + "genA_" + str(self.epoch) + ".hdf5", include_optimizer=False)
        self.genB.save(self.checkpoints_dir + os.sep + "genB_" + str(self.epoch) + ".hdf5", include_optimizer=False)

        self.discA.save(self.checkpoints_dir + os.sep + "discA_" + str(self.epoch) + ".hdf5", include_optimizer=False)
        self.discB.save(self.checkpoints_dir + os.sep + "discB_" + str(self.epoch) + ".hdf5", include_optimizer=False)

    def run(self):
        # Setting up the training
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.dataA.initializer)
        self.sess.run(self.dataB.initializer)

        epoch_iters = int(self.cfg.dataset_size / self.cfg.batch_size)
        # Do the actual training
        for epoch in range(self.cfg.epochs):
            self.epoch = epoch

            bar = progressbar.ProgressBar(widgets=[
                "Epoch " + str(epoch), ' ',
                progressbar.Percentage(), ' ',
                progressbar.SimpleProgress(format='(%s)' % progressbar.SimpleProgress.DEFAULT_FORMAT),
                progressbar.Bar(), ' ',
                progressbar.Timer(), ' ',
                progressbar.AdaptiveETA()
            ], maxvalue=epoch_iters, redirect_stdout=True)

            for it in bar(range(epoch_iters)):
                # Training discriminators
                _, _, dA_err, dB_err = self.sess.run([self.dA_cost, self.dB_cost, self.discA_summ, self.discB_summ])

                # Training generators
                _, _, ganA_err, ganB_err, cycA_err, cycB_err, norm_AS_err, conic_dist_err, totalA_err, totalB_err  = \
                    self.sess.run([self.gA_cost, self.gB_cost, self.ganA_summ, self.ganB_summ, self.cycA_summ,
                                   self.cycB_summ, self.norm_AS_summ, self.conic_dist_summ, self.totalA_summ,
                                   self.totalB_summ])

                # Logging losses
                t = epoch_iters * epoch + it
                self.writer.add_summary(dA_err, t)
                self.writer.add_summary(dB_err, t)
                self.writer.add_summary(ganA_err, t)
                self.writer.add_summary(ganB_err, t)
                self.writer.add_summary(cycA_err, t)
                self.writer.add_summary(cycB_err, t)
                self.writer.add_summary(totalA_err, t)
                self.writer.add_summary(totalB_err, t)
                self.writer.add_summary(norm_AS_err, t)
                self.writer.add_summary(conic_dist_err, t)

            self.__log_images()
            self.__log_weights_histograms()
            self.__checkpoint_models()
            self.writer.flush()

        # Run end
        self.writer.close()


def do_run(filepath, rgb_path, polar_path, extension=""):
    name = filepath.split('/')[-1].replace('.py', '') + extension
    config = loadconfig(filepath)

    print("\nRunning " + name + "\n")

    if not os.path.exists("./runs"):
        os.mkdir("./runs")
    os.mkdir("./runs/" + name)
    shutil.copy2(filepath, './runs/' + name + "/config.py")
    checkpoints_dir = "./runs/" + name + "/checkpoints/"
    logs_dir = "./runs/" + name + "/logs/"

    trainer = Trainer(config, rgb_path, polar_path, logs_dir, checkpoints_dir)
    trainer.run()


def main():
    if len(sys.argv) < 4:
        print("Usage : python train_polarcycle.py config_file rgb_path polar_path")
        exit(1)

    filepath = sys.argv[1]
    rgb_path = sys.argv[2]
    polar_path = sys.argv[2]

    if os.path.isfile(filepath):
        do_run(filepath, rgb_path, polar_path)
    else:
        for run in os.listdir(filepath):
            if ".py" in run:
                do_run(filepath + "/" + run, rgb_path, polar_path)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
