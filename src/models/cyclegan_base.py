import os

import numpy as np
import tensorflow as tf
from deeplauncher.filesystem.path import Paths

from deeplauncher.models.abstractmodel import AbstractModel
from base_configs.cyclegan_config import CycleGANConfig
from deeplauncher.utils.log import custom_bar, detect_nan
from src.utils.pool import from_pool, update_pool


class CycleGANBase(AbstractModel):

    def __init__(self, cfg: CycleGANConfig, paths: Paths, sess: tf.compat.v1.Session, resume=None, epoch=0):
        self.sess = sess
        self.cfg = cfg
        self.paths = paths
        self.resume = resume
        self.start_epoch = epoch
        self.global_step = tf.Variable(epoch, trainable=False)
        self.sess.run(tf.compat.v1.variables_initializer([self.global_step]))
        self.dbgg = []
        self.dbgd = []

    def dbgdt(self, tensor, name):
        self.dbgd += [
            detect_nan(tensor, name)
        ]

    def dbggt(self, tensor, name):
        self.dbgg += [
            detect_nan(tensor, name)
        ]

    def setup_datasets(self):
        iter_a = tf.compat.v1.data.make_initializable_iterator(
            self.cfg.dataloader(self.paths.data_paths[0], self.cfg.batch_size, self.cfg.dataA_channels,
                                           self.cfg.image_size, self.cfg.crop_size).train().repeat())

        iter_b = tf.compat.v1.data.make_initializable_iterator(self.cfg.dataloader(self.paths.data_paths[1], self.cfg.batch_size,
                                           self.cfg.dataB_channels,
                                           self.cfg.image_size, self.cfg.crop_size).train().repeat())

        self.sess.run(iter_a.initializer)
        self.sess.run(iter_b.initializer)

        self.inputA = iter_a.get_next()
        self.inputB = iter_b.get_next()

    def build_models(self):
        # A: RGB, B: Pol
        with tf.device(self.device_0):
            self.genA = self.cfg.genA(self.inputB, **self.cfg.genA_args)
            self.discA = self.cfg.discA(self.inputA, **self.cfg.discA_args)

            self.poolA = tf.Variable(trainable=False, dtype=tf.float32,
                                     initial_value=np.ones((self.cfg.pool_size,
                                                            self.cfg.crop_size,
                                                            self.cfg.crop_size,
                                                            self.cfg.dataA_channels)))

        self.out_gA = self.genA.output
        self.out_dA = self.discA.output
        self.sess.run(tf.compat.v1.variables_initializer(self.genA.variables))
        self.sess.run(tf.compat.v1.variables_initializer(self.discA.variables))
        self.sess.run(tf.compat.v1.variables_initializer([self.poolA]))

        with tf.device(self.device_1):
            self.genB = self.cfg.genB(self.inputA, **self.cfg.genB_args)
            self.discB = self.cfg.discB(self.inputB, **self.cfg.discB_args)
            self.poolB = tf.Variable(trainable=False, dtype=tf.float32,
                                     initial_value=np.zeros((self.cfg.pool_size,
                                                             self.cfg.crop_size,
                                                             self.cfg.crop_size,
                                                             self.cfg.dataB_channels)))

        self.out_gB = self.genB.output
        self.out_dB = self.discB.output
        self.sess.run(tf.compat.v1.variables_initializer(self.genB.variables))
        self.sess.run(tf.compat.v1.variables_initializer(self.discB.variables))
        self.sess.run(tf.compat.v1.variables_initializer([self.poolB]))

        with tf.device(self.device_0):
            with tf.name_scope("cycA"):
                self.cycA = self.genA(self.out_gB)

            with tf.name_scope("ganA"):
                self.ganA = self.discA(from_pool(self.poolA, self.out_gA, self.cfg.pool_size, self.cfg.batch_size))
            self.poolA_update = update_pool(self.poolA, self.out_gA, self.cfg.pool_size, self.cfg.batch_size)

        with tf.device(self.device_1):
            with tf.name_scope("cycB"):
                self.cycB = self.genB(self.out_gA)

            with tf.name_scope("ganB"):
                self.ganB = self.discB(from_pool(self.poolB, self.out_gB, self.cfg.pool_size, self.cfg.batch_size))
            self.poolB_update = update_pool(self.poolB, self.out_gB, self.cfg.pool_size, self.cfg.batch_size)

    def create_objectives(self):
        # Objectives
        with tf.device(self.device_0):
            with tf.name_scope("gA_objective"):
                self.ganA_obj = tf.reduce_mean((self.ganA - 1) ** 2)
            self.cycA_obj = tf.reduce_mean((self.inputA - self.cycA) ** 2)

            self.gA_obj = self.ganA_obj + self.cfg.cyc_factor * self.cycA_obj

        with tf.name_scope("dA_objective"):
            self.dA_obj = (tf.reduce_mean(self.ganA ** 2) + tf.reduce_mean((self.discA.output - 1) ** 2)) / 2

        with tf.device(self.device_1):
            with tf.name_scope("gB_objective"):
                self.ganB_obj = tf.reduce_mean((self.ganB - 1) ** 2)
                self.cycB_obj = tf.reduce_mean((self.inputB - self.cycB) ** 2)

                self.gB_obj = self.ganB_obj + self.cfg.cyc_factor * self.cycB_obj

            with tf.name_scope("dB_objective"):
                self.dB_obj = (tf.reduce_mean(self.ganB ** 2) + tf.reduce_mean((self.discB.output - 1) ** 2)) / 2

    def create_optimizers(self):
        lr = tf.Variable(self.cfg.learning_rate)
        self.decayed_lr = tf.compat.v1.train.polynomial_decay(
            lr, tf.maximum(self.global_step - (self.cfg.epochs // 2), 0),
            self.cfg.epochs // 2, self.cfg.learning_rate // 100)

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.decayed_lr, beta1=0.5)

        with tf.device(self.device_0):
            self.gA_cost = optimizer.minimize(self.gA_obj, var_list=self.genA.trainable_weights)
            self.dA_cost = optimizer.minimize(self.dA_obj, var_list=self.discA.trainable_weights)

        with tf.device(self.device_1):
            self.gB_cost = optimizer.minimize(self.gB_obj, var_list=self.genB.trainable_weights)
            self.dB_cost = optimizer.minimize(self.dB_obj, var_list=self.discB.trainable_weights)

        self.sess.run(tf.compat.v1.variables_initializer([lr]))
        self.sess.run(tf.compat.v1.variables_initializer(optimizer.variables()))

    def setup_logging(self):
        # Logging images and weight histograms
        self.summaries = tf.compat.v1.summary.merge([
            tf.compat.v1.summary.scalar("Cyclic_recoA", self.cycA_obj),
            tf.compat.v1.summary.scalar("Cyclic_recoB", self.cycB_obj),
            tf.compat.v1.summary.scalar("GenA_gan", self.ganA_obj),
            tf.compat.v1.summary.scalar("GenB_gan", self.ganB_obj),
            tf.compat.v1.summary.scalar("GenA_sum", self.gA_obj),
            tf.compat.v1.summary.scalar("GenB_sum", self.gB_obj),
            tf.compat.v1.summary.scalar("DiscA_gan", self.dA_obj),
            tf.compat.v1.summary.scalar("DiscB_gan", self.dB_obj),
            tf.compat.v1.summary.scalar("Learning_rate", self.decayed_lr),

            tf.compat.v1.summary.image("GT_A", self.inputA),
            tf.compat.v1.summary.image("GT_B", self.inputB),
            tf.compat.v1.summary.image("Gen_A", self.out_gA),
            tf.compat.v1.summary.image("Gen_B", self.out_gB),
            tf.compat.v1.summary.image("Rec_A", self.cycA),
            tf.compat.v1.summary.image("Rec_B", self.cycB),

        ])

        if self.cfg.debug:
            self.dbggt(self.out_gA, "out_GA")
            self.dbggt(self.out_gB, "out_GB")
            self.dbggt(self.out_dA, "out_DA")
            self.dbggt(self.out_dB, "out_DB")

            self.dbgdt(self.out_gA, "out_GA")
            self.dbgdt(self.out_gB, "out_GB")
            self.dbgdt(self.out_dA, "out_DA")
            self.dbgdt(self.out_dB, "out_DB")

            self.dbggt(self.cycA, "cycA")
            self.dbggt(self.cycB, "cycB")

            self.dbggt(self.ganA, "ganB")
            self.dbggt(self.ganB, "ganA")
            self.dbgdt(self.ganA, "ganB")
            self.dbgdt(self.ganB, "ganA")

            self.dbggt(self.ganA_obj, "ganA_obj")
            self.dbggt(self.ganB_obj, "ganB_obj")
            self.dbgdt(self.ganA_obj, "ganA_obj")
            self.dbgdt(self.ganB_obj, "ganB_obj")

            self.dbggt(self.gA_obj, "gA_obj")
            self.dbggt(self.gB_obj, "gB_obj")
            self.dbgdt(self.dA_obj, "dA_obj")
            self.dbgdt(self.dB_obj, "dB_obj")
            self.dbggt(self.cycA_obj, "cycA_obj")
            self.dbggt(self.cycB_obj, "cycB_obj")

            self.dbgdt(self.inputA, "InputA")
            self.dbgdt(self.inputB, "InputB")

            self.dbggt(self.inputA, "InputA")
            self.dbggt(self.inputB, "InputB")

        self.writer = tf.compat.v1.summary.FileWriter(
            self.paths.logs_dir + os.sep + self.cfg.name, tf.compat.v1.get_default_graph())

        self.writer.flush()

    def checkpoint_models(self, epoch):
        with tf.device(self.device_0):
            self.genA.save(self.paths.checkpoints_dir + os.sep + "genA_" + str(epoch) + ".hdf5", include_optimizer=True)
            self.discA.save(self.paths.checkpoints_dir + os.sep + "discA_" + str(epoch) + ".hdf5", include_optimizer=True)

        with tf.device(self.device_1):
            self.genB.save(self.paths.checkpoints_dir + os.sep + "genB_" + str(epoch) + ".hdf5", include_optimizer=True)
            self.discB.save(self.paths.checkpoints_dir + os.sep + "discB_" + str(epoch) + ".hdf5", include_optimizer=True)

    def train_step(self):
        self.sess.run([self.dA_cost, self.dB_cost] + self.dbgd)
        self.sess.run([self.gA_cost, self.gB_cost] + self.dbgg)
        self.sess.run([self.poolA_update, self.poolB_update])

    def train(self):
        epoch_iters = int(self.cfg.dataset_size / self.cfg.batch_size)
        for epoch in range(self.start_epoch, self.cfg.epochs):
            for mb in custom_bar(epoch, epoch_iters)(range(epoch_iters)):
                self.train_step()

            self.writer.add_summary(self.sess.run(self.summaries), epoch)
            self.checkpoint_models(epoch)
            self.sess.run(tf.compat.v1.assign(self.global_step, self.global_step + 1))



        # Run end
        self.writer.close()

    def resume_models(self, resume, epoch):
        with tf.device(self.device_0):
            self.genA.load_weights(os.path.join(resume,
                                                ), by_name=False)
            self.discA.load_weights(os.path.join(resume, "discA_" + str(epoch) + ".hdf5"), by_name=False)

        with tf.device(self.device_1):
            self.genB.load_weights(os.path.join(resume, "genB_" + str(epoch) + ".hdf5"), by_name=False)
            self.discB.load_weights(os.path.join(resume, "discB_" + str(epoch) + ".hdf5"), by_name=False)
