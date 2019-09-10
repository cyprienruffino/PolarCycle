import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
# from scipy.misc import imsave
import matplotlib.pyplot as plt
import os
import shutil
from PIL import Image
import time
import random
import sys
import cv2

from layers import *
from model import *
from Solver import *

# *********************************************************************************************************
max_img = 100
nb_epoch = 200
# *********************************************************************************************************
img_height = 500
img_width = 500

img_layer_A = 3
img_layer_B = 4

img_size = img_height * img_width
img_sizeB = img_height * img_width
path = os.getcwd()

temp_check = 0

batch_size = 5
pool_size = 50
sample_size = 10
ngf = 32
ndf = 64
lamda = 3

l_g_loss_cyc = []
l_g_loss_gc1 = []
l_g_loss_gc2 = []

output_path = "./Resultat_tmp"
output_fakeRGB = output_path + "/fakeRGB"
output_fakePolar = output_path + "/fakePolar"
output_fig = output_path + "/fig"
out = "./output"
check_dir = out + "/checkpoints/"


# ***********************************************************************************************************
def fake_image_pool(num_fakes, fake, fake_pool):
    if (num_fakes < pool_size):
        fake_pool[num_fakes] = fake
        return fake
    else:
        p = random.random()
        if p > 0.5:
            random_id = random.randint(0, pool_size - 1)
            temp = fake_pool[random_id]
            fake_pool[random_id] = fake
            return temp
        else:
            return fake


class CycleGAN():

    def __init__(self):

        # Read image on the data base
        os.chdir(path)
        img_A = ProcessInputImgA()
        img_B = ProcessInputImgB()

        self.fake_images_A = np.zeros((pool_size, 1, img_height, img_width, img_layer_A))
        self.fake_images_B = np.zeros((pool_size, 1, img_height, img_width, img_layer_B))

        self.image_A = img_A
        self.image_B = img_B

        self.A_input = img_A
        self.B_input = img_B

        self.input_A = tf.placeholder(tf.float32, [batch_size, img_height, img_width, img_layer_A], name="input_A")
        self.input_B = tf.placeholder(tf.float32, [batch_size, img_height, img_width, img_layer_B], name="input_B")

        self.fake_pool_A = tf.placeholder(tf.float32, [None, img_height, img_width, img_layer_A], name="fake_pool_A")
        self.fake_pool_B = tf.placeholder(tf.float32, [None, img_height, img_width, img_layer_B], name="fake_pool_B")

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.lr = tf.placeholder(tf.float32, shape=[], name="lr")

        with tf.variable_scope("Model") as scope:
            self.fake_B = build_generator_resnet_9blocks(self.input_A, img_layerB, name="g_A")
            self.fake_A = build_generator_resnet_9blocks(self.input_B, img_layerA, name="g_B")
            self.rec_A = build_gen_discriminator(self.input_A, "d_A")
            self.rec_B = build_gen_discriminator(self.input_B, "d_B")

            scope.reuse_variables()

            self.fake_rec_A = build_gen_discriminator(self.fake_A, "d_A")
            self.fake_rec_B = build_gen_discriminator(self.fake_B, "d_B")
            self.cyc_A = build_generator_resnet_9blocks(self.fake_B, img_layerA, "g_B")
            self.cyc_B = build_generator_resnet_9blocks(self.fake_A, img_layerB, "g_A")

            scope.reuse_variables()

            self.fake_pool_rec_A = build_gen_discriminator(self.fake_pool_A, "d_A")
            self.fake_pool_rec_B = build_gen_discriminator(self.fake_pool_B, "d_B")

    def loss_calc(self):

        cyc_loss = tf.reduce_mean(tf.abs(self.input_A - self.cyc_A)) + tf.reduce_mean(tf.abs(self.input_B - self.cyc_B))

        disc_loss_A = tf.reduce_mean(tf.squared_difference(self.fake_rec_A, 1))
        disc_loss_B = tf.reduce_mean(tf.squared_difference(self.fake_rec_B, 1))

        errC1 = EvalImageConstraintC1(self.fake_B)
        errC2 = EvalImageConstraintC2(self.fake_B)

        g_loss_A = disc_loss_B + 10 * cyc_loss + lamda * errC1
        g_loss_B = disc_loss_A + 10 * cyc_loss

        d_loss_A = (tf.reduce_mean(tf.square(self.fake_pool_rec_A)) + tf.reduce_mean(
            tf.squared_difference(self.rec_A, 1))) / 2.0
        d_loss_B = (tf.reduce_mean(tf.square(self.fake_pool_rec_B)) + tf.reduce_mean(
            tf.squared_difference(self.rec_B, 1))) / 2.0

        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)

        self.model_vars = tf.trainable_variables()

        d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
        d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        g_B_vars = [var for var in self.model_vars if 'g_B' in var.name]

        self.d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars)
        self.d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars)
        self.g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars)
        self.g_B_trainer = optimizer.minimize(g_loss_B, var_list=g_B_vars)

        for var in self.model_vars: print(var.name)

        self.g_A_loss_summ = tf.summary.scalar("g_A_loss", g_loss_A)
        self.g_B_loss_summ = tf.summary.scalar("g_B_loss", g_loss_B)
        self.d_A_loss_summ = tf.summary.scalar("d_A_loss", d_loss_A)
        self.d_B_loss_summ = tf.summary.scalar("d_B_loss", d_loss_B)

        # cycle loss setter 
        self.cyc_loss = cyc_loss
        self.errC1 = errC1
        self.errC2 = errC2

    def train(self, sess):
        # Read image
        os.chdir(path)

        # Loss function calculations
        self.loss_calc()

        saver = tf.train.Saver()
        # Compute error on the real images
        """
        L_errC1 = []
        L_errC2 = []
        for i in range(len(img_B)):
            tmp_errC1 = sess.run([ComputeErrC1], feed_dict={x: img_B[i]})
            tmp_errC1 = tmp_errC1[0]

            tmp_errC2 = sess.run([ComputeErrC2], feed_dict={x: img_B[i]})
            tmp_errC2 = tmp_errC2[0]

            print("errC1 " + str(i) + " : ", tmp_errC1)
            print("errC2 " + str(i) + " : ", tmp_errC2)

            L_errC1.append(tmp_errC1)
            L_errC2.append(tmp_errC2)

        l_g_loss_gc1.append(np.mean(L_errC1))
        l_g_loss_gc2.append(np.mean(L_errC2))
        l_g_loss_cyc.append(0.9)
        """
        # Initializing the global variables
        init = ([tf.global_variables_initializer(), tf.local_variables_initializer()])
        sess.run(init)
        writer = tf.summary.FileWriter(out + "/2")

        # Training Loop
        for epoch in range(sess.run(self.global_step), nb_epoch):
            print("In the epoch ", epoch)
            saver.save(sess, os.path.join(check_dir, "cyclegan.ckpt"), global_step=epoch)

            # Dealing with the learning rate as per the epoch number
            if(epoch < 100) :
                curr_lr = 0.0002
            else:
                curr_lr = 0.0002 - 0.0002*(0.99**(epoch-99))

            tmp_cyc_loss = []
            tmp_gc1_loss = []
            tmp_gc2_loss = []
            for ptr in range(0, max_img, batch_size):
                print("In the iteration ", ptr)

                # Optimizing the G_A network

                _, fake_B_temp, summary_str = sess.run([self.g_A_trainer, self.fake_B, self.g_A_loss_summ],
                                                           feed_dict={self.input_A: np.squeeze(self.A_input[ptr:ptr+batch_size]),
                                                                      self.input_B: np.squeeze(self.B_input[ptr:ptr+batch_size]),
                                                                      self.lr: curr_lr})
                writer.add_summary(summary_str, epoch * max_img + ptr)
                fake_B_temp1 = fake_image_pool(self.num_fake_inputs, fake_B_temp, self.fake_images_B)

                # Optimizing the D_B network
                _, summary_str = sess.run([self.d_B_trainer, self.d_B_loss_summ],
                                              feed_dict={self.input_A: np.squeeze(self.A_input[ptr:ptr+batch_size]),
                                                         self.input_B: np.squeeze(self.B_input[ptr:ptr+batch_size]),
                                                         self.lr: curr_lr,
                                                         self.fake_pool_B: fake_B_temp1})
                writer.add_summary(summary_str, epoch * max_img + ptr)

                # Optimizing the G_B network
                _, fake_A_temp, summary_str = sess.run([self.g_B_trainer, self.fake_A, self.g_B_loss_summ],
                                                           feed_dict={self.input_A: np.squeeze(self.A_input[ptr:ptr+batch_size]),
                                                                      self.input_B: np.squeeze(self.B_input[ptr:ptr+batch_size]),
                                                                      self.lr: curr_lr})

                writer.add_summary(summary_str, epoch * max_img + ptr)

                fake_A_temp1 = fake_image_pool(self.num_fake_inputs, fake_A_temp, self.fake_images_A)

                # Optimizing the D_A network
                _, summary_str = sess.run([self.d_A_trainer, self.d_A_loss_summ],
                                              feed_dict={self.input_A: np.squeeze(self.A_input[ptr:ptr+batch_size]),
                                                         self.input_B: np.squeeze(self.B_input[ptr:ptr+batch_size]), self.lr: curr_lr,
                                                         self.fake_pool_A: fake_A_temp1})

                writer.add_summary(summary_str, epoch * max_img + ptr)

                self.num_fake_inputs += 1

                # loss calcul
                cyc_loss, fake_B, errC11, errC22 = sess.run([self.cyc_loss, self.fake_B, self.errC1, self.errC2],
                                                                feed_dict={self.input_A: np.squeeze(self.A_input[ptr:ptr+batch_size]),
                                                                           self.input_B: np.squeeze(self.B_input[ptr:ptr+batch_size]),
                                                                           self.lr: curr_lr,
                                                                           self.fake_pool_B: fake_B_temp1,
                                                                           self.fake_pool_A: fake_A_temp1})

                print("errC1 : ", errC11)
                print("errC2 : ", errC22)

                tmp_cyc_loss.append(cyc_loss)
                tmp_gc1_loss.append(errC11)
                tmp_gc2_loss.append(errC22)
            # ----------------------------------------------
            os.chdir(path)
            if not os.path.exists(output_fig):
                os.makedirs(output_fig)
            os.chdir(output_fig)

            tt = np.arange(max_img)

            plt.plot(tt, tmp_gc1_loss, label="C1")
            plt.plot(tt, tmp_gc2_loss, label="C2")
            plt.plot(tt, tmp_cyc_loss, label="cyc_loss")
            plt.xlabel("img")
            plt.ylabel("error")
            plt.legend()
            plt.title("epoch_" + str(epoch))
            plt.savefig("fig" + str(epoch) + ".png")
            plt.close()

            # ---------------------------------------------------------------------------------------------------
            l_g_loss_cyc.append(np.mean(tmp_cyc_loss))
            l_g_loss_gc1.append(np.mean(tmp_gc1_loss))
            l_g_loss_gc2.append(np.mean(tmp_gc2_loss))

            sess.run(tf.assign(self.global_step, epoch + 1))

            # ************************save image *************************************************************

            for i in range(10):
                fake_B, fake_A = sess.run([self.fake_B, self.fake_A],
                                              feed_dict={self.input_A: self.A_input[i], self.input_B: self.B_input[i],
                                                         self.lr: curr_lr, self.fake_pool_B: fake_B_temp1,
                                                         self.fake_pool_A: fake_A_temp1})

                os.chdir(path)
                if not os.path.exists(output_fakePolar):
                    os.makedirs(output_fakePolar)
                os.chdir(output_fakePolar)
                WriteOutputImgB(fake_B, epoch, i)

                os.chdir(path)
                if not os.path.exists(output_fakeRGB):
                    os.makedirs(output_fakeRGB)
                os.chdir(output_fakeRGB)
                WriteOutputImgA(fake_A, epoch, i)

            os.chdir(path)
            writer.add_graph(sess.graph)

        # save loss +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        os.chdir(path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        os.chdir(output_path)

        t = np.arange(nb_epoch + 1)
        # np.savetxt('loss_gc1.txt', l_g_loss_gc1, delimiter=',')
        # np.savetxt('loss_cyc.txt', l_g_loss_cyc, delimiter=',')
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        plt.plot(t, l_g_loss_gc1, label="C1")
        plt.plot(t, l_g_loss_gc2, label="C2")
        plt.plot(t, l_g_loss_cyc, label="cyc_loss")
        plt.xlabel("epoch")
        plt.ylabel("mean error per epoch")
        plt.legend()
        plt.title(output_path)
        plt.savefig("fig.png")
        plt.close()
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def main():
    with tf.Session() as sess:
        model = CycleGAN()
        model.train(sess)


if __name__ == '__main__':
    main()
