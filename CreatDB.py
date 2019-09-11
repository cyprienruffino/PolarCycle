#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 13:19:59 2019

@author: seck
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:26:45 2019

@author: Brane

SeckFileProcess1 est un script permetant d'avaluer les contraintes polarimetriques C1 et C2 
dans les images. Il permet aussi de connstruite une nouvelle base de donnée contenant les meilleurs images.
"""

import os, fnmatch
import numpy as np
import cv2
import tensorflow as tf

# **************************************************************************************************
# on recupere le chemein absolu du répertoire mère
path = os.getcwd()
# taille des images apres redimensionnement
res = (500, 500)
# le nombre exact d'image à utiliser apres traitement(filtrage)
max_img = 100
# utiliser pour eviter de diviser par zeros
epsilon = 1e-5
# batch_size = 1
batch_size = 1
# img_height = 500
img_height = 500
# img_width
img_width = 500
# img_layer_B
img_layer_B = 4
# repertoire contenant tous les images 
path_input = "./input/rgb2pol/trainBB"
if not os.path.exists(path_input):
    print("****")
    print("Le repertoire contenant tous les images trainBB n'est pas trouver")
    print("****")
# repertoire contenant les images séléctionner
path_output = "./DB" + str(max_img)
if not os.path.exists(path_output):
    os.makedirs(path_output)

x = tf.placeholder(tf.float32, shape=[1, 500, 500, 4], name='x')
# initialisation des graphs de calcul c1 et c2

# *****************************************************************************************************
# ttheorical param polarimetric
A = 0.5 * np.array([[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1]])


def EvalImageConstraintC2(x, name="ImageConstraintC2"):
    with tf.variable_scope(name):
        x = np.squeeze(x)
        x = (x + 1) * 128

        img_I0 = x[:, :, 0]
        img_I45 = x[:, :, 1]
        img_I90 = x[:, :, 2]
        img_I135 = x[:, :, 3]

        S0 = img_I0 + img_I90
        S1 = img_I0 - img_I90
        S2 = img_I45 - img_I135

        a = tf.sqrt(S1 ** 2 + S2 ** 2)
        b = tf.add(S0, epsilon)
        PHI = tf.subtract(tf.div(a, b), 1)
        PHI = tf.maximum(PHI, 0)
        err = tf.norm(PHI)
        err = tf.div(err, 500 * 500)
        return err


# **************************************************************************************************
ComputeErrC2 = EvalImageConstraintC2(x)


# **************************************************************************************************
def ProcessInputImgB():
    # On stoque sur X tous les images

    # on se place sur le répertoire trainBB
    os.chdir(path_input)
    # on stoque sur ch la list de tous les images terminant par I0.png
    ch = fnmatch.filter(os.listdir(), '*' + '*I90.png')
    # on convertis la liste en tableau array
    ch = np.array(ch)
    # on compte le nombre d'image
    nb = len(ch)
    print("nb img : ", nb)
    j = 0
    init = ([tf.global_variables_initializer(), tf.local_variables_initializer()])
    with tf.Session() as sess:
        sess.run(init)

        for i in range(nb):
            os.chdir(path)
            os.chdir(path_input)
            # on recupere l'index de tous les images
            index, _ = ch[i].split("_")

            I0 = "none"
            I45 = "none"
            I90 = "none"
            I135 = "none"

            I0 = fnmatch.filter(os.listdir('.'), str(index) + '*I0.png')
            I45 = fnmatch.filter(os.listdir('.'), str(index) + '*I45.png')
            I90 = fnmatch.filter(os.listdir('.'), str(index) + '*I90.png')
            I135 = fnmatch.filter(os.listdir('.'), str(index) + '*I135.png')

            if I0 == "none" or I45 == "none" or I90 == "none" or I135 == "none":
                print("ignore ...")
            else:
                img_I0 = cv2.imread(I0[0], 0)
                img_I45 = cv2.imread(I45[0], 0)
                img_I90 = cv2.imread(I90[0], 0)
                img_I135 = cv2.imread(I135[0], 0)

                merged = cv2.merge((img_I0, img_I45, img_I90, img_I135))

                # normalized data to ]-1 , 1[

                merged = merged / 128 - 1
                merged = merged.reshape((batch_size, img_height, img_width, img_layer_B))

                errC2Init = sess.run([ComputeErrC2], feed_dict={x: merged})
                errC2Init = errC2Init[0]
                print("errC2Init : ", errC2Init)

                if errC2Init == 0.0:
                    os.chdir(path)
                    os.chdir(path_output)

                    print("success... ")

                    cv2.imwrite(str(j) + "_I0.png", (img_I0).astype(np.uint8))
                    cv2.imwrite(str(j) + "_I45.png", (img_I45).astype(np.uint8))
                    cv2.imwrite(str(j) + "_I90.png", (img_I90).astype(np.uint8))
                    cv2.imwrite(str(j) + "_I135.png", (img_I135).astype(np.uint8))
                    j += 1
                    if j == max_img:
                        break

    print("J : ", j)


# --------
ProcessInputImgB()
