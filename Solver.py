import os, fnmatch
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize
import tensorflow as tf

# **************************************************************************************************
path = os.getcwd()
res = (500, 500)
max_img = 100
PIA = True
img_height = 500
img_width = 500

batch_size = 5
img_layer_A = 3
img_layer_B = 4
# *****************************************************************************************************

A = tf.cast(0.5 * np.array([[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1]]), tf.float32)
x = tf.placeholder(tf.float32, shape=[1, 500, 500, 4], name='x')


# *****************************************************************************************************
def EvalImageConstraintC1(xi, name="ImageConstraintC1"):
    err = 0
    with tf.variable_scope(name):
        for i in range(batch_size):
            x = tf.squeeze(xi[i])
            x = (x + 1) * 128

            img_I0 = x[:, :, 0]
            img_I45 = x[:, :, 1]
            img_I90 = x[:, :, 2]
            img_I135 = x[:, :, 3]

            S0 = tf.add(img_I0, img_I90)
            S1 = tf.subtract(img_I0, img_I90)
            S2 = tf.subtract(img_I45, img_I135)

            S = tf.cast(tf.reshape(tf.stack([S0, S1, S2]), shape=(3, 500 * 500)), tf.float32)
            I = tf.cast(tf.reshape(x, shape=(4, 500 * 500)), tf.float32)
            I_child = tf.matmul(A, S)

            delta1 = I - I_child

            err += tf.norm(delta1) / tf.add(tf.norm(I), tf.norm(I_child))
        return err


# *****************************************************************************************************
def EvalImageConstraintC2(xi, name="ImageConstraintC2"):
    err = 0
    with tf.variable_scope(name):
        for i in range(batch_size):
            x = tf.squeeze(xi[i])
            x = (x + 1) * 128

            img_I0 = x[:, :, 0]
            img_I45 = x[:, :, 1]
            img_I90 = x[:, :, 2]
            img_I135 = x[:, :, 3]

            S0 = img_I0 + img_I90
            S1 = img_I0 - img_I90
            S2 = img_I45 - img_I135
            S0 = S0 + 1

            PHI = tf.subtract(tf.div(tf.sqrt(S1 ** 2 + S2 ** 2), S0), 1)
            PHI = tf.maximum(PHI, 0)

            err += tf.div(tf.norm(PHI), 500 * 500)
        return err


# ******************************************************************************************************
def ProcessInputImgA(f="./input/rgb2pol/trainA"):
    os.chdir(path)
    X = []
    os.chdir(f)
    list_img = os.listdir(".")
    for i in range(max_img):
        im = cv2.imread(list_img[i], -1)
        image_resized = cv2.resize(im, res)
        image_resized = image_resized / 128 - 1
        image_resized = image_resized.reshape((1, img_height, img_width, img_layer_A))
        X.append(image_resized)

    print("Processing Input ImgA done ...")
    os.chdir(path)
    return X


# **************************************************************************************************

def ProcessInputImgB(f="./input/rgb2pol/trainB"):
    os.chdir(path)
    X = []
    os.chdir(f)
    ch = fnmatch.filter(os.listdir(), '*' + '*I0.png')
    ch = np.array(ch)
    print("len(ch) : ", len(ch))

    for i in range(max_img):
        index, _ = ch[i].split("_")

        I0 = I45 = I90 = I135 = None

        I0 = fnmatch.filter(os.listdir('.'), str(index) + '*I0.png')
        I45 = fnmatch.filter(os.listdir('.'), str(index) + '*I45.png')
        I90 = fnmatch.filter(os.listdir('.'), str(index) + '*I90.png')
        I135 = fnmatch.filter(os.listdir('.'), str(index) + '*I135.png')

        if I0 == None or I45 == None or I90 == None or I135 == None:
            print("we do not have the cople of 4 image in the index " + str(index))

        else:
            img_I0 = cv2.imread(I0[0], 0)
            img_I45 = cv2.imread(I45[0], 0)
            img_I90 = cv2.imread(I90[0], 0)
            img_I135 = cv2.imread(I135[0], 0)

            merged = cv2.merge((img_I0, img_I45, img_I90, img_I135))

            # normalized data to ]-1 , 1[

            merged = merged / 128 - 1
            merged = merged.reshape((1, img_height, img_width, img_layer_B))
            X.append(merged)

    print("Processing Input ImgB done ...")
    os.chdir(path)
    return X


# ***********************************************************************************************

def WriteOutputImgB(img, epoch, i):
    img = np.squeeze(img)
    img_I0 = img[:, :, 0]
    img_I45 = img[:, :, 1]
    img_I90 = img[:, :, 2]
    img_I135 = img[:, :, 3]

    cv2.imwrite(str(epoch) + "_" + str(i) + "_I0.jpg", ((img_I0 + 1) * 127.5).astype(np.uint8))
    cv2.imwrite(str(epoch) + "_" + str(i) + "_I45.jpg", ((img_I45 + 1) * 127.5).astype(np.uint8))
    cv2.imwrite(str(epoch) + "_" + str(i) + "_I90.jpg", ((img_I90 + 1) * 127.5).astype(np.uint8))
    cv2.imwrite(str(epoch) + "_" + str(i) + "_I135.jpg", ((img_I135 + 1) * 127.5).astype(np.uint8))

    print("cople of 4 images " + str(i) + " has been save ...")


# *********************************************************************************************
def WriteOutputImgA(img, epoch, i):
    img = np.squeeze(img)
    img = ((img + 1) * 128).astype(np.uint8)
    cv2.imwrite(str(epoch) + "_" + str(i) + ".jpg", img)
