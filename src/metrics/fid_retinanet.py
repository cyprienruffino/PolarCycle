import tensorflow as tf
from deeplauncher.metrics.abstractmetric import AbstractMetric
from keras_retinanet.models import load_model


class FIDRetinanet(AbstractMetric):

    @staticmethod
    def __tf_cov(x):
        mean_x = tf.reduce_mean(x, axis=0, keep_dims=True)
        mx = tf.matmul(tf.transpose(mean_x), mean_x)
        vx = tf.matmul(tf.transpose(x), x) / tf.cast(tf.shape(x)[0], tf.float32)
        cov_xx = vx - mx
        return cov_xx

    @staticmethod
    def __tf_sqrtm_sym(mat, eps=1e-10):
        # WARNING : This only works for symmetric matrices !
        s, u, v = tf.compat.v1.svd(mat)
        si = tf.where(tf.less(s, eps), s, tf.sqrt(s))
        return tf.matmul(tf.matmul(u, tf.compat.v1.diag(si)), v, transpose_b=True)

    def __init__(self, model_path, nb_data, img_shape, lays_cut=1, batch_size=8):
        self.model_path = model_path
        self.nb_data = nb_data
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.lays_cut = lays_cut

    def build(self):

        with tf.name_scope("FID_activations"):
            cl = load_model(self.model_path, backbone_name="resnet50")
            self.fidmodel = tf.compat.v1.keras.Model(cl.inputs, cl.layers[-self.lays_cut - 1].output)

        with tf.name_scope("FID"):
            self.imgs_real = tf.compat.v1.placeholder(tf.float32, shape=(self.nb_data, ) + self.img_shape)
            self.imgs_fake = tf.compat.v1.placeholder(tf.float32, shape=(self.nb_data, ) + self.img_shape)

            acts_real = tf.reshape(self.fidmodel.predict(self.imgs_real, batch_size=self.batch_size), (self.batch_size, -1))
            acts_fake = tf.reshape(self.fidmodel.predict(self.imgs_fake, batch_size=self.batch_size), (self.batch_size, -1))

            mu_real = tf.reduce_mean(acts_real, axis=0)
            mu_fake = tf.reduce_mean(acts_fake, axis=0)

            sigma_real = self.__tf_cov(acts_real)
            sigma_fake = self.__tf_cov(acts_fake)
            diff = mu_real - mu_fake
            mu2 = tf.reduce_sum(tf.multiply(diff, diff))

            # Computing the sqrt of sigma_real * sigma_fake
            # See https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py
            sqrt_sigma = self.__tf_sqrtm_sym(sigma_real)
            sqrt_a_sigma_a = tf.matmul(sqrt_sigma, tf.matmul(sigma_fake, sqrt_sigma))

            tr = tf.compat.v1.trace(sigma_real + sigma_fake - 2 * self.__tf_sqrtm_sym(sqrt_a_sigma_a))
            self.fid = mu2 + tr

    def evaluate(self, real, out, sess):
        return sess.run(self.fid, feed_dict={self.imgs_real: real, self.imgs_fake: out})
