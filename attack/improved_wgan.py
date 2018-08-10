# -*- coding:utf-8 -*-
"""
Authors: guangzhuwu@gmail.com
"""

import tensorflow as tf

from gan_base import DiscriminatorBase
from gan_base import GanBase
from gan_base import GeneratorBase


class Generator(GeneratorBase):
    """
    Generator
    """

    def generate(self, source_data, target_data, noise_data):
        """
        :return: The generated adversarial data of the generator.
        """
        v = tf.concat([source_data, noise_data], 1)
        for i in xrange(len(self.var_list) / 2 - 1):
            i *= 2
            w = self.var_list[i]
            b = self.var_list[i + 1]
            v = tf.nn.relu(tf.matmul(v, w) + b)
        # out layer
        w = self.var_list[-2]
        b = self.var_list[-1]
        v = tf.nn.tanh(tf.matmul(v, w) + b)

        alf = 0.2
        beta = 1.0
        return (alf * v + beta * source_data + alf) / (beta + 2.0 * alf)
        # result = alf * v + beta * source_data
        # return tf.nn.relu(result) - tf.nn.relu(result - 1.0)

    def train_one_iteration(self, source_data, target_data, noise_data, iteration):
        """
        :param source_data:
        :param target_data:
        :param noise_data:
        :param iteration:
        :return:
        """
        self.gan.session.run(
            self.optimizer,
            feed_dict={self.gan.target_data: target_data,
                       self.gan.source_data: source_data,
                       self.gan.noise_data: noise_data
                       }
        )


class Discriminator(DiscriminatorBase):
    """
    Discriminator
    """

    def _discriminate(self, in_data):
        """
        :param in_data:
        :return:
        """
        v = in_data
        # hidden layer
        for i in xrange(len(self.var_list) / 2 - 1):
            i *= 2
            w = self.var_list[i]
            b = self.var_list[i + 1]
            v = tf.nn.relu(tf.matmul(v, w) + b)
        # out layer
        w = self.var_list[-2]
        b = self.var_list[-1]
        v = tf.matmul(v, w) + b
        return v

    def construct_loss(self, generator):
        """
        Construct the loss function.
        :param generator:
        :return:
        """
        fake_data = generator.generate(self.gan.source_data, self.gan.target_data,
                                       self.gan.noise_data)
        real_data = self.gan.target_data
        d_fake = self._discriminate(fake_data)
        d_real = self._discriminate(real_data)

        residual_data = (fake_data - real_data + 1.0) / 2.0  # 让背景长的不像target
        d_residual_data = self._discriminate(residual_data)

        add_loss = tf.sqrt(tf.reduce_sum(tf.square(fake_data - self.gan.source_data),
                                         reduction_indices=[1]))

        # Discriminator loss
        epsilon = tf.random_uniform([self.gan.param_dict["MB_SIZE"], 1], minval=0.0, maxval=1.0)
        x_hat = epsilon * real_data + (1.0 - epsilon) * fake_data
        # ∇xˆDw(xˆ)
        grad = tf.gradients(self._discriminate(x_hat), [x_hat])[0]
        # ||∇xˆDw(xˆ)||2
        grad_l2_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), reduction_indices=[1]))
        # λ(||∇xˆDw(xˆ)||2 − 1)^2
        grad_penalty = self.gan.param_dict["LAMBDA"] * tf.reduce_mean((grad_l2_norm - 1.0) ** 2)

        d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(
            d_real) + grad_penalty + 0.15 * add_loss + 0.3 * tf.reduce_mean(d_residual_data)

        self.add_additional_loss(d_loss)

        # Generator loss
        g_loss = - tf.reduce_mean(d_fake) + 0.01 * add_loss
        generator.add_additional_loss(g_loss)

    def train_one_iteration(self, source_data, target_data, noise_data, iteration):
        """
        :param source_data:
        :param target_data:
        :param noise_data:
        :param iteration:
        :return:
        """
        for _ in xrange(self.gan.param_dict['CRITIC_ITERS']):
            self.gan.session.run(
                self.optimizer,
                feed_dict={self.gan.target_data: target_data,
                           self.gan.source_data: source_data,
                           self.gan.noise_data: noise_data
                           }
            )


class Gan(GanBase):
    """
    GAN
    """

    def train_one_iteration(self, source_data, target_data, noise_data, iteration):
        """
        :param source_data:
        :param target_data:
        :param noise_data:
        :param iteration:
        :return:
        """
        self.generator.train_one_iteration(source_data, target_data, noise_data, iteration)
        for d in self.discriminator:
            d.train_one_iteration(source_data, target_data, noise_data, iteration)

    def before_train_one_iteration(self, source_data, target_data, noise_data, iteration):
        """
        :param source_data:
        :param target_data:
        :param noise_data:
        :param iteration:
        :return:
        """
        self.generator.before_train_one_iteration(source_data, target_data, noise_data, iteration)
        for d in self.discriminator:
            d.before_train_one_iteration(source_data, target_data, noise_data, iteration)

    def after_train_one_iteration(self, source_data, target_data, noise_data, iteration):
        """
        :param source_data:
        :param target_data:
        :param noise_data:
        :param iteration:
        :return:
        """
        self.generator.after_train_one_iteration(source_data, target_data, noise_data, iteration)
        for d in self.discriminator:
            d.after_train_one_iteration(source_data, target_data, noise_data, iteration)
