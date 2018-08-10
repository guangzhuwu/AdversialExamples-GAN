# -*- coding:utf-8 -*-
"""
Authors: wuguangzhu@baidu.com
"""

import abc

import tensorflow as tf


def xavier_init(size):
    """
    Return an xavier inited tensor.

    :param size: the size of the tensor
    :return: A tensor of the specified size filled with xavier inited values.
    """
    in_dim = size[0]
    xavier_stddev = 1.0 / tf.sqrt(in_dim / 2.0)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


class Trainable(object):
    """
    Trainable is an abstract base class that can be trained.
    """
    __metaclass__ = abc.ABCMeta

    def before_train_one_iteration(self, source_data, target_data, noise_data, iteration):
        """一次训练迭代前要做的"""
        pass

    @abc.abstractmethod
    def train_one_iteration(self, source_data, target_data, noise_data, iteration):
        """一次训练迭代"""
        pass

    def after_train_one_iteration(self, source_data, target_data, noise_data, iteration):
        """一次训练迭代后要做的"""
        pass


class AdversaryBase(Trainable):
    """
    The base class of Generator/Discriminator and GAN.
    """

    def __init__(self, gan, shape):
        """
        :param shape:
        """
        self.shape = shape  # The NN layer shape
        self.loss = None
        self.loss_placeholder = None
        self.optimizer = None
        self.gan = gan
        self.processor = None

        # Construct the var_list and optimizer
        self.var_list = []
        for i in xrange(len(self.shape) - 1):
            w = tf.Variable(xavier_init([self.shape[i], self.shape[i + 1]]))
            b = tf.Variable(tf.zeros(self.shape[i + 1]))
            self.var_list.append(w)
            self.var_list.append(b)

    def add_additional_loss(self, loss):
        """
        Add additional loss.
        :param loss:
        :return:
        """
        if self.loss is None:
            self.loss = loss
        else:
            self.loss += loss

    def construct_optimizer(self):
        """
        Construct optimizer. This function must be called after the loss has been constructed.
        :return: Void
        """
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss, var_list=self.var_list)


class GeneratorBase(AdversaryBase):
    """
    Generator base.
    """

    @abc.abstractmethod
    def generate(self, source_data, target_data, noise_data):
        """
        Construct the data flow layer by layer.
        :return: The generated data.
        """


class DiscriminatorBase(AdversaryBase):
    """
    Discriminator base.
    """

    @abc.abstractmethod
    def construct_loss(self, generator):
        """
        Construct the loss function.
        :param generator:
        :return:
        """


class GanBase(Trainable):
    """
    Base class of GAN
    """

    def __init__(self, generator, *discriminators):
        """
        :param generator:
        :param discriminator:
        """
        self.session = None
        self.generator = generator  # type:GeneratorBase
        self.discriminator = discriminators  # type:DiscriminatorBase

        source_dim = generator.shape[-1]
        target_dim = generator.shape[-1]
        noise_dim = generator.shape[0] - source_dim

        self.source_data = tf.placeholder(tf.float32, shape=[None, source_dim])
        self.target_data = tf.placeholder(tf.float32, shape=[None, target_dim])
        self.noise_data = tf.placeholder(tf.float32, shape=[None, noise_dim])

        # Construct loss. The loss of generator will be constructed by discriminators.
        for d in discriminators:
            d.construct_loss(generator)

        # Construct optimizer
        generator.construct_optimizer()
        for d in discriminators:
            d.construct_optimizer()

    def run(self, get_next_source_data, get_next_target_data, get_next_noise_data, iterations):
        """
        Run
        :param get_next_source_data:
        :param get_next_target_data:
        :param get_next_noise_data:
        :param iterations:
        :return:
        """

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        for iteration in xrange(iterations):
            source_data = get_next_source_data()
            target_data = get_next_target_data()
            noise_data = get_next_noise_data()
            self.before_train_one_iteration(source_data, target_data, noise_data, iteration)
            self.train_one_iteration(source_data, target_data, noise_data, iteration)
            self.after_train_one_iteration(source_data, target_data, noise_data, iteration)

    def generate_sample(self, source, target, noise):
        """
        Generate adversarial examples.
        :param source:
        :param target:
        :param noise:
        :return:
        """
        return self.session.run(
            self.generator.generate(self.source_data, self.target_data, self.noise_data),
            feed_dict={self.source_data: source,
                       self.target_data: target,
                       self.noise_data: noise}
        )
