# -*- coding:utf-8 -*-
"""
Authors: wuguangzhu@baidu.com
"""

import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import mnist_model
from improved_wgan import Discriminator
from improved_wgan import Gan
from improved_wgan import Generator

noise_dim = 10
target_dim = 784
source_dim = target_dim + noise_dim
MB_SIZE = 16
mnist_data = input_data.read_data_sets('../data/mnist', one_hot=True)


def sample_source(mb_size):
    """
    Get source data.
    :param mb_size:
    :return:
    """
    x, _ = mnist_data.train.next_batch(mb_size)
    return x


def sample_noise(mb_size):
    """
    Get noise data.
    :param mb_size:
    :return:
    """
    # return np.zeros((mb_size, source_data), 'float32')
    return np.random.randn(mb_size, noise_dim).astype('float32') / 100.0


def plot(samples):
    """
    Plot imgs.
    :param samples:
    :return:
    """
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        if i >= 32:
            break
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    return fig


def get_target_data(mnist_data, target, mb_size=MB_SIZE):
    """
    Get target data.
    :param mnist_data:
    :param target:
    :param mb_size:
    :return:
    """
    result = []
    images, lables = mnist_data.train.next_batch(10 * mb_size)
    for i in xrange(len(images)):
        if lables[i][target] == 1:
            result.append(images[i])
        if len(result) == mb_size:
            return result
    while len(result) < mb_size:
        result.append(result[0])
    return result


target = 2
test_target = get_target_data(mnist_data, target, 100)
test_source = sample_source(100)
test_noise = sample_noise(100)
target_data = get_target_data(mnist_data, target)
rx = 0


def get_fixed_source():
    """

    :return:
    """
    return test_source[:MB_SIZE]


def get_next_source():
    """

    :return:
    """
    return sample_source(MB_SIZE)


def get_next_noise():
    """

    :return:
    """
    return sample_noise(MB_SIZE)


def get_next_target():
    """

    :return:
    """
    global target_data
    global rx
    rx = rx + 1
    if rx % 10 == 0:
        target_data = get_target_data(mnist_data, target)
    return target_data[0:MB_SIZE]


class AttackByGan(Gan):
    """
    Do attack!
    """

    def __init__(self, generator_shape, discriminator_shape):
        """
        __init__
        :param generator_shape:
        :param discriminator_shape:
        """
        self.name = "adversarial_examples"
        self.param_dict = {
            'LAMBDA': 10,
            'CRITIC_ITERS': 10,
            'LEARNING_RATE': 1e-4,
            'MB_SIZE': MB_SIZE,
            'ITERS': 10000
        }
        Gan.__init__(self,
                     Generator(self, shape=generator_shape),
                     Discriminator(self, shape=discriminator_shape)
                     )
        self.out_path = self.name + "_out"
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)

    def after_train_one_iteration(self, source_data, target_data, noise_data, iteration):
        """
        :param source_data:
        :param target_data:
        :param noise_data:
        :param iteration:
        :return:
        """
        Gan.after_train_one_iteration(self, source_data, target_data, noise_data, iteration)
        global test_source
        global test_target
        if iteration % 100 == 0:
            gen_samples = self.generate_sample(test_source, test_target, test_noise)
            s = []
            untargeted_success_cnt = 0.0
            success_cnt = 0.0
            total_cnt = 0.0
            for i in xrange(len(gen_samples)):
                src = mnist_model.predict([test_source[i]])
                gen = mnist_model.predict([gen_samples[i]])
                if src != target and gen == target:
                    success_cnt += 1
                    s.append(gen_samples[i] + 0.0)
                    s.append((gen_samples[i] - test_source[i] + 1.0) / 2.0)
                    # s.append(test_source[i])
                if src != target:
                    total_cnt += 1
                if src != gen:
                    untargeted_success_cnt += 1
            print("iteration %d, untargeted:%f, success:%f" %
                  (iteration,
                   untargeted_success_cnt / len(gen_samples),
                   success_cnt / total_cnt
                   ))
            if success_cnt < 32:
                zeros = np.zeros(target_dim)
                s.append(zeros)
                s.append(zeros)
                for i in xrange(32 - int(success_cnt) - 1):
                    s.append(gen_samples[i] + 0.0)
                    s.append((gen_samples[i] - test_source[i] + 1.0) / 2.0)
            save_path = os.path.join('{}', '{}.png').format(self.out_path, str(iteration))
            fig = plot(s)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)


def get_target_data(mnist_data, target):
    """
    :param mnist_data:
    :param target:
    :return:
    """
    target_data = []
    images, lables = mnist_data.train.next_batch(10 * MB_SIZE)
    for i in xrange(len(images)):
        if lables[i][target] == 1:
            target_data.append(images[i])
        if len(target_data) == MB_SIZE:
            return target_data
    while len(target_data) < MB_SIZE:
        target_data.append(target_data[0])
    return target_data

gan = AttackByGan(generator_shape=[source_dim, 512, 512, 256, target_dim],
                  discriminator_shape=[target_dim, 512, 64, 32]
                  )
gan.run(get_next_source_data=get_next_source,
        get_next_target_data=get_next_target,
        get_next_noise_data=get_next_noise,
        iterations=10000)
