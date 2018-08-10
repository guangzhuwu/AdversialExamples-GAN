# -*- coding:utf-8 -*-
"""
Authors: wuguangzhu@baidu.com
"""

# 导入mnist数据库
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/mnist", one_hot=True)

import tensorflow as tf

# 创建会话
sess = tf.Session()

# 定义输入变量
x = tf.placeholder(tf.float32, [None, 784])

# 定义参数
W1 = tf.Variable(tf.zeros([784, 300]))
b1 = tf.Variable(tf.zeros([300]))
out = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.zeros([300, 10]))
b2 = tf.Variable(tf.zeros([10]))
out = tf.matmul(out, W2) + b2

# 定义模型和激励函数
y = tf.nn.softmax(out)

# 定义模型保存对象
saver = tf.train.Saver([W1, b1, W2, b2])

# 恢复模型
saver.restore(sess, "mnist/ckp")

print("恢复模型成功！")


def predict(img):
    """
    Predict the number of the image.
    :param img:
    :return:
    """
    ret = sess.run(y, feed_dict={x: img})
    return ret.argmax()


# 取出一个测试图片
idx = 0
img = mnist.test.images[idx]

# 根据模型计算结果
ret = sess.run(y, feed_dict={x: img.reshape(1, 784)})

print("计算模型结果成功！")

# 显示测试结果
print("预测结果:%d" % (ret.argmax()))
print("实际结果:%d" % (mnist.test.labels[idx].argmax()))
