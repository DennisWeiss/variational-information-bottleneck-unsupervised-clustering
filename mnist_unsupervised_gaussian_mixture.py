import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.distributions as tfd
import sklearn.metrics
import sklearn.cluster
import numpy as np
import seaborn as sns
import math
import sys
import matplotlib.pyplot as plt

import utils

BETA = 1.0
EPOCHS = 200

tf.logging.set_verbosity(tf.logging.ERROR)
tf.reset_default_graph()

config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = tf.InteractiveSession(config=config)


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_sample = mnist.train.num_examples


layer_1_size = 500
layer_2_size = 500
layer_3_size = 2000
encoding_dimension = 10
layer_5_size = 2000
layer_6_size = 500
layer_7_size = 500

images = tf.placeholder(tf.float32, shape=[None, 28*28], name='images')
labels = tf.placeholder(tf.int64, shape=[None, 10], name='labels')


def weights(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)


def bias(shape, name):
    bias = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(bias)


def encoder(images, weights, bias):
    W1 = weights([28 * 28, layer_1_size], 'W1')
    b1 = bias([layer_1_size], 'b1')
    W2 = weights([layer_1_size, layer_2_size], 'W2')
    b2 = bias([layer_2_size], 'b2')
    W3 = weights([layer_2_size, layer_3_size], 'W3')
    b3 = bias([layer_3_size], 'b3')
    W4 = weights([layer_3_size, 2 * encoding_dimension], 'W4')
    b4 = bias(([2 * encoding_dimension]), 'b4')

    layer_1 = tf.nn.relu(tf.add(tf.matmul(images, W1), b1))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, W2), b2))
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, W3), b3))
    encoding_layer = tf.add(tf.matmul(layer_3, W4), b4)
    encoding_distribution = tfd.NormalWithSoftplusScale(encoding_layer[:, :encoding_dimension], encoding_layer[:, encoding_dimension:])

    return encoding_distribution


def decoder(encoding_distribution, weights, bias):
    W5 = weights([encoding_dimension, layer_5_size], 'W5')
    b5 = bias([layer_5_size], 'b5')
    W6 = weights([layer_5_size, layer_6_size], 'W6')
    b6 = bias([layer_6_size], 'b6')
    W7 = weights([layer_6_size, layer_7_size], 'W7')
    b7 = bias([layer_7_size], 'b7')
    W8 = weights([layer_7_size, 28 * 28], 'W8')
    b8 = bias([28 * 28], 'b8')

    layer_5 = tf.nn.relu(tf.add(tf.matmul(encoding_distribution.sample(), W5), b5))
    layer_6 = tf.nn.relu(tf.add(tf.matmul(layer_5, W6), b6))
    layer_7 = tf.nn.relu(tf.add(tf.matmul(layer_6, W7), b7))
    output_distribution = tf.nn.sigmoid(tf.add(tf.matmul(layer_7, W8), b8))

    return output_distribution


def model(images, weights, bias):
    encoding_distribution = encoder(images, weights, bias)
    output_distribution = decoder(encoding_distribution, weights, bias)
    return encoding_distribution, output_distribution


encoding_distribution, output_distribution = model(images, weights, bias)

mixture_probabilities = list(map(lambda _: 1 / 10, range(10)))
mixture_components = list(map(lambda i: tfd.Normal(0.0, 1.0), range(10)))

prior = tfd.Mixture(
    cat=tfd.Categorical(probs=mixture_probabilities),
    components=mixture_components
)

class_loss = tf.reduce_mean(-tf.reduce_sum(images * tf.log(output_distribution + 1e-14) + (1 - images) * tf.log(1 - output_distribution + 1e-14), axis=1))

info_loss = tf.constant(0.0)
for mixture_component in prior.components:
    info_loss += tf.exp(-tf.reduce_mean(tf.reduce_mean(tfd.kl_divergence(encoding_distribution, mixture_component), 0)) / math.log(2)) / 10

info_loss = -tf.math.log(info_loss + 1e-14)

total_loss = class_loss + BETA * info_loss

# accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(output_distribution, 1), tf.arg_max(labels, 1)), tf.float32))

batch_size = 100
steps_per_batch = int(mnist.train.num_examples / batch_size)

global_step = tf.contrib.framework.get_or_create_global_step()
learning_rate = tf.train.exponential_decay(1e-4, global_step, decay_steps=2*steps_per_batch, decay_rate=0.97, staircase=True)
opt = tf.train.AdamOptimizer(learning_rate, 0.5)

ma = tf.train.ExponentialMovingAverage(0.999, zero_debias=True)
ma_update = ma.apply(tf.model_variables())

train_tensor = tf.contrib.training.create_train_op(total_loss, opt, global_step, update_ops=[ma_update])

tf.global_variables_initializer().run()
merged_summary_op = tf.summary.merge_all()


def evaluate(data):
    class_loss_value, info_loss_value, total_loss_value, encoding_value = sess.run([class_loss, info_loss, total_loss, encoding_distribution.sample()], feed_dict={images: data.images, labels: data.labels})
    return class_loss_value, info_loss_value, total_loss_value, encoding_value


def get_unsupervised_clustering_accuracy(M, permutation):
    correct = 0
    for i in range(M.shape[0]):
        correct += M[i, permutation[i]]
    return correct / np.sum(M)


for epoch in range(EPOCHS):
    for step in range(int(steps_per_batch)):
        im, ls = mnist.train.next_batch(batch_size)
        sess.run([train_tensor], feed_dict={images: im, labels: ls})

    train_class_loss_value, train_info_loss_value, train_total_loss_value, train_encoding_value = evaluate(mnist.train)
    test_class_loss_value, test_info_loss_value, test_total_loss_value, test_encoding_value = evaluate(mnist.test)

    clustering = sklearn.cluster.KMeans(n_clusters=10).fit_predict(test_encoding_value)
    actual_labels = np.argmax(mnist.test.labels, axis=1)
    confusion_matrix = sklearn.metrics.confusion_matrix(actual_labels, clustering)
    maximum_weight_permutation = utils.maximum_weight_permutation(confusion_matrix)

    unsupervised_clustering_accuracy = get_unsupervised_clustering_accuracy(confusion_matrix, maximum_weight_permutation)

    sns.heatmap(confusion_matrix, annot=True, fmt='d')
    plt.show()

    print("After epoch {}: train_class_loss={:.3f}\t train_info_loss={:.3f}\t train_total_loss={:.5f}\n\t\t\t   test_class_loss={:.3f}\t test_info_loss={:.3f}\t test_total_loss={:.5f}\t unsupervised_clustering_accuracy={:.2f}%".format(epoch + 1, train_class_loss_value, train_info_loss_value, train_total_loss_value, test_class_loss_value, test_info_loss_value, test_total_loss_value, 100 * unsupervised_clustering_accuracy))
    # for mixture_component in prior.components:
    #     mixture_component_value = sess.run(mixture_component)
    #     print(mixture_component_value.loc)
    sys.stdout.flush()



