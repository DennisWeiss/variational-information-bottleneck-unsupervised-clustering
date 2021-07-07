import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.distributions as tfd
import math
import sys

BETA = 0
EPOCHS = 40


tf.reset_default_graph()

config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = tf.InteractiveSession(config=config)


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_sample = mnist.train.num_examples


layer_1_size = 1024
layer_2_size = 1024
encoding_dimension = 256
output_dimension = 10

images = tf.placeholder(tf.float32, shape=[None, 28*28], name='images')
labels = tf.placeholder(tf.int64, shape=[None, 10], name='labels')


def weights(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)


def bias(shape, name):
    bias = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(bias)


def model(images, weights, bias):
    W1 = weights([28 * 28, layer_1_size], 'W1')
    b1 = bias([layer_1_size], 'b1')
    W2 = weights([layer_1_size, layer_2_size], 'W2')
    b2 = bias([layer_2_size], 'b2')
    W3 = weights([layer_2_size, 2 * encoding_dimension], 'W3')
    b3 = bias(([2 * encoding_dimension]), 'b3')
    W4 = weights([encoding_dimension, output_dimension], 'W4')
    b4 = bias([output_dimension], 'b4')

    layer_1 = tf.nn.relu(tf.add(tf.matmul(images, W1), b1))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, W2), b2))
    encoding_layer = tf.add(tf.matmul(layer_2, W3), b3)
    encoding_distribution = tf.contrib.distributions.NormalWithSoftplusScale(encoding_layer[:, :256], encoding_layer[:, 256:])
    output = tf.add(tf.matmul(encoding_distribution.sample(), W4), b4)

    return encoding_distribution, output


encoding, pred = model(images, weights, bias)


mixture_probabilities = list(map(lambda _: 1 / output_dimension, range(output_dimension)))
# prior = tf.contrib.distributions.Normal(0.0, 1.0)
prior = tf.contrib.distributions.Mixture(
    cat=tf.contrib.distributions.Categorical(probs=mixture_probabilities),
    components=list(map(lambda i: tf.contrib.distributions.Normal(0.0, 1.0), range(output_dimension)))
)
#prior_2 = tf.contrib.distributions.Normal(mu_zy, rho_zy)

class_loss = tf.losses.softmax_cross_entropy(logits=pred, onehot_labels=labels)

info_loss = 0
for mixture_component in prior.components:
    info_loss += tf.reduce_sum(tf.reduce_mean(tfd.kl_divergence(encoding, mixture_component), 0)) / math.log(2)

total_loss = class_loss + BETA * info_loss

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(pred, 1), tf.arg_max(labels, 1)), tf.float32))

IZY_bound = math.log(10, 2) - class_loss

IZX_bound = info_loss

batch_size = 100
steps_per_batch = int(mnist.train.num_examples / batch_size)

global_step = tf.contrib.framework.get_or_create_global_step()
learning_rate = tf.train.exponential_decay(1e-4, global_step, decay_steps=2*steps_per_batch,decay_rate=0.97, staircase=True)
opt = tf.train.AdamOptimizer(learning_rate, 0.5)

ma = tf.train.ExponentialMovingAverage(0.999, zero_debias=True)
ma_update = ma.apply(tf.model_variables())

train_tensor = tf.contrib.training.create_train_op(total_loss, opt, global_step, update_ops=[ma_update])

tf.global_variables_initializer().run()
merged_summary_op = tf.summary.merge_all()


def evaluate_test():
    IZY, IZX, acc = sess.run([IZY_bound, IZX_bound, accuracy], feed_dict={images: mnist.test.images, labels: mnist.test.labels})
    return IZY, IZX, acc, 1-acc


for epoch in range(EPOCHS):
    for step in range(int(steps_per_batch)):
        im, ls = mnist.train.next_batch(batch_size)
        _, c = sess.run([train_tensor, total_loss], feed_dict={images: im, labels: ls})
    print("{}: IZY_Test={:.2f}\t IZX_Test={:.2f}\t acc_Test={:.4f}\t err_Test={:.4f}".format(epoch, *evaluate_test()))
    sys.stdout.flush()
