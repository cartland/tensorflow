import input_data
print 'Reading MNIST_data/'
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# Placeholder for input data.
# Undefined number of images, each with 784 pixels.
x = tf.placeholder(tf.float32, [None, 784])

# Weights for each data point, and its significance for a possible output.
# 784 pixels, each with an effect on 10 possible outputs (digits 0-9).
W = tf.Variable(tf.zeros([784, 10]))
# Bias for each of the possible outputs.
# Some digits appear more often than others.
b = tf.Variable(tf.zeros([10]))

# softmax(evidence)
# softmax(x) = normalize(exp(x))
# Softmax shapes output into a probability distribution.
# y is a vector with 10 dimensions.
y = tf.nn.softmax(tf.matmul(x, W) + b)
# TODO: Figure out why x and W are swapped. I expect Wx instead of xW.


# Real result for each input, with 1-hot output.
y_ = tf.placeholder(tf.float32, [None, 10])

# TODO: Understand why cross-entropy works.
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# Tweak variables in order to improve the prediction.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


# Initalize.
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Train model 1000 times, each with 100 random samples of training data.
print 'Training 1000 times'
for i in range(1000):
  print 'Training batch %d' % i
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Prediction is the y value with greatest probability.
# Actual result is the maximum y_ value.
# correct_prediction is an list of booleans.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# Cast booleans to float, then find the mean to determine percent accuracy.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Should be 91%
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

