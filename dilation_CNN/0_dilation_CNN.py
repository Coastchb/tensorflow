import tensorflow as tf
import functools
import numpy as np

x_placeholder = tf.placeholder(tf.float32,(None, 10, 3))

conv = tf.layers.Conv1D(filters=1,
                        kernel_size=3,
                        strides=1,
                        dilation_rate=1,
                        padding='valid',
                        activation=functools.partial(tf.nn.leaky_relu,alpha=0.02),
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        name="dilation_CNN")

output = conv(x_placeholder)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output_shape = sess.run(tf.shape(output), feed_dict={x_placeholder: np.ones((2,10,3))})

    print(output_shape)