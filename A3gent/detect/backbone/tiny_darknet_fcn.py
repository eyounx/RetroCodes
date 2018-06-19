import numpy as np
import tensorflow as tf

SCALE = 32
GRID_W, GRID_H = 7, 7
N_ANCHORS = 5
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH = GRID_H * SCALE, GRID_W * SCALE, 3


def lrelu(x, leak):
    return tf.maximum(x, leak * x, name='relu')


def maxpool_layer(x, size, stride, name, collections=None, trainable=True):
    with tf.name_scope(name):
        x = tf.layers.max_pooling2d(x, size, stride, padding='SAME')

    return x


def conv_layer(x, kernel, depth, train_logical, name, use_bias=True, norm=True, collections=None, trainable=True):
    with tf.variable_scope(name):
        # x = tf.layers.conv2d(x, depth, kernel, padding='SAME',
        # 					 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), use_bias=use_bias)

        # print (x.get_shape())

        w = tf.get_variable("conv2d/kernel", shape=[kernel[0], kernel[1], x.shape[-1], depth],
                            initializer=tf.contrib.layers.xavier_initializer(), collections=collections,
                            trainable=trainable)
        x = tf.nn.conv2d(x, filter=w, strides=[1, 1, 1, 1], padding='SAME')
        # print (x.get_shape())
        if use_bias == True:
            b = tf.get_variable("conv2d/bias", shape=(depth,), initializer=tf.constant_initializer(0.0),
                                collections=collections, trainable=trainable)
            b = tf.reshape(b, [1, 1, 1, depth])
            x = x + b

        if norm:
            x = lrelu(x, 0.1)
    return x


def yolo_net(x, train_logical=False, n_class=1, collections=None, trainable=True):
    x = conv_layer(x, (3, 3), 16, train_logical, 'conv1', collections=collections, trainable=trainable)
    x = maxpool_layer(x, (2, 2), (2, 2), 'maxpool1', collections=collections, trainable=trainable)

    x = conv_layer(x, (3, 3), 32, train_logical, 'conv2', collections=collections, trainable=trainable)
    x = maxpool_layer(x, (2, 2), (2, 2), 'maxpool2', collections=collections, trainable=trainable)

    x = conv_layer(x, (1, 1), 16, train_logical, 'conv3', collections=collections, trainable=trainable)
    x = conv_layer(x, (3, 3), 128, train_logical, 'conv4', collections=collections, trainable=trainable)
    x = conv_layer(x, (1, 1), 16, train_logical, 'conv5', collections=collections, trainable=trainable)
    x = conv_layer(x, (3, 3), 128, train_logical, 'conv6', collections=collections, trainable=trainable)
    x = maxpool_layer(x, (2, 2), (2, 2), 'maxpool6', collections=collections, trainable=trainable)

    x = conv_layer(x, (1, 1), 32, train_logical, 'conv7', collections=collections, trainable=trainable)
    x = conv_layer(x, (3, 3), 256, train_logical, 'conv8', collections=collections, trainable=trainable)
    x = conv_layer(x, (1, 1), 32, train_logical, 'conv9', collections=collections, trainable=trainable)
    x = conv_layer(x, (3, 3), 256, train_logical, 'conv10', collections=collections, trainable=trainable)
    x = maxpool_layer(x, (2, 2), (2, 2), 'maxpool10', collections=collections, trainable=trainable)

    x = conv_layer(x, (1, 1), 64, train_logical, 'conv11', collections=collections, trainable=trainable)
    x = conv_layer(x, (3, 3), 512, train_logical, 'conv12', collections=collections, trainable=trainable)
    x = conv_layer(x, (1, 1), 64, train_logical, 'conv13', collections=collections, trainable=trainable)
    x = conv_layer(x, (3, 3), 512, train_logical, 'conv14', collections=collections, trainable=trainable)
    x = conv_layer(x, (1, 1), 128, train_logical, 'conv15', collections=collections, trainable=trainable)
    x = maxpool_layer(x, (2, 2), (2, 2), 'maxpool15', collections=collections, trainable=trainable)

    x = conv_layer(x, (3, 3), 512, train_logical, 'conv16', collections=collections, trainable=trainable)
    x = conv_layer(x, (1, 1), 128, train_logical, 'conv17', collections=collections, trainable=trainable)
    x = conv_layer(x, (3, 3), 512, train_logical, 'conv18', collections=collections, trainable=trainable)

    x = conv_layer(x, (1, 1), N_ANCHORS * (n_class + 5), train_logical, 'conv19', use_bias=True, norm=False,
                   collections=collections, trainable=trainable)

    y = tf.reshape(x, shape=(-1, GRID_H, GRID_W, N_ANCHORS, 4 + 1 + n_class), name='y')

    print(tf.global_variables())

    return y


def load_weights(sess, pre_model_path):
    saver = tf.train.Saver(max_to_keep=None)
    saver.restore(sess, pre_model_path)
    print('Weights loaded.')


def load_from_binary(weights_file, offset=4):
    # graph = tf.get_default_graph()

    class WeightReader:
        def __init__(self, weight_file):
            self.offset = 4
            self.all_weights = np.fromfile(weight_file, dtype='float32')

        def read_bytes(self, size):
            self.offset = self.offset + size
            return self.all_weights[self.offset - size:self.offset]

        def reset(self, offset=4):
            self.offset = offset

    weight_reader = WeightReader(weights_file)
    weight_reader.reset(offset)
    assign_kernel = []
    assign_bias = []
    for i in range(1, 20):
        # print (">>>>>>>>")
        # print (i)
        # # kernel = graph.get_tensor_by_name("conv{}/conv2d/kernel:0".format(i))
        # allv = tf.global_variables()
        # print (allv)
        # for v in allv:
        # 	print(v.name + " " + "conv{}/conv2d/kernel:0".format(i))
        # 	print("--" + str(v.name == "conv{}/conv2d/kernel:0".format(i)))

        kernel = [v for v in tf.local_variables() if v.name == "conv{}/conv2d/kernel:0".format(i)][0]

        t_kernel = weight_reader.read_bytes(np.prod(kernel.shape.as_list()))
        t_kernel = t_kernel.reshape(list(reversed(kernel.shape.as_list())))
        t_kernel = t_kernel.transpose([2, 3, 1, 0])

        # bias = graph.get_tensor_by_name("conv{}/conv2d/bias:0".format(i))
        bias = [v for v in tf.local_variables() if v.name == "conv{}/conv2d/bias:0".format(i)][0]
        t_bias = weight_reader.read_bytes(np.prod(bias.shape.as_list()))

        assign_kernel.append(tf.assign(kernel, t_kernel))
        assign_bias.append(tf.assign(bias, t_bias))
    return assign_kernel, assign_bias
