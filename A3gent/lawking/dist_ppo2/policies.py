import numpy as np
import tensorflow as tf
from baselines.a2c.utils import ortho_init, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype


def conv(x, scope, *, nf, rf, stride, pad='VALID', init_scale=1.0, data_format='NHWC', collections=None, trainable=True):
    if data_format == 'NHWC':
        channel_ax = 3
        strides = [1, stride, stride, 1]
        bshape = [1, 1, 1, nf]
    elif data_format == 'NCHW':
        channel_ax = 1
        strides = [1, 1, stride, stride]
        bshape = [1, nf, 1, 1]
    else:
        raise NotImplementedError
    nin = x.get_shape()[channel_ax].value
    wshape = [rf, rf, nin, nf]
    with tf.variable_scope(scope):
        w = tf.get_variable("w", wshape, initializer=ortho_init(init_scale), collections=collections, trainable=trainable)
        b = tf.get_variable("b", [1, nf, 1, 1], initializer=tf.constant_initializer(0.0), collections=collections, trainable=trainable)

        if trainable and not collections:
            tf.add_to_collection("l2_losses", tf.contrib.layers.l2_regularizer(1.0)(w))
            tf.add_to_collection("l2_losses", tf.contrib.layers.l2_regularizer(1.0)(b))

        if data_format == 'NHWC': b = tf.reshape(b, bshape)
        return b + tf.nn.conv2d(x, w, strides=strides, padding=pad, data_format=data_format)


def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0, collections=None, trainable=True):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale), collections=collections, trainable=trainable)
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias), collections=collections, trainable=trainable)

        if trainable and not collections:
            tf.add_to_collection("l2_losses", tf.contrib.layers.l2_regularizer(1.0)(w))
            tf.add_to_collection("l2_losses", tf.contrib.layers.l2_regularizer(1.0)(b))

        return tf.matmul(x, w)+b


def conv_to_fc(x):
    nh = np.prod([v.value for v in x.get_shape()[1:]])
    x = tf.reshape(x, [-1, nh])
    return x


def nature_cnn(unscaled_images, collections=None, trainable=True):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2), collections=collections, trainable=trainable))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), collections=collections, trainable=trainable))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), collections=collections, trainable=trainable))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2), collections=collections, trainable=trainable))

class LnLstmPolicy(object):
    def __init__(self, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact)
            vf = fc(h5, 'v', 1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(sess, ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(sess, ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class LstmPolicy(object):

    def __init__(self, ob_space, ac_space, nbatch, nsteps, scope="model", nlstm=256, reuse=False, collections=None, trainable=True):
        nenv = nbatch // nsteps

        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope(scope, reuse=reuse):
            h = nature_cnn(X, collections=collections, trainable=trainable)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm, collections=collections, trainable=trainable)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact, collections=collections, trainable=trainable)
            vf = fc(h5, 'v', 1, collections=collections, trainable=trainable)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(sess, ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(sess, ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy(object):

    def __init__(self, ob_space, ac_space, nbatch, nsteps, scope="model", reuse=False, collections=None, trainable=True): #pylint: disable=W0613
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n

        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        given_a = tf.placeholder(tf.int64, (nbatch,))

        with tf.variable_scope(scope, reuse=reuse):
            h = nature_cnn(X, collections=collections, trainable=trainable)
            if nsteps == -1:
                h = tf.nn.relu(fc(h, 'fc2', nh=256, init_scale=np.sqrt(2), collections=collections, trainable=trainable))
            pi = fc(h, 'pi', nact, init_scale=0.01, collections=collections, trainable=trainable)
            vf = fc(h, 'v', 1, collections=collections, trainable=trainable)[:,0]

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        # print(a0.shape)
        neglogp0 = self.pd.neglogp(a0)
        given_neglogp0 = self.pd.neglogp(given_a)
        self.initial_state = None

        def step(sess, ob, *_args, **_kwargs):
            # print(a0)
            # print(vf)
            # print(neglogp0)
            # print(X)
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def step_v2(sess, ob, *_args, **_kwargs):
            # print(a0)
            # print(vf)
            # print(neglogp0)
            # print(X)
            a, v, neglogp, logits = sess.run([a0, vf, neglogp0, pi], {X:ob})
            return a, v, self.initial_state, neglogp, logits

        def step_action(sess, ob, *_args, **_kwargs):
            a, v = sess.run([a0, vf], {X:ob})
            return a, v

        def value(sess, ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        def step_given_action(sess, ob, actions, *_args, **_kwargs):
            v, neglogp = sess.run([vf, given_neglogp0], {X:ob, given_a: actions})
            return actions, v, self.initial_state, neglogp

        self.X = X
        self.pi = pi
        self.vf = vf
        self.l2_loss = tf.add_n(tf.get_collection("l2_losses")) if trainable and not collections else None
        self.step = step
        self.step_v2 = step_v2
        self.step_action = step_action
        self.value = value
        self.step_given_action = step_given_action


class MlpPolicy(object):
    def __init__(self, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            h1 = activ(fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            pi = fc(h2, 'pi', actdim, init_scale=0.01)
            h1 = activ(fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(h2, 'vf', 1)[:,0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(sess, ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(sess, ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
