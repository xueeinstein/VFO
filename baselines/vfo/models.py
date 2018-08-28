import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from baselines.a2c import utils
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, \
    seq_to_batch


def nature_cnn(unscaled_images, convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
               **conv_kwargs):
    """
    CNN from Nature paper.
    """
    out = tf.cast(unscaled_images, tf.float32) / 255.
    with tf.variable_scope("convnet"):
        for num_outputs, kernel_size, stride in convs:
            out = layers.convolution2d(out,
                                       num_outputs=num_outputs,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       activation_fn=tf.nn.relu,
                                       **conv_kwargs)

    return out


def cnn(convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], nh=512, **conv_kwargs):
    """
    cnn-flatten-fc net

    Parameters:
    -----------

    conv:       list of triples (filter_number, filter_size, stride) specifying parameters for each layer.
    nh:         number of hidden units for FC layer

    Returns:

    function that takes tensorflow tensor as input and returns the output of the last convolutional layer and output of fc layer after flatten cnn
    """
    def network_fn(X):
        fm = nature_cnn(X, convs=convs, **conv_kwargs)  # cnn feature map
        out = conv_to_fc(fm)
        out = tf.nn.relu(fc(out, 'fc1', nh=nh, init_scale=np.sqrt(2)))

        return fm, out, None

    return network_fn


def cnn_small(**conv_kwargs):
    def network_fn(X):
        h = tf.cast(X, tf.float32) / 255.

        activ = tf.nn.relu
        h = activ(conv(h, 'c1', nf=8, rf=8, stride=4, init_scale=np.sqrt(2),
                       **conv_kwargs))
        h = activ(conv(h, 'c2', nf=16, rf=4, stride=2, init_scale=np.sqrt(2),
                       **conv_kwargs))
        fm = h
        h = conv_to_fc(h)
        h = activ(fc(h, 'fc1', nh=128, init_scale=np.sqrt(2)))
        return fm, h, None
    return network_fn


def cnn_lstm(nh=512, nlstm=128, layer_norm=False, **conv_kwargs):
    def network_fn(X, nenv=1):
        nbatch = X.shape[0]
        nsteps = nbatch // nenv

        fm = nature_cnn(X, **conv_kwargs)
        fm_flat = conv_to_fc(fm)
        h = tf.nn.relu(fc(fm_flat, 'fc1', nh=nh, init_scale=np.sqrt(2)))

        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, 2*nlstm])  # states

        xs = batch_to_seq(h, nenv, nsteps)
        ms = batch_to_seq(M, nenv, nsteps)

        if layer_norm:
            h5, snew = utils.lnlstm(xs, ms, S, scope='lnlstm', nh=nlstm)
        else:
            h5, snew = utils.lstm(xs, ms, S, scope='lstm', nh=nlstm)

        h = seq_to_batch(h5)
        initial_state = np.zeros(S.shape.as_list(), dtype=float)

        return fm, h, {'S': S, 'M': M, 'state': snew, 'initial_state': initial_state}

    return network_fn


def get_network_builder(network_type):
    if network_type == 'cnn':
        return cnn
    elif network_type == 'cnn_small':
        return cnn_small
    elif network_type == 'cnn_lstm':
        return cnn_lstm
    else:
        raise NotImplementedError


def nn_discriminator(num_options=64, hidden_sizes=[100, 100],
                     activation=tf.nn.relu):
    """
    Discriminator to evaluate which option to use for given observations
    This discriminator is modeled as posterior q(z | s)

    Parameters:
    -----------

    num_options:      number of options to evaluate

    hidden_sizes:     size of fully-connected layers

    activation:       type of activation function for FC hidden layers, for output layer, the activation function is none

    Returns:
    -------

    function that builds FC network with given latent state tensor
    """
    def network_fn(S):
        h = tf.layers.flatten(S)
        for i, hs in enumerate(hidden_sizes):
            if i != len(hidden_sizes) - 1:
                h = activation(fc(h, 'fc_{}'.format(i), nh=hs,
                                  init_scale=np.sqrt(2)))
            else:
                h = fc(h, 'eval', nh=num_options,
                       init_scale=np.sqrt(2))

        return h

    return network_fn
