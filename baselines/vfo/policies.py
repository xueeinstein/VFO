import numpy as np
import tensorflow as tf
from baselines.a2c.utils import fc
from baselines.common import tf_util
from baselines.common.distributions import make_pdtype
from baselines.common.policies import _normalize_clip_observation
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.tf_util import adjust_shape
from baselines.vfo.models import get_network_builder, nn_discriminator
from baselines.vfo.utils import get_action_dim


class PolicyWithValue(object):
    """
    Encapsulates fields and methods for RL policy, options policy and value
    function estimation with shared parameters
    """
    def __init__(self, env, observations, next_observations, actions, option_z,
                 dones, feature_map, next_feature_map, latent, option_latent,
                 q_latent, vf, next_vf, sess=None, **tensors):
        """
        Parameters:
        -----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        next_observations tensorflow placeholder in which the next observations will be fed

        actions         tensorflow placeholder in which the actions will be fed

        option_z        tensorflow placeholder in which the option one-hot will be fed

        dones           tensorflow placeholder in which whether env terminal is true

        feature_map     feature map (tensorflow tensor) from last conv layer

        next_feature_map feature map for next observaion which is used to train option policy

        latent          latent state from which policy distribution parameters should be inferred

        option_latent   latent state from which option policy distribution parameters should be inferred

        q_latent        latent state from which soft q function for option should be inferred

        vf              value function (tensorflow tensor)

        next_vf         value function for next observaion which is used to train option policy

        sess            tensorflow session to run calculations in (if None, default session is used)

        data_format     data format of Conv layer

        **tensors       tensorflow tensors for additional attributes such as state or mask
        """
        self.X = observations
        self.X_next = next_observations
        self.ac = actions
        self.op_z = option_z
        self.dones = dones
        self.noptions = option_z.get_shape().as_list()[-1]
        self.prior_op_z = np.full(self.noptions, 1.0 / self.noptions)
        self.state = tf.constant([])
        self.initial_state = None
        self.__dict__.update(tensors)

        self.fm = feature_map
        self.fm_next = next_feature_map
        self.vf = vf
        self.vf_next = next_vf

        latent = tf.layers.flatten(latent)
        self.pdtype = make_pdtype(env.action_space)
        self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=0.01)

        self.action = self.pd.sample()
        self.deterministic_action = self.pd.mode()
        self.neglogp = self.pd.neglogp(self.action)
        self.sess = sess

        with tf.variable_scope('option'):
            self.option_pdtype = make_pdtype(env.action_space)
            self.option_pd, self.option_pi = self.option_pdtype.pdfromlatent(
                option_latent, init_scale=0.01)
            self.option_action = self.option_pd.sample()
            self.deterministic_option_action = self.option_pd.mode()
            self.option_neglogp = self.option_pd.neglogp(self.action)

            # soft q function for option
            self.option_q = fc(q_latent, 'option_q', 1)
            # vf-feature map activation grad
            va_grads = tf.gradients(self.vf, self.fm, name='vf_fm_grad')[0]
            # assume data_format 'NHWC'
            # FIXME: get wrong pvfs
            print('vf:', self.vf.get_shape().as_list())
            print('fm:', self.fm.get_shape().as_list())
            print('va_grads:', va_grads.get_shape().as_list())
            self.pvfs = tf.reduce_sum(tf.multiply(self.fm, va_grads), axis=[1, 2])
            print('pvfs:', self.pvfs.get_shape().as_list())

            next_va_grads = tf.gradients(self.vf_next, self.fm_next,
                                         name='next_vf_fm_grad')[0]
            self.next_pvfs = tf.reduce_sum(
                tf.multiply(self.fm_next, next_va_grads), axis=[1, 2])
            print('pvfs_next:', self.next_pvfs.get_shape().as_list())

            with tf.variable_scope('discriminator'):
                no_grad_latent = tf.stop_gradient(latent)
                self.option_discriminator, self.option_discriminator_logits = \
                    nn_discriminator(num_options=self.noptions)(no_grad_latent)

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess or tf.get_default_session()
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def step(self, observation, stochastic=True, **extra_feed):
        """
        Compute next action(s) given the observaion(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """
        if stochastic:
            action = self.action
        else:
            action = self.deterministic_action

        a, v, state, neglogp = self._evaluate(
            [action, self.vf, self.state, self.neglogp],
            observation, **extra_feed)
        if state.size == 0:
            state = None
        return a, v, state, neglogp

    def option_step(self, option_z, observation, stochastic=True, **extra_feed):
        """
        Compute next action(s) given the observaion(s) and option one_hot
        """
        if stochastic:
            action = self.option_action
        else:
            action = self.deterministic_option_action

        extra_feed.update({'op_z': option_z})
        a, state, neglogp = self._evaluate(
            [action, self.state, self.option_neglogp],
            observation, **extra_feed)
        if state.size == 0:
            state = None
        return a, state, neglogp

    def option_select(self, observation, **extra_feed):
        """
        Run option discriminator
        """
        discriminator_value = self._evaluate(
            self.option_discriminator, observation, **extra_feed)
        return discriminator_value

    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observaion(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.vf, ob, *args, **kwargs)

    def proto_value(self, option_z, ob, *args, **kwargs):
        pvs = self._evaluate(self.pvfs, ob, *args, **kwargs)  # [N, C]
        batch_op_z = tf.tile(tf.expand_dims(option_z, 0), [tf.shape(ob)[0], 1])

        pv = tf.reduce_sum(tf.multiply(pvs, batch_op_z), 1)
        return pv

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)


def global_average_pooling(h, data_format='NHWC', **kwargs):
    tf.assert_rank(h, 4, message='Rank of input tensor for GAP should be 4')
    out = h
    with tf.variable_scope('gap'):
        if data_format == 'NHWC':
            out = tf.reduce_mean(out, [1, 2])
        elif data_format == 'NCHW':
            out = tf.reduce_mean(out, [2, 3])
        else:
            raise NotImplementedError

    return out


def build_policy(env, policy_network, noptions, value_network=None,
                 normalize_observations=False, estimate_q=False,
                 **policy_kwargs):
    assert isinstance(policy_network, str), 'only accept string'
    assert policy_network.startswith('cnn'), 'VFO currently supports CNN based policy network'
    network_type = policy_network
    policy_network = get_network_builder(network_type)(**policy_kwargs)

    def pi_vf_fn(X, extra_tensors, nbatch, nsteps, recurrent_subname=None):
        """Shared network to extract latent feature for ob, ob_next"""
        ob_space = env.observation_space
        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X

        encoded_x = encode_observation(ob_space, encoded_x)

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            cnn_fm, policy_latent, recurrent_tensors = policy_network(encoded_x)

            if recurrent_tensors is not None:
                # recurrent architecture, need a few more steps
                nenv = nbatch // nsteps
                assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
                cnn_fm, policy_latent, recurrent_tensors = policy_network(
                    encoded_x, nenv)

                if recurrent_subname is not None:
                    new_recurrent_tensors = {}
                    for k, v in recurrent_tensors.items():
                        new_recurrent_tensors[recurrent_subname + '_' + k] = v
                    extra_tensors.update(new_recurrent_tensors)
                else:
                    extra_tensors.update(recurrent_tensors)

        with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
            _v_net = value_network

            if _v_net is None or _v_net == 'shared':
                vf_latent = policy_latent
            elif _v_net == 'gap':
                vf_latent = global_average_pooling(cnn_fm, **policy_kwargs)
            else:
                raise NotImplementedError

            vf = fc(vf_latent, 'vf_fc', 1)[:, 0]

        return cnn_fm, policy_latent, vf

    def policy_fn(noptions=64, nbatch=None, nsteps=None, sess=None,
                  observ_placeholder=None, action_placeholder=None,
                  option_z_placeholder=None, dones_placeholder=None):
        ob_space = env.observation_space
        ac_n = get_action_dim(env)

        X = observ_placeholder if observ_placeholder is not None else \
            observation_placeholder(ob_space, batch_size=nbatch, name='ob')
        X_next = observation_placeholder(
            ob_space, batch_size=nbatch, name='ob_next')
        ac = action_placeholder if action_placeholder is not None else \
            tf.placeholder(tf.float32, shape=(nbatch, ac_n), name='ac')
        op = option_z_placeholder if option_z_placeholder is not None else \
            tf.placeholder(tf.float32, shape=(nbatch, noptions), name='op_z')
        dones = dones_placeholder if dones_placeholder is not None else \
            tf.placeholder(tf.float32, shape=(nbatch), name='dones')

        extra_tensors = {}

        cnn_fm, policy_latent, vf = pi_vf_fn(X, extra_tensors, nbatch, nsteps)
        next_cnn_fm, _, next_vf = pi_vf_fn(
            X_next, extra_tensors, nbatch, nsteps, recurrent_subname='next')
        assert noptions == cnn_fm.get_shape().as_list()[-1], \
            'number of options for VFO should equal to channels of last conv layer'

        tf.assert_rank(policy_latent, 2)
        option_latent = tf.concat([policy_latent, op], 1)
        q_latent = tf.concat([policy_latent, op, ac], 1)

        policy = PolicyWithValue(
            env=env,
            observations=X,
            next_observations=X_next,
            actions=ac,
            option_z=op,
            dones=dones,
            feature_map=cnn_fm,
            next_feature_map=next_cnn_fm,
            latent=policy_latent,
            option_latent=tf.stop_gradient(option_latent),
            q_latent=tf.stop_gradient(q_latent),
            vf=vf,
            next_vf=next_vf,
            sess=sess,
            **extra_tensors
        )
        return policy

    return policy_fn
