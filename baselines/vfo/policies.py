import tensorflow as tf
from baselines.a2c.utils import fc
from baselines.common import tf_util
from baselines.common.distributions import make_pdtype
from baselines.common.policies import _normalize_clip_observation
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.tf_util import adjust_shape
from baselines.vfo.models import get_network_builder
from baselines.vfo.utils import get_action_dim


class PolicyWithValue(object):
    """
    Encapsulates fields and methods for RL policy, options policy and value
    function estimation with shared parameters
    """
    def __init__(self, env, observations, actions, option_z, feature_map,
                 latent, option_latent, q_latent, vf_latent=None, sess=None,
                 **tensors):
        """
        Parameters:
        -----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        actions         tensorflow placeholder in which the actions will be fed

        option_z        tensorflow placeholder in which the option one-hot will be fed

        feature_map     feature map (tensorflow tensor) from last conv layer

        latent          latent state from which policy distribution parameters should be inferred

        option_latent   latent state from which option policy distribution parameters should be inferred

        q_latent        latent state from which soft q function for option should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        data_format     data format of Conv layer

        **tensors       tensorflow tensors for additional attributes such as state or mask
        """
        self.X = observations
        self.ac = actions
        self.op_z = option_z
        self.state = tf.constant([])
        self.initial_state = None
        self.__dict__.update(tensors)

        self.fm = feature_map
        vf_latent = vf_latent if vf_latent is not None else latent

        self.vf_latent = tf.layers.flatten(vf_latent)
        latent = tf.layers.flatten(latent)

        self.pdtype = make_pdtype(env.action_space)
        self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=0.01)

        self.action = self.pd.sample()
        self.neglogp = self.pd.neglogp(self.action)
        self.sess = sess

        self.vf = fc(self.vf_latent, 'vf', 1)
        self.proto_vf = self.vf
        self.vf = self.vf[:, 0]

        with tf.variable_scope('option'):
            self.option_pdtype = make_pdtype(env.action_space)
            self.option_pd, self.option_pi = self.option_pdtype.pdfromlatent(
                option_latent, init_scale=0.01)
            self.option_action = self.option_pd.sample()
            self.option_neglogp = self.option_pd.neglogp(self.action)

            # soft q function for option
            self.option_q = fc(q_latent, 'option_q', get_action_dim(env))
            # vf-feature map activation grad
            va_grads = tf.gradients(self.vf, self.fm, name='vf_fm_grad')
            # assume data_format 'NHWC'
            self.pvfs = tf.reduce_sum(tf.multiply(self.fm, va_grads), axis=[1, 2])

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess or tf.get_default_session()
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def step(self, observation, **extra_feed):
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
        a, v, state, neglogp = self._evaluate(
            [self.action, self.vf, self.state, self.neglogp],
            observation, **extra_feed)
        if state.size == 0:
            state = None
        return a, v, state, neglogp

    def option_step(self, option_z, observation, **extra_feed):
        """
        Compute next action(s) given the observaion(s) and option one_hot
        """
        extra_feed.update({'op_z': option_z})
        a, state, neglogp = self._evaluate(
            [self.option_action, self.state, self.option_neglogp],
            observation, **extra_feed)
        if state.size == 0:
            state = None
        return a, state, neglogp

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


def build_policy(env, policy_network, value_network='gap',
                 normalize_observations=False, estimate_q=False,
                 **policy_kwargs):
    assert isinstance(policy_network, str), 'only accept string'
    assert policy_network.startswith('cnn'), 'VFO currently supports CNN based policy network'
    network_type = policy_network
    policy_network = get_network_builder(network_type)(**policy_kwargs)

    def policy_fn(noptions=64, nbatch=None, nsteps=None, sess=None,
                  observ_placeholder=None, action_placeholder=None,
                  option_z_placeholder=None):
        ob_space = env.observation_space
        ac_n = get_action_dim(env)

        X = observ_placeholder if observ_placeholder is not None else \
            observation_placeholder(ob_space, batch_size=nbatch, name='ob')
        ac = action_placeholder if action_placeholder is not None else \
            tf.placeholder(tf.float32, shape=(nbatch, ac_n), name='ac')
        op = option_z_placeholder if option_z_placeholder is not None else \
            tf.placeholder(tf.float32, shape=(nbatch, noptions), name='op_z')

        extra_tensors = {}

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
                cnn_fm, policy_latent, recurrent_tensors = policy_network(encoded_x, nenv)
                extra_tensors.update(recurrent_tensors)
            if recurrent_tensors is not None:
                # recurrent architecture, need a few more steps
                nenv = nbatch // nsteps
                assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
                cnn_fm, policy_latent, recurrent_tensors = policy_network(encoded_x, nenv)
                extra_tensors.update(recurrent_tensors)

        _v_net = value_network

        if _v_net is None or _v_net == 'shared':
            vf_latent = policy_latent
        elif _v_net == 'gap':
            vf_latent = global_average_pooling(cnn_fm, **policy_kwargs)
        else:
            raise NotImplementedError

        tf.assert_rank(policy_latent, 2)
        option_latent = tf.concat([policy_latent, op], 1)
        q_latent = tf.concat([policy_latent, op, ac], 1)

        policy = PolicyWithValue(
            env=env,
            observations=X,
            actions=ac,
            option_z=op,
            feature_map=cnn_fm,
            latent=policy_latent,
            option_latent=option_latent,
            q_latent=q_latent,
            vf_latent=vf_latent,
            sess=sess,
            **extra_tensors
        )
        return policy

    return policy_fn
