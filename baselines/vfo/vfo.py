import os
import time
import functools
import numpy as np
import tensorflow as tf
from tensorflow import losses

from baselines import logger
from baselines.common import set_global_seeds, explained_variance
from baselines.common import tf_util
from baselines.a2c.utils import Scheduler, find_trainable_variables
from baselines.a2c.runner import Runner

from baselines.vfo.policies import build_policy
from baselines.vfo.buffer import Buffer
from baselines.vfo.runner import OptionsRunner


class Model(object):
    def __init__(self, policy, env, nsteps, ent_coef=0.01, vf_coef=0.5,
                 max_grad_norm=0.5, lr=7e-4, alpha=0.99, epsilon=1e-5,
                 diverse_r_coef=0.1, gamma=0.99, total_timesteps=int(80e6),
                 lrschedule='linear'):
        sess = tf_util.get_session()
        nenvs = env.num_envs
        nbatch = nenvs*nsteps

        with tf.variable_scope('vfo_model', reuse=tf.AUTO_REUSE):
            step_model = policy(nbatch=nenvs, nsteps=1, sess=sess)
            train_model = policy(nbatch=nbatch, nsteps=nsteps, sess=sess)

        A = tf.placeholder(train_model.action.dtype, train_model.action.shape)
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])
        params = find_trainable_variables('vfo_model')
        print(params)

        # ==============================
        # model-free actor-critic loss
        # ==============================
        with tf.variable_scope('mf_loss'):
            neglogpac = train_model.pd.neglogp(A)
            entropy = tf.reduce_mean(train_model.pd.entropy())

            pg_loss = tf.reduce_mean(ADV * neglogpac)
            vf_loss = losses.mean_squared_error(tf.squeeze(train_model.vf), R)

            loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

            grads = tf.gradients(loss, params)
            if max_grad_norm is not None:
                grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            grads = list(zip(grads, params))

        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha,
                                            epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        # ==============================
        # diverse options policy loss
        # ==============================
        option_train_ops = []
        option_losses = []
        option_losses_names = []
        option_distil_train_op = None
        with tf.variable_scope('options_loss'):
            diversity_reward = -1 * tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=train_model.op_z, logits=train_model.option_discriminator)
            diversity_reward = tf.check_numerics(
                diversity_reward, 'Check numerics (1): diversity_reward')
            diversity_reward -= tf.log(
                tf.reduce_sum(train_model.prior_op_z * train_model.op_z) + 1e-6)
            print('d_reward:', diversity_reward.get_shape().as_list())

            intrinsic_reward = tf.multiply(
                train_model.next_pvfs - train_model.pvfs, train_model.op_z)
            intrinsic_reward = tf.reduce_sum(intrinsic_reward, 1)
            print('i_reward:', intrinsic_reward.get_shape().as_list())
            reward = diverse_r_coef * diversity_reward + intrinsic_reward

            with tf.variable_scope('critic'):
                next_vf = tf.reduce_sum(
                    tf.multiply(train_model.next_pvfs, train_model.op_z), 1)
                print('next_vf:', next_vf.get_shape().as_list())
                option_q_y = tf.stop_gradient(
                    reward + (1 - train_model.dones) * gamma * next_vf)
                option_q = tf.squeeze(train_model.option_q, 1)
                print('option_q_y:', option_q_y.get_shape().as_list())
                print('option_q:', option_q.get_shape().as_list())

                option_q_loss = 0.5 * tf.reduce_mean(
                    (option_q_y - option_q) ** 2)

            with tf.variable_scope('actor'):
                log_op_pi_t = train_model.option_pd.logp(A)
                log_target_t = tf.squeeze(train_model.option_q, 1)
                pvf = tf.reduce_sum(
                    tf.multiply(train_model.pvfs, train_model.op_z), 1)
                print('op_pi:', log_op_pi_t.get_shape().as_list())
                print('op_t:', log_target_t.get_shape().as_list())
                print('pvf:', pvf.get_shape().as_list())
                kl_surrogate_loss = tf.reduce_mean(
                    log_op_pi_t * tf.stop_gradient(log_op_pi_t - log_target_t - pvf))

            with tf.variable_scope('discriminator'):
                print('op_z:', train_model.op_z.get_shape().as_list())
                print('op_dis:', train_model.option_discriminator.get_shape().as_list())
                discriminator_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=train_model.op_z,
                        logits=train_model.option_discriminator_logits))

            with tf.variable_scope('distillation'):
                # NOTE: to train distillation, op_z should be feed with q(z|s)
                print('mf_pi:', train_model.pi.get_shape().as_list())
                print('op_pi:', train_model.option_pi.get_shape().as_list())
                distillation_loss = losses.mean_squared_error(
                    tf.stop_gradient(train_model.pi), train_model.option_pi)

        _train_option_q = tf.train.AdamOptimizer(lr).minimize(
            loss=option_q_loss, var_list=params)
        option_train_ops.append(_train_option_q)
        option_losses.append(option_q_loss)
        option_losses_names.append('option_critic')

        _train_option_policy = tf.train.AdamOptimizer(lr).minimize(
            loss=kl_surrogate_loss, var_list=params)
        option_train_ops.append(_train_option_policy)
        option_losses.append(kl_surrogate_loss)
        option_losses_names.append('option_actor')

        _train_option_disc = tf.train.AdamOptimizer(lr).minimize(
            loss=discriminator_loss, var_list=params)
        option_train_ops.append(_train_option_disc)
        option_losses.append(discriminator_loss)
        option_losses_names.append('option_discriminator')

        option_distil_train_op = tf.train.AdamOptimizer(lr).minimize(
            loss=distillation_loss, var_list=params)

        tf.summary.FileWriter(logger.get_dir(), sess.graph)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()

            td_map = {train_model.X: obs, A: actions, ADV: advs, R: rewards, LR: cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        def train_options(obs, next_obs, states, next_states, masks,
                          next_masks, actions, actions_full, dones, options_z):
            feed = {train_model.X: obs, train_model.X_next: next_obs,
                    A: actions, train_model.ac: actions_full,
                    train_model.dones: dones, train_model.op_z: options_z}
            if states is not None:
                feed[train_model.S] = states
                feed[train_model.next_S] = next_states
                feed[train_model.M] = masks
                feed[train_model.next_M] = next_masks

            record_loss_values = []
            for name, loss, train_op in zip(option_losses_names, option_losses,
                                            option_train_ops):
                loss_value, _ = sess.run([loss, train_op], feed)
                record_loss_values.append((name + '_loss', loss_value))

            return record_loss_values

        def distill_mf_to_options(obs, states, masks):
            feed = {train_model.X: obs}
            if states is not None:
                feed[train_model.S] = states
                feed[train_model.M] = masks

            option_ensembles = sess.run(train_model.option_discriminator, feed)
            feed[train_model.op_z] = option_ensembles
            distillation_loss_value, _ = sess.run(
                [distillation_loss, option_distil_train_op], feed)

            return distillation_loss_value

        self.train = train
        self.train_options = train_options
        self.distill_mf_to_options = distill_mf_to_options
        self.train_model = train_model
        self.prior_op_z = train_model.prior_op_z
        self.step_model = step_model
        self.step = step_model.step
        self.option_step = step_model.option_step
        self.option_select = step_model.option_select
        self.selective_option_step = step_model.selective_option_step
        self.value = step_model.value
        self.proto_value = step_model.proto_value
        self.initial_state = step_model.initial_state
        self.save = functools.partial(tf_util.save_variables, sess=sess)
        self.load = functools.partial(tf_util.load_variables, sess=sess)
        tf.global_variables_initializer().run(session=sess)


def learn(
        network,
        env,
        seed=None,
        nsteps=5,
        noptions=64,
        replay_buffer_size=1000,
        total_timesteps=int(80e6),
        start_op_at=0.8,
        options_update_iter=10,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        lr=7e-4,
        lrschedule='linear',
        epsilon=1e-5,
        diverse_r_coef=0.1,
        alpha=0.99,
        gamma=0.99,
        log_interval=100,
        load_path=None,
        **network_kwargs):
    '''
    Main entrypoint for VFO algorithm. Train a policy with given network architecture on a given environment using vfo algorithm.

    Parameters:
    -----------

    network:            policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                        specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                        tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                        neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                        See baselines.common/policies.py/lstm for more details on using recurrent nets in policies


    env:                RL environment. Should implement interface similar to VecEnv (baselines.common/vec_env) or be wrapped with DummyVecEnv (baselines.common/vec_env/dummy_vec_env.py)


    seed:               seed to make random number sequence in the alorightm reproducible. By default is None which means seed from system noise generator (not reproducible)

    nsteps:             int, number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                        nenv is number of environment copies simulated in parallel)

    noptions:           int, number of options for VFO, i.e. channels of last Conv layer

    replay_buffer_size  int, size of replay buffer which is used to train options

    total_timesteps:    int, total number of timesteps to train on (default: 80M)

    start_op_at:        float, after trainign mf policy for `start_op_at * total_timesteps` steps, begin to train options policy

    options_update_iter: int, number of call for train_options per sample

    vf_coef:            float, coefficient in front of value function loss in the total loss function (default: 0.5)

    ent_coef:           float, coeffictiant in front of the policy entropy in the total loss function (default: 0.01)

    max_gradient_norm:  float, gradient is clipped to have global L2 norm no more than this value (default: 0.5)

    lr:                 float, learning rate for RMSProp (current implementation has RMSProp hardcoded in) (default: 7e-4)

    lrschedule:         schedule of learning rate. Can be 'linear', 'constant', or a function [0..1] -> [0..1] that takes fraction of the training progress as input and
                        returns fraction of the learning rate (specified as lr) as output

    epsilon:            float, RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update) (default: 1e-5)

    diverse_r_coef:     float, scaling factor for diversity reward when training option policy

    alpha:              float, RMSProp decay parameter (default: 0.99)

    gamma:              float, reward discounting parameter (default: 0.99)

    log_interval:       int, specifies how frequently the logs are printed out (default: 100)

    **network_kwargs:   keyword arguments to the policy / network builder. See baselines.vfo/policies.py/build_policy and arguments to a particular type of network
    '''
    # set_global_seeds(seed)

    nenvs = env.num_envs
    policy = build_policy(env, network, noptions, **network_kwargs)
    assert replay_buffer_size > 100, 'Replay buffer is too small'
    replay_buffer = Buffer(env, nsteps, size=replay_buffer_size)

    model = Model(
        policy=policy, env=env, nsteps=nsteps, ent_coef=ent_coef,
        vf_coef=vf_coef, max_grad_norm=max_grad_norm, lr=lr, alpha=alpha,
        epsilon=epsilon, diverse_r_coef=diverse_r_coef, gamma=gamma,
        total_timesteps=total_timesteps, lrschedule=lrschedule)
    if load_path is not None:
        model.load(load_path)
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)
    options_runner = OptionsRunner(env, model, noptions, nsteps=nsteps,
                                   gamma=gamma)

    nbatch = nenvs * nsteps
    tstart = time.time()
    to_train_options, init_replay_buffer_done = False, False
    total_updates = total_timesteps // nbatch+1
    for update in range(1, total_updates):
        if update % 300 == 0:
            model.save(os.path.join(logger.get_dir(), "snapshot"))
        if not to_train_options:
            obs, states, rewards, masks, actions, values = runner.run()
            policy_loss, value_loss, policy_entropy = model.train(
                obs, states, rewards, masks, actions, values)
            nseconds = time.time()-tstart
            fps = int((update*nbatch)/nseconds)
            if update % log_interval == 0 or update == 1:
                ev = explained_variance(values, rewards)
                logger.record_tabular("nupdates", update)
                logger.record_tabular("total_timesteps", update*nbatch)
                logger.record_tabular("fps", fps)
                logger.record_tabular("policy_entropy", float(policy_entropy))
                logger.record_tabular("value_loss", float(value_loss))
                logger.record_tabular("policy_loss", float(policy_loss))
                logger.record_tabular("explained_variance", float(ev))
                logger.dump_tabular()

            if update > total_updates * start_op_at:
                to_train_options = True
        else:
            obs, next_obs, states, next_states, masks, next_masks, actions, \
                actions_full, dones, options_z = options_runner.run()
            replay_buffer.put(
                obs, next_obs, states, next_states, masks, next_masks,
                actions, actions_full, dones, options_z)

            options_runner.sample_option_z(prior=model.prior_op_z)

            if replay_buffer.num_in_buffer > 100:
                init_replay_buffer_done = True

            if not init_replay_buffer_done:
                logger.info('Sample data using option policy...')
                continue

            for _ in range(options_update_iter):
                obs, next_obs, states, next_states, masks, next_masks, \
                    actions, actions_full, dones, options_z = \
                    replay_buffer.get()
                # distillation_loss_value = model.distill_mf_to_options(
                #     obs, states, masks)
                record_loss_values = model.train_options(
                    obs, next_obs, states, next_states, masks, next_masks,
                    actions, actions_full, dones, options_z)
                # record_loss_values.append(
                #     ('distillation_loss', distillation_loss_value))

            nseconds = time.time()-tstart
            fps = int((update*nbatch)/nseconds)
            if update % log_interval == 0 or update == 1:
                logger.record_tabular("nupdates", update)
                logger.record_tabular("total_timesteps", update*nbatch)
                logger.record_tabular("fps", fps)
                for loss_name, loss_value in record_loss_values:
                    logger.record_tabular(loss_name, loss_value)
                logger.dump_tabular()

    env.close()
    return model
