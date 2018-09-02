# Modified from baselines.a2c.runner
import numpy as np
from gym.spaces import Discrete
from baselines.a2c.utils import discount_with_dones
from baselines.common.runners import AbstractEnvRunner
from baselines.vfo.utils import get_action_dim


STEPS_TO_TRUST_DISCRIMINATOR = 1000


class OptionsRunner(AbstractEnvRunner):
    def __init__(self, env, model, noptions, nsteps=5, gamma=0.99,
                 use_selective_option=False, top_n_options=8):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.gamma = gamma
        self.batch_action_shape = [x if x is not None else -1 for x in
                                   model.train_model.action.shape.as_list()]
        self.ob_dtype = model.train_model.X.dtype.as_numpy_dtype
        self.noptions = noptions
        self.nenv = env.num_envs
        self.need_one_hot_ac = isinstance(env.action_space, Discrete)
        self.action_n = get_action_dim(env)
        self.sample_option_z(prior=None)  # init option_z with uniform distri
        self.use_selective_option = use_selective_option
        if use_selective_option:
            assert top_n_options < noptions, 'top_n_options is larger than noptions'
            self.top_n_options = top_n_options
            self.calls_run = 0

    def sample_option_z(self, prior):
        """Sample new option-z"""
        option_id = np.random.choice(self.noptions, p=prior)
        option_z = np.zeros([self.nenv, self.noptions], dtype=np.float32)
        option_z[:, option_id] = 1.0
        self.option_z = option_z

    def run(self):
        mb_obs, mb_next_obs, mb_rewards, mb_actions, mb_values, mb_dones, \
            mb_options_z = [], [], [], [], [], [], []
        mb_states = self.states
        mb_next_states = self.states

        if self.use_selective_option:
            self.calls_run += 1

        for n in range(self.nsteps):
            if self.use_selective_option and \
               self.calls_run > STEPS_TO_TRUST_DISCRIMINATOR:
                # after steps discriminator is trustable, then use
                # selective_option_step, this setting avoid influence
                # from nn_discriminator initialization
                actions, values, states, _, self.option_z = \
                    self.model.selective_option_step(
                        self.obs, top_n=self.top_n_options, S=self.states,
                        M=self.dones)
            else:
                actions, values, states, _ = self.model.option_step(
                    self.option_z, self.obs, S=self.states, M=self.dones)

            if n == 0:
                mb_next_states = states

            mb_options_z.append(self.option_z)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            self.obs = obs
            mb_next_obs.append(np.copy(obs))
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(
            mb_obs, dtype=self.ob_dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_next_obs = np.asarray(
            mb_next_obs, dtype=self.ob_dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=self.model.train_model.action.dtype.name).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_next_masks = mb_dones[:, 1:]

        if self.gamma > 0.0:
            # discount/bootstrap off value fn
            last_values = self.model.value(self.obs, S=self.states, M=self.dones).tolist()
            for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                rewards = rewards.tolist()
                dones = dones.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)

                mb_rewards[n] = rewards

        mb_dones = mb_dones[:, 1:]

        # reshape for output
        mb_rewards = mb_rewards.flatten()
        mb_values = mb_values.flatten()
        mb_actions = mb_actions.reshape(self.batch_action_shape)
        mb_options_z = np.stack(mb_options_z).reshape([-1, self.noptions])
        mb_masks = mb_masks.flatten()
        mb_next_masks = mb_next_masks.flatten()
        mb_dones = mb_dones.flatten()
        if self.need_one_hot_ac:
            mb_actions_full = np.zeros([mb_actions.shape[0], self.action_n],
                                       dtype=np.float32)
            mb_actions_full[np.arange(mb_actions.shape[0]), mb_actions] = 1.0
        else:
            mb_actions_full = mb_actions

        return mb_obs, mb_next_obs, mb_states, mb_next_states, \
            mb_masks, mb_next_masks, mb_actions, mb_actions_full, \
            mb_rewards, mb_values, mb_dones, mb_options_z
