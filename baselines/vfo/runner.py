# Modified from baselines.a2c.runner
import numpy as np
from gym.spaces import Discrete
from baselines.common.runners import AbstractEnvRunner
from baselines.vfo.utils import get_action_dim


class OptionsRunner(AbstractEnvRunner):
    def __init__(self, env, model, noptions, nsteps=5, gamma=0.99):
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

    def sample_option_z(self, prior):
        """Sample new option-z"""
        option_id = np.random.choice(self.noptions, p=prior)
        option_z = np.zeros([self.nenv, self.noptions], dtype=np.float32)
        option_z[:, option_id] = 1.0
        self.option_z = option_z

    def run(self):
        mb_obs, mb_next_obs, mb_actions, mb_dones, mb_options_z = [], [], [], [], []
        mb_states = self.states
        mb_next_states = self.states
        for n in range(self.nsteps):
            actions, states, _ = self.model.option_step(
                self.option_z, self.obs, S=self.states, M=self.dones)

            if n == 0:
                mb_next_states = states

            mb_options_z.append(self.option_z)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            self.obs = obs
            mb_next_obs.append(np.copy(obs))
        mb_dones.append(self.dones)

        mb_obs = np.asarray(
            mb_obs, dtype=self.ob_dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_next_obs = np.asarray(
            mb_next_obs, dtype=self.ob_dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_actions = np.asarray(mb_actions, dtype=self.model.train_model.action.dtype.name).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_next_masks = mb_dones[:, 1:]
        mb_dones = mb_dones[:, 1:]

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
            mb_dones, mb_options_z
