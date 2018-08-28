# Modified from baselines.acer.buffer
import numpy as np


class Buffer(object):
    """
    Buffer for options off-policy training
    """
    def __init__(self, env, nsteps, size=1000):
        self.nenv = env.num_envs
        self.nsteps = nsteps
        self.nh, self.nw, self.nc = env.observation_space.shape
        self.nbatch = self.nenv * self.nsteps
        self.size = size // (self.nsteps)  # Each loc contains nenv * nsteps frames, thus total buffer is nenv * size frames

        # Memory
        self.obs = None
        self.next_obs = None
        self.states = None
        self.next_states = None
        self.masks = None
        self.next_masks = None
        self.actions = None
        self.actions_full = None
        self.dones = None
        self.options_z = None

        # Size indexes
        self.next_idx = 0
        self.num_in_buffer = 0

    def has_atleast(self, frames):
        # Frames per env, so total (nenv * frames) Frames needed
        # Each buffer loc has nenv * nsteps frames
        return self.num_in_buffer >= (frames // self.nsteps)

    def can_sample(self):
        return self.num_in_buffer > 0

    def _create_empty_memory(self, data):
        return np.empty([self.size] + list(data.shape), dtype=data.dtype)

    def put(self, obs, next_obs, states, next_states, masks, next_masks,
            actions, actions_full, dones, options_z):
        if self.obs is None:
            self.obs = self._create_empty_memory(obs)
            self.next_obs = self._create_empty_memory(next_obs)
            self.masks = self._create_empty_memory(masks)
            self.next_masks = self._create_empty_memory(next_masks)
            self.actions = self._create_empty_memory(actions)
            self.actions_full = self._create_empty_memory(actions_full)
            self.dones = self._create_empty_memory(dones)
            self.options_z = self._create_empty_memory(options_z)

            if states is not None:
                self.states = self._create_empty_memory(states)
                self.next_states = self._create_empty_memory(next_states)

        self.obs[self.next_idx] = obs
        self.next_obs[self.next_idx] = next_obs
        self.masks[self.next_idx] = masks
        self.next_masks[self.next_idx] = next_masks
        self.actions[self.next_idx] = actions
        self.actions_full[self.next_idx] = actions_full
        self.dones[self.next_idx] = dones
        self.options_z[self.next_idx] = options_z

        if states is not None:
            self.states[self.next_idx] = states
            self.next_states[self.next_idx] = next_states

        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

    def take(self, x, idx, envx):
        """Take env steps data from different memory position"""
        nenv = self.nenv
        # for states, nsteps=1, for others, nsteps=self.nsteps
        nsteps = x.shape[1] // nenv
        out = np.empty([nenv, nsteps] + list(x.shape[2:]), dtype=x.dtype)
        for i in range(nenv):
            offset = envx[i] * nsteps
            out[i] = x[idx[i], offset:offset+nsteps]
        return out.reshape([-1] + list(x.shape[2:]))

    def get(self):
        nenv = self.nenv
        assert self.can_sample()

        # Sample exactly one id per env.
        # If you sample across envs, then higher correlation in samples from same env.
        idx = np.random.randint(0, self.num_in_buffer, nenv)
        envx = np.arange(nenv)

        take = lambda x: self.take(x, idx, envx)
        obs = take(self.obs)
        next_obs = take(self.next_obs)
        masks = take(self.masks)
        next_masks = take(self.next_masks)
        actions = take(self.actions)
        actions_full = take(self.actions_full)
        dones = take(self.dones)
        options_z = take(self.options_z)

        if self.states is not None:
            states = take(self.states)
            next_states = take(self.next_states)
        else:
            states = None
            next_states = None

        return obs, next_obs, states, next_states, masks, next_masks, \
            actions, actions_full, dones, options_z
