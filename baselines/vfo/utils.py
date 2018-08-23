def get_action_dim(env):
    if len(env.action_space.shape) == 0:
        return env.action_space.n
    else:
        return env.action_space.shape[0]
