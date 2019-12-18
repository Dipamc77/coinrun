import numpy as np
from coinrun import setup_utils, make

setup_utils.setup_and_load()
env = make(env_id='standard', num_envs=1)
for _ in range(100):
    acts = np.array([env.action_space.sample() for _ in range(env.num_envs)])
    _obs, _rews, _dones, _infos = env.step(acts)
env.close()
print(_infos)
