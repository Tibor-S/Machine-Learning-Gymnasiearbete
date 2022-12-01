import gym
import numpy as np
import matplotlib.pyplot as plt
from baselines.common.atari_wrappers import wrap_deepmind

env = gym.make('BreakoutNoFrameskip-v4')
env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(42)

arr = np.array(env.reset())


print(np.max(arr))
plt.matshow(arr)
plt.show()
