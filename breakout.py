import gym
import numpy as np
import tensorflow as tf
from baselines.common.atari_wrappers import wrap_deepmind
from keras import models

MODEL_PATH = 'save\\adam-model'

seed = 98
env = gym.make('BreakoutNoFrameskip-v4', render_mode='human')
env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(seed)
act_model = models.load_model(MODEL_PATH)

while True:
    state = np.array(env.reset())
    while True:
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = act_model(state_tensor, training=False)
        action = tf.argmax(action_probs[0]).numpy()

        # Update env
        state_next, _, done, _ = env.step(action)
        state = np.array(state_next)

        # Exit if done
        if done:
            break
