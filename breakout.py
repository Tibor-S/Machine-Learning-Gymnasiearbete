import gym
import numpy as np
import tensorflow as tf
from baselines.common.atari_wrappers import wrap_deepmind
from keras import models

MODEL_PATH = 'save/adam-model'

seed = 42

env = gym.make('BreakoutNoFrameskip-v4', render_mode='human')
env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(seed)
act_model = models.load_model(MODEL_PATH)

frameBuffer = []

while True:
    state = np.array(env.reset())
    while True:
        frameBuffer.append(state)
        print(len(frameBuffer))
        if len(frameBuffer) == 2:
            state_tensor = tf.convert_to_tensor(frameBuffer)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = act_model(state_tensor, training=False)
            print(action_probs)
            action = tf.argmax(action_probs[0]).numpy()
            frameBuffer.pop(0)
        else:
            action = 0
        # Update env
        state_next, _, done, _ = env.step(action)
        state = np.array(state_next)

        # Exit if done
        if done:
            break
