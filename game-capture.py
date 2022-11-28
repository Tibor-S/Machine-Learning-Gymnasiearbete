import sys
import time
import gym
from baselines.common.atari_wrappers import wrap_deepmind
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, models
import keyboard


def getInp():
    if keyboard.is_pressed('left'):
        return 3
    if keyboard.is_pressed('right'):
        return 2

    return 0


SAVE_PATH = 'train-dataset.npz'
FRAMES_PER_SAMPLE = 1
NUM_ACTIONS = 4
MAX_STEPS_PER_EP = 10000
try:
    print('LOADING \"samples\" AND \"labels\"')
    with np.load(SAVE_PATH) as data:
        samples = list(data['samples'])
        labels = list(data['labels'])
except BaseException as e:
    print('WARNING: ERROR OCCURED')
    print('WARNING: \"samples\" AND \"labels\" will be initialized as empty arrays')
    samples = []
    labels = []

print(type(samples))
print(type(labels))

# Use the Baseline Atari environment because of Deepmind helper functions
env = gym.make('BreakoutNoFrameskip-v4',
               render_mode='human')
# Warp the frames, grey scale, stake four frame and scale to smaller ratio
env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(42)
spec = gym.envs.registration.spec('BreakoutNoFrameskip-v4')

frameCount = 0
frameBuffer = []
actionBuffer = []
print('\n')
print('SAVE AND EXIT VIA CTRL + C')
try:
    while True:
        state = np.array(env.reset())
        for timestep in range(1, MAX_STEPS_PER_EP):
            frameCount += 1
            time.sleep(0.05)

            # User
            action = getInp()

            # Log state and action
            frameBuffer.append(state)
            actionProb = [0 for _ in range(4)]
            actionProb[action] = 1

            # Append state and action
            if len(frameBuffer) % FRAMES_PER_SAMPLE == 0:
                samples.append(frameBuffer)
                labels.append(actionProb)
                frameBuffer.pop(0)

            # Update env
            state_next, _, done, _ = env.step(action)
            state = np.array(state_next)

            # Exit if done
            if done:
                break
except:
    while True:
        conf = input('SAVE? (y/n): ')
        if conf.lower() == 'y':
            print('Saving...')
            np.savez_compressed(SAVE_PATH, samples=samples, labels=labels)
            print('Closing...')
            break
        elif conf.lower() == 'n':
            print('Closing...')
            break
