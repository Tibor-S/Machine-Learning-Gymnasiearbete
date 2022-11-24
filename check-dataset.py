import numpy as np

SAVE_PATH = 'train-dataset.npz'

with np.load(SAVE_PATH) as data:
    print('SAMPLES SHAPE:', data['samples'].shape)
    print('LABELS SHAPE:', data['labels'].shape)
