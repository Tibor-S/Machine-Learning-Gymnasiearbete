import tensorflow as tf
import numpy as np
from keras import layers, Model
with tf.device('cpu:0'):
    num_actions = 4
    n = 33000
    DATASET_PATH = 'train-dataset.npz'
    SAVE_PATH = 'save/adam-model'
    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = n

    def get_dataset():
        with np.load(DATASET_PATH) as data:
            samples = data['samples']
            labels = data['labels']

        indices = np.arange(samples.shape[0])
        np.random.shuffle(indices)

        samples = samples[indices]
        labels = labels[indices]
        return tf.data.Dataset.from_tensor_slices((samples[:n], labels[:n]))

    def create_q_model():
        # Network defined by the Deepmind paper
        inputs = layers.Input(
            shape=(
                2,
                84,
                84,
                4,
            )
        )

        # Convolutions on the frames on the screen
        layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
        layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

        layer4 = layers.Flatten()(layer3)

        layer5 = layers.Dense(512, activation="relu")(layer4)
        action = layers.Dense(num_actions, activation="linear")(layer5)

        model = Model(inputs=inputs, outputs=action)
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.Huber())
        return model

    print('Loading model')
    act_model = create_q_model()
    print('Loading dataset')
    dataset = get_dataset().shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    print(f'Fit model (total epochs: {n // BATCH_SIZE})')
    try:
        act_model.fit(dataset, epochs=(n // BATCH_SIZE))
        print('Save model')
        act_model.save(SAVE_PATH)
    except:
        print('Save model')
        act_model.save(SAVE_PATH)
