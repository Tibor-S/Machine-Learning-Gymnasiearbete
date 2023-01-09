import tensorflow as tf
lay = tf.keras.layers


mnist_db = tf.keras.datasets.cifar10
(sample_train, label_train), (sample_test, label_test) = mnist_db.load_data()


def ff_model(input_shape, output_shape):
    model = tf.keras.models.Sequential()
    model.add(lay.Flatten(input_shape=input_shape))
    model.add(lay.Dense(128, activation='relu'))
    model.add(lay.Dense(128, activation='relu'))
    model.add(lay.Dense(128, activation='relu'))
    model.add(lay.Dense(128, activation='relu'))
    model.add(lay.Dense(128, activation='relu'))
    model.add(lay.Dense(128, activation='relu'))
    model.add(lay.Dense(output_shape, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model
