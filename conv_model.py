import tensorflow as tf
lay = tf.keras.layers
mnist_db = tf.keras.datasets.cifar10
(sample_train, label_train), (sample_test, label_test) = mnist_db.load_data()


def conv_model(input_shape, output_shape):
    model = tf.keras.models.Sequential()

    model.add(lay.Conv2D(16, kernel_size=(5, 5), padding='same', activation='relu',
                         input_shape=input_shape))
    model.add(lay.MaxPool2D((2, 2), 2))
    model.add(lay.Conv2D(32, kernel_size=(5, 5), activation='relu'))
    model.add(lay.MaxPool2D((2, 2), 2))
    model.add(lay.Flatten())
    model.add(lay.Dense(128, activation='relu'))
    model.add(lay.Dense(128, activation='relu'))
    model.add(lay.Dense(output_shape, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model

# his = model.fit(
#     sample_train,
#     label_train,
#     epochs=2
# )
# with open('conv-model-history.log', 'w') as f:
#     to_write = ''
#     for i in range(len(his.history['loss'])):
#         loss = his.history['loss'][i]
#         acc = his.history['accuracy'][i]
#         to_write += f'Epoch: {i+1}; Loss: {loss}; Accuracy: {acc};\n'
#     f.write(to_write)


# model.save("save/conv-mnist-model")
# model.evaluate(sample_test, label_test)
