import os
from conv_model import conv_model as gen_conv_model
from ff_model import ff_model as gen_ff_model
from time import time
import numpy as np
import tensorflow as tf

DATASETS = {
    1: ('mnist', tf.keras.datasets.mnist),
    2: ('fashion-mnist', tf.keras.datasets.fashion_mnist),
    3: ('cifar-10', tf.keras.datasets.cifar10),
    4: ('cifar-100', tf.keras.datasets.cifar100)
}

ACCEPT = {
    1: 'Feed forward nätverkets summering',
    2: 'Konvolutionära nätverkets summering',
}


def main():
    inp = -1
    while not inp in DATASETS.keys():
        os.system('cls')
        print('Välj dataset:')
        for key, (val, _) in DATASETS.items():
            print(f' {key} - {val}')
        print('')
        try:
            inp = int(input())
        except:
            inp = -1

    ds = DATASETS[inp][1]
    (sample_train, label_train), (sample_test, label_test) = ds.load_data()

    n_dim = sample_train.ndim
    if n_dim < 4:  # * Conv2D kräver 4 dimensioner
        sample_train = np.expand_dims(sample_train, axis=n_dim)
        sample_test = np.expand_dims(sample_test, axis=n_dim)
    print(sample_test.shape)
    n_out = len(set(np.ndarray.flatten(label_train)))

    os.system('cls')
    print('Laddar nätverken...\n')

    ff_model = gen_ff_model(sample_train.shape[1:], n_out)
    conv_model = gen_conv_model(sample_train.shape[1:], n_out)

    inp = -1
    while inp != len(ACCEPT.keys()):
        os.system('cls')
        for key, val in ACCEPT.items():
            print(f' {key} - {val}')
        i = len(ACCEPT.keys())
        print(f' {i} - Fortsätt')
        try:
            inp = int(input())
        except:
            inp = -1
        else:
            if inp == 1:
                os.system('cls')
                ff_model.summary()
                input('(Enter)')
            elif inp == 2:
                os.system('cls')
                conv_model.summary()
                input('(Enter)')

    inp = -1
    while inp <= 0:
        os.system('cls')
        try:
            inp = int(input('Antal epochs: '))
        except:
            inp = -1

    os.system('cls')
    print('Tränar feed forward nätverket\n')
    ff_time = time()
    ff_history = ff_model.fit(
        sample_train,
        label_train,
        epochs=inp
    )
    ff_time = time() - ff_time

    os.system('cls')
    print('Värderar feed forward nätverket\n')
    ff_eval = ff_model.evaluate(sample_test, label_test)

    os.system('cls')
    print('Tränar konvolutionära nätverket\n')
    conv_time = time()
    conv_history = conv_model.fit(
        sample_train,
        label_train,
        epochs=inp
    )
    conv_time = time() - conv_time

    os.system('cls')
    print('Värderar feed forward nätverket\n')
    conv_eval = conv_model.evaluate(sample_test, label_test)

    os.system('cls')
    print('Spara feed forward nätverket')
    dest = input('Destination: ')
    ff_model.save(dest)

    os.system('cls')
    print('Spara konvolutionära nätverket')
    dest = input('Destination: ')
    conv_model.save(dest)

    os.system('cls')
    print('Spara en log för feed forward nätverkets träning')
    dest = input('Destination: ')
    with open(dest, 'w') as f:
        to_write = f'Tid: {ff_time} sekunder\n'
        to_write += f'Eval; Loss: {ff_eval[0]}; Accuracy: {ff_eval[1]};\n'
        loss = ff_history.history['loss']
        acc = ff_history.history['accuracy']
        for i in range(len(loss)):
            to_write += f'Epoch: {i+1}; Loss: {loss[i]}; Accuracy: {acc[i]};\n'
        f.write(to_write)
        f.close()

    os.system('cls')
    print('Spara en log för konvolutionära nätverkets träning')
    dest = input('Destination: ')
    with open(dest, 'w') as f:
        to_write = f'Tid: {conv_time} sekunder\n'
        to_write += f'Eval; Loss: {conv_eval[0]}; Accuracy: {conv_eval[1]};\n'
        loss = conv_history.history['loss']
        acc = conv_history.history['accuracy']
        for i in range(len(loss)):
            to_write += f'Epoch: {i+1}; Loss: {loss[i]}; Accuracy: {acc[i]};\n'
        f.write(to_write)
        f.close()


if __name__ == '__main__':
    main()
