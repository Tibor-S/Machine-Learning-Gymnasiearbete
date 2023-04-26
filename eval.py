import matplotlib.pyplot as plt
import os
import re
import numpy as np
from train import DATASETS


MODELS = [('Feed forward', 'ff'), ('Konvolutionellt', 'conv')]
ACTIONS = {
    1: ('Visa graf', lambda inp: show(DATASETS[inp])),
    2: ('Spara graf', lambda inp: save(DATASETS[inp])),
    3: ('Visa exempel', lambda inp: layer_outputs(DATASETS[inp])),
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
    ds_i = inp

    inp = -1
    while not inp in ACTIONS.keys():
        os.system('cls')
        for key, (val, _) in ACTIONS.items():
            print(f' {key} - {val}')
        print('')
        try:
            inp = int(input())
        except:
            inp = -1
    act = inp
    ACTIONS[act][1](ds_i)


def models(dataset):
    d = dataset[0]
    for name, short in MODELS:
        path = f'log/{d}-{short}.log'


def parse_models(dataset):
    d = dataset[0]
    model_log = []
    for name, short in MODELS:
        path = f'log/{d}-{short}.log'
        with open(path, 'r') as f:
            lines = f.readlines()

        time = -1
        eval_loss = -1
        eval_acc = -1
        epochs = []
        loss_log = []
        acc_log = []

        mode = 'TIME'
        for line in lines:
            if mode == 'TIME':
                cap = re.match(r'Tid: ([\d.]+)', line)
                try:
                    time = float(cap.group(1))
                except:
                    print("Trubbel!")
                finally:
                    mode = 'EVAL'
            elif mode == 'EVAL':
                cap = re.match(r'.+;.+: ([\d.]+);.+: ([\d.]+)', line)
                cap_loss = cap.group(1)
                cap_acc = cap.group(2)
                try:
                    eval_loss = float(cap_loss)
                    eval_acc = float(cap_acc)
                except:
                    print('Trubbel!')
                finally:
                    mode = 'EPOCH'
            elif mode == 'EPOCH':
                cap = re.match(r'.+: ([\d]+);.+: ([\d.]+);.+: ([\d.]+)', line)
                cap_ep = cap.group(1)
                cap_loss = cap.group(2)
                cap_acc = cap.group(3)
                try:
                    epochs.append(int(cap_ep))
                    loss_log.append(float(cap_loss))
                    acc_log.append(float(cap_acc))
                except:
                    print('Trubbel!')
            else:
                print('Omöjligt!')

        model_log.append({
            'name': name,
            'short_name': short,
            'time': time,
            'eval_loss': eval_loss,
            'eval_acc': eval_acc,
            'epochs': epochs,
            'loss_log': loss_log,
            'acc_log': acc_log,
        })
    return model_log


def layer_outputs(dataset):
    model = parse_models(dataset)
    ds = dataset[1]
    (_, _), (sample_test, label_test) = ds.load_data()

    fig, ax = plt.subplots(1, 1)


def graph(dataset):
    model_logs = parse_models(dataset)
    fig, ax = plt.subplots(1, 3)

    for model in model_logs:
        epochs = model['epochs']
        acc_log = model['acc_log']
        name = model['name']
        ax[0].plot(epochs, 100*np.array(acc_log), label=name)
    ax[0].set_ylim(0, 100)
    ax[0].legend()
    ax[0].set_title('Träffsäkerhet under träning')
    ax[0].set_ylabel('Procent')
    ax[0].set_yticks(range(0, 100, 10))
    ax[0].yaxis.grid(True, which='major', color='grey', alpha=.25)
    ax[0].axhline(50, color='grey', alpha=0.25)
    ax[0].set_xlabel('Epoch')

    vals = []
    names = []
    for model in model_logs:
        vals.append(100 * model['eval_acc'])
        names.append(model['name'])
    ax[1].bar(names, vals, width=0.5, align='center')
    ax[1].set_ylim(0, 100)
    ax[1].set_title('Värderad träffsäkerhet')
    ax[1].set_ylabel('Procent')
    ax[1].set_yticks(range(0, 100, 10))
    ax[1].yaxis.grid(True, which='major', color='grey', alpha=.25)
    ax[1].axhline(50, color='grey', alpha=0.25)

    vals = []
    names = []
    for model in model_logs:
        vals.append(model['time'] / 60)
        names.append(model['name'])
    ax[2].bar(names, vals, width=0.5, align='center')
    ax[2].set_title('Tid att träna')
    ax[2].set_ylabel('Tid (min)')
    ax[2].set_yticks(range(0, int(max(vals)), 5))
    ax[2].yaxis.grid(True, which='major', color='grey', alpha=.25)

    fig.suptitle(dataset[0].upper())
    fig.set_figwidth(12)


def show(dataset):
    graph(dataset)
    plt.show()


def save(dataset):
    graph(dataset)
    path = f'graph/{dataset[0]}.png'
    plt.savefig(path)


if __name__ == '__main__':
    main()
