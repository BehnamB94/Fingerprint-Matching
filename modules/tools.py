import matplotlib.pyplot as plt
import numpy as np


def split(data, label, first_part):
    size = data.shape[0]
    selected = np.random.choice(size, int(first_part * size), replace=False)
    mask = np.zeros((size,), dtype=np.bool)
    mask[selected] = 1
    return data[mask], label[mask], data[~mask], label[~mask]


def plot(train_loss, train_acc, valid_acc, tag):
    fig, ax1 = plt.subplots()
    x_range = np.arange(len(train_acc)) + 1
    ax1.plot(x_range, train_loss, 'red')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('train loss', color='red')
    ax1.tick_params('y', colors='red')
    ax2 = ax1.twinx()
    ax2.plot(x_range, train_acc, 'blue')
    ax2.plot(x_range, valid_acc, 'b--')
    ax2.set_ylabel('validation and train accuracy', color='blue')
    ax2.tick_params('y', colors='blue')
    fig.tight_layout()
    plt.savefig('plot-{}.png'.format(tag))


def check_data(data, labels):
    for i in range(data.shape[0]):
        plt.imsave('check/image-{}-{}'.format(i, labels[i]),
                   np.concatenate([data[i, 0, :, :], data[i, 1, :, :]], axis=1),
                   cmap='gray')


def check_sample(sample1, sample2):
    for i, (s1, s2) in enumerate(zip(sample1, sample2)):
        plt.imsave('check/sample-{}'.format(i),
                   np.concatenate([s1[0], s2[0]], axis=1),
                   cmap='gray')
