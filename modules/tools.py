import matplotlib.pyplot as plt
import numpy as np


def make_xy(sample1, sample2):
    size = sample1.shape[0]
    x = np.concatenate([sample1, sample2], axis=1)
    fake_x = np.copy(x)
    fake_x[:, 0, :, :] = fake_x[::-1, 0, :, :]
    x = np.concatenate([x, fake_x], axis=0)
    y = np.array([1] * size + [0] * size)
    return x, y


def add_miss_block(samples, block_size=10, blocks_on_image=3):
    size = samples.shape[0]
    image_rows = samples.shape[2]
    image_columns = samples.shape[3]

    res = np.copy(samples)
    for itr in range(blocks_on_image):
        rand_r = np.random.randint(image_rows - block_size, size=size)
        rand_c = np.random.randint(image_columns - block_size, size=size)
        for i, (r, c) in enumerate(zip(rand_r, rand_c)):
            res[i, 0, r:r + block_size, c:c + block_size] = 0
    return res


def cut_image(samples, ratio):
    image_rows = samples.shape[2]
    new_rows = int(image_rows * ratio)
    useless_rows = image_rows - new_rows
    top_gap = useless_rows // 2

    upper_res = np.zeros_like(samples)
    upper_res[:, :, top_gap:top_gap + new_rows, :] = samples[:, :, :-useless_rows, :]

    lower_res = np.zeros_like(samples)
    lower_res[:, :, top_gap:top_gap + new_rows, :] = samples[:, :, useless_rows:, :]
    return upper_res, lower_res


def combine_pairs(parts):
    num_parts = len(parts)
    res = list()
    for i in range(num_parts):
        for j in range(i + 1, num_parts):
            res.append(
                np.concatenate([parts[i], parts[j]], axis=1)
            )
    return np.concatenate(res, axis=0)


def make_train_xy(sample1, sample2):
    assert sample1.shape == sample2.shape

    x_list = list()
    # mb1 = add_miss_block(sample1)
    # mb2 = add_miss_block(sample2)
    # x_list.append(combine_pairs([sample1, mb1, sample2, mb2]))
    x_list.append(combine_pairs([sample1, sample2]))

    p11, p12 = cut_image(sample1, .8)
    p21, p22 = cut_image(sample2, .8)
    # x_list.append(combine_pairs([p11, p12, p21, p22]))
    x_list.append(combine_pairs([sample1, p11]))
    x_list.append(combine_pairs([sample2, p22]))

    # p11, p12 = cut_image(sample1, .6)
    # p21, p22 = cut_image(sample2, .6)
    # x_list.append(combine_pairs([p11, p12, p21, p22]))

    x = np.concatenate(x_list, axis=0)
    fake_x = np.copy(x)
    fake_x[:, 0, :, :] = fake_x[::-1, 0, :, :]

    y = np.array([1] * x.shape[0] + [0] * x.shape[0])
    x = np.concatenate([x, fake_x], axis=0)

    # add reverse
    x = np.concatenate([x, x[:, ::-1, :, :]], axis=0)
    y = np.concatenate([y, y], axis=0)
    return x, y


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
    plt.savefig('results/plot-{}.png'.format(tag))


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
