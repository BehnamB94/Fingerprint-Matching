import matplotlib.pyplot as plt
import numpy as np


def make_xy(sample_list):
    x_list = list()
    x_list.append(combine_pairs(sample_list))
    x = np.concatenate(x_list, axis=0)
    size = x.shape[0]
    fake_x = np.copy(x)
    fake_x[:, 0, :, :] = fake_x[::-1, 0, :, :]
    x = np.concatenate([x, fake_x], axis=0)
    y = np.array([1] * size + [0] * size)
    return x, y


def add_miss_block(samples, block_size=20, blocks_on_image=3):
    size = samples.shape[0]
    image_rows = samples.shape[2]
    image_columns = samples.shape[3]

    res = np.copy(samples)
    for itr in range(blocks_on_image):
        rand_r = np.random.randint(image_rows - block_size, size=size)
        rand_c = np.random.randint(image_columns - block_size, size=size)
        for i, (r, c) in enumerate(zip(rand_r, rand_c)):
            res[i, 0, r:r + block_size, c:c + block_size] = res[i, 0].max()
    return res


def cut_image(samples, ratio):
    image_rows = samples.shape[2]
    new_rows = int(image_rows * ratio)
    useless_rows = image_rows - new_rows
    top_gap = useless_rows // 2

    upper_res = np.zeros_like(samples)
    lower_res = np.zeros_like(samples)
    for i, max_val in enumerate(samples.reshape((samples.shape[0], -1)).max(axis=1)):
        upper_res[i, :, :, :] = max_val
        lower_res[i, :, :, :] = max_val
    upper_res[:, :, top_gap:top_gap + new_rows, :] = samples[:, :, :-useless_rows, :]
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


def make_train_xy(sample_list):
    for i in range(len(sample_list) - 1):
        assert sample_list[i].shape == sample_list[i + 1].shape
    x_list = list()

    mb_list = [add_miss_block(s) for s in sample_list]
    x_list.append(combine_pairs(sample_list + mb_list))

    tuple_list = [cut_image(s, .8) for s in sample_list]
    p8_list = list()
    for u, l in tuple_list: p8_list += [u, l]
    x_list.append(combine_pairs(p8_list))

    tuple_list = [cut_image(s, .6) for s in sample_list]
    p6_list = list()
    for u, l in tuple_list: p6_list += [u, l]
    x_list.append(combine_pairs(p6_list))

    # SMALLER DATABASE
    # x_list.append(combine_pairs([sample_list[0], sample_list[1]]))
    # x_list.append(combine_pairs([sample_list[0], p8_list[0]]))
    # x_list.append(combine_pairs([sample_list[1], p8_list[2]]))

    x = np.concatenate(x_list, axis=0)
    x_sizes = [0] + [len(i) for i in x_list]
    fake_x = np.zeros_like(x)
    for i in range(len(x_sizes) - 1):
        x_sizes[i + 1] += x_sizes[i]
        s = x_sizes[i]
        e = x_sizes[i + 1]
        tmp = np.copy(x[s:e, :, :, :])
        tmp[:, 0, :, :] = tmp[::-1, 0, :, :]
        fake_x[s:e, :, :, :] = tmp

    y = np.array([1] * x.shape[0] + [0] * x.shape[0])
    x = np.concatenate([x, fake_x], axis=0)

    # add reverse
    x = np.concatenate([x, x[:, ::-1, :, :]], axis=0)
    y = np.concatenate([y, y], axis=0)
    return x, y


def plot(train_loss, valid_loss, train_acc, valid_acc, tag):
    fig, ax1 = plt.subplots()
    x_range = np.arange(len(train_acc)) + 1
    ax1.plot(x_range, train_loss, 'red')
    ax1.plot(x_range, valid_loss, 'r--')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('validation and train loss', color='red')
    ax1.tick_params('y', colors='red')
    ax2 = ax1.twinx()
    ax2.plot(x_range, train_acc, 'blue')
    ax2.plot(x_range, valid_acc, 'b--')
    ax2.set_ylabel('validation and train accuracy', color='blue')
    ax2.tick_params('y', colors='blue')
    fig.tight_layout()
    plt.savefig('results/{}-plot.png'.format(tag))


def plot_hist(x, y, bin_num, tag):
    # bins = np.linspace(min(x + y), max(x + y), bin_num)
    bins = np.linspace(-10, +10, bin_num)
    plt.clf()
    plt.hist(x, bins, density=True, facecolor='g', alpha=0.5)
    plt.hist(y, bins, density=True, facecolor='r', alpha=0.5)
    plt.xlabel('Match_Score - Non_Match_Score')
    plt.title('Histogram of differences')
    plt.savefig('results/{}-hist.png'.format(tag))


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
