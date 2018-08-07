import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from modules.dataset import ImageDataset
from modules.net import MyCnn
from modules.tools import plot, make_xy, make_train_xy, plot_hist

batch_size = 150
learning_rate = 5e-5
max_loss_diff = 0.04
min_epochs = 100
max_epochs = 5

IMAGE_ROW = 181
IMAGE_COL = 181
VALID_DBs = ['NIST', 'FVC1', 'FVC2', 'FVC3', 'FVC4']

parser = argparse.ArgumentParser()
parser.add_argument('Dataset', choices=VALID_DBs, help='Dataset name')
parser.add_argument('-tag', dest='TAG', default='TEST', help='set a tag (use for save results)')
parser.add_argument('-cont', dest='CONT', type=int, default=None, help='continue last run from specific epoch')
parser.add_argument('--test', dest='TEST', action='store_true', default=False, help='only test from existing model')
parser.add_argument('--gpu', dest='GPU', action='store_true', default=False, help='use gpu for running')
args = parser.parse_args()
if args.CONT is not None and args.TEST is True:
    raise Exception('Can not use --test and -cont options in the same time')


def print_and_log(*content):
    content = ' '.join(content)
    print(content)
    with open('results/{}-log.txt'.format(args.TAG), 'a') as file:
        file.write('{}\n'.format(content))


print_and_log('\n', ''.join(['#'] * 50),
              '\nTRAIN' if not args.TEST else '\nTEST',
              args.TAG, '(continue)' if args.CONT is not None else '', )

#######################################################################################
# PREPARE DATA
#######################################################################################
if args.Dataset == 'NIST':
    loaded = np.load('dataset/images_181_181.npz')
    sample1 = loaded['sample1'].reshape((-1, 1, IMAGE_ROW, IMAGE_COL))
    sample2 = loaded['sample2'].reshape((-1, 1, IMAGE_ROW, IMAGE_COL))
    sample_list = [sample1, sample2]
    test_size = 400  # from 2000
    valid_size = 100  # from 2000
else:  # FVC2002
    loaded = np.load('dataset/fvc_181_181.npz')
    loaded = loaded['DB{}'.format(args.Dataset[-1])]
    sample_list = [loaded[:, i, :, :].reshape(-1, 1, IMAGE_ROW, IMAGE_COL) for i in range(8)]
    test_size = 10  # from 110
    valid_size = 10  # from 110

# SHUFFLE DATA
np.random.seed(0)
ind = np.random.permutation(range(sample_list[0].shape[0])).astype(np.int)
for i in range(len(sample_list)):
    sample_list[i] = sample_list[i][ind]

# PREPARE TEST SET
test_x, test_y = make_xy([s[-test_size:] for s in sample_list])
for i in range(len(sample_list)):
    sample_list[i] = sample_list[i][:-test_size]

# PREPARE VALIDATION SET
if not args.TEST:
    valid_x, valid_y = make_xy([s[-valid_size:] for s in sample_list])
    for i in range(len(sample_list)):
        sample_list[i] = sample_list[i][:-valid_size]

    # PREPARE TRAIN SET
    train_x, train_y = make_train_xy(sample_list)
np.random.seed()

print_and_log('Data Prepared:\n',
              '\tTRAIN:{}\n'.format(train_x.shape[0]) if not args.TEST else '',
              '\tVALID:{}\n'.format(valid_x.shape[0]) if not args.TEST else '',
              '\tTEST:{}'.format(test_x.shape[0]))

#######################################################################################
# LOAD OR CREATE MODEL
#######################################################################################
net = MyCnn()
start_epoch = 0
if args.CONT is not None:
    net.load_state_dict(torch.load('results/{}-model.pkl'.format(args.TAG)))
    start_epoch = args.CONT
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

if args.GPU: net = net.cuda()

if not args.TEST:
    #######################################################################################
    # DO TRAIN
    #######################################################################################
    train_loader = DataLoader(ImageDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(ImageDataset(valid_x, valid_y), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ImageDataset(test_x, test_y), batch_size=batch_size, shuffle=True)

    plot_train_loss, plot_train_acc, plot_valid_loss, plot_valid_acc = [], [], [], []
    plot_test_loss, plot_test_acc = [], []
    for epoch in range(start_epoch, max_epochs):
        train_loss_list = list()
        valid_loss_list = list()
        test_loss_list = list()
        train_correct = 0
        valid_correct = 0
        test_correct = 0
        for features, labels in train_loader:  # For each batch, do:
            features = torch.autograd.Variable(features.float())
            labels = torch.autograd.Variable(labels.long())
            if args.GPU:
                features = features.cuda()
                labels = labels.cuda()
            outputs = net(features)
            train_correct += torch.sum(torch.argmax(outputs, 1) == labels)
            loss = loss_fn(outputs, labels)
            train_loss_list.append(loss.data.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for features, labels in valid_loader:  # For each batch, do:
            features = torch.autograd.Variable(features.float())
            labels = torch.autograd.Variable(labels.long())
            if args.GPU:
                features = features.cuda()
                labels = labels.cuda()
            outputs = net(features)
            valid_correct += torch.sum(torch.argmax(outputs, 1) == labels)
            loss = loss_fn(outputs, labels)
            valid_loss_list.append(loss.data.item())

        for features, labels in test_loader:  # For each batch, do:
            features = torch.autograd.Variable(features.float())
            labels = torch.autograd.Variable(labels.long())
            if args.GPU:
                features = features.cuda()
                labels = labels.cuda()
            outputs = net(features)
            test_correct += torch.sum(torch.argmax(outputs, 1) == labels)
            loss = loss_fn(outputs, labels)
            test_loss_list.append(loss.data.item())

        saved = False
        train_acc = train_correct.item() / train_x.shape[0]
        valid_acc = valid_correct.item() / valid_x.shape[0]
        test_acc = test_correct.item() / test_x.shape[0]
        train_loss = sum(train_loss_list) / len(train_loss_list)
        valid_loss = sum(valid_loss_list) / len(valid_loss_list)
        test_loss = sum(test_loss_list) / len(test_loss_list)

        plot_train_acc.append(train_acc)
        plot_train_loss.append(train_loss)
        plot_valid_loss.append(valid_loss)
        plot_valid_acc.append(valid_acc)
        plot_test_loss.append(test_loss)
        plot_test_acc.append(test_acc)

        if valid_loss == min(plot_valid_loss):
            torch.save(net.state_dict(), 'results/{}-model.pkl'.format(args.TAG))
            saved = True
        print_and_log(str(epoch + 1),
                      '\ttrain loss={:.3f}'.format(train_loss),
                      '\ttrain acc={:.3f}'.format(train_acc),
                      '\tvalid loss={:.3f}'.format(valid_loss),
                      '\tvalid acc={:.3f}'.format(valid_acc),
                      '\t> saved as best model!' if saved else '')
        if epoch > min_epochs:
            if valid_loss - min(plot_valid_loss[-5:]) > max_loss_diff:
                break
    plot(plot_train_loss, plot_valid_loss, plot_test_loss, plot_train_acc, plot_valid_acc, plot_test_acc, tag=args.TAG)
#######################################################################################
# TEST BEST MODEL
#######################################################################################
test_loader = DataLoader(ImageDataset(test_x, test_y), batch_size=batch_size, shuffle=True)

net.load_state_dict(torch.load('results/{}-model.pkl'.format(args.TAG)))
net.eval()
test_correct = 0
true_list = list()
false_list = list()
for features, labels in test_loader:  # For each batch, do:
    features = torch.autograd.Variable(features.float())
    labels = torch.autograd.Variable(labels.long())
    if args.GPU:
        features = features.cuda()
        labels = labels.cuda()
    outputs = net(features)
    diff = outputs[:, 1] - outputs[:, 0]
    true_list += diff[labels == 1].tolist()
    false_list += diff[labels == 0].tolist()
    test_correct += torch.sum(torch.argmax(outputs, 1) == labels)
print_and_log('>>> test acc on best model =', str(test_correct.item() / test_x.shape[0]))
plot_hist(true_list, false_list, bin_num=100, tag=args.TAG)
