import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from modules.dataset import ImageDataset
from modules.net import TrainedAlexnet
from torch.optim.lr_scheduler import StepLR
from modules.tools import plot, make_xy, make_train_xy, plot_hist

cpu_batch_size = 400
gpu_batch_size = 115

learning_rate = 1e-2
momentum = 0.9
weight_decay = 0.0005
step_size = 220
gamma = 0.1

max_loss_diff = 0.04
min_epochs = 40
max_epochs = 100

IMAGE_ROW = 227
IMAGE_COL = 227
VALID_DBs = ['NIST', 'FVC1', 'FVC2', 'FVC3', 'FVC4']

parser = argparse.ArgumentParser()
parser.add_argument('Dataset', choices=VALID_DBs, help='Dataset name')
parser.add_argument('-tag', dest='TAG', default='TEST', help='set a tag (use for save results)')
parser.add_argument('-cont', dest='CONT', type=int, default=None, help='continue last run from specific epoch')
parser.add_argument('--test', dest='TEST', action='store_true', default=False, help='only test from existing model')
parser.add_argument('--gpu', dest='GPU', action='store_true', default=False, help='use gpu for running')
args = parser.parse_args()  # ['FVC1', '--gpu']
if args.CONT is not None and args.TEST is True:
    raise Exception('Can not use --test and -cont options in the same time')
batch_size = gpu_batch_size if args.GPU else cpu_batch_size


def print_and_log(*content):
    content = ' '.join(content)
    print(content)
    with open('results/{}-log.txt'.format(args.TAG), 'a') as file:
        file.write('{}\n'.format(content))


print_and_log('\n', ''.join(['#'] * 50),
              '\nTRAIN' if not args.TEST else '\nTEST',
              '\ton Dataset:', args.Dataset,
              '\ttag:', args.TAG, '(continue)' if args.CONT is not None else '')

#######################################################################################
# PREPARE DATA
#######################################################################################
if args.Dataset == 'NIST':
    loaded = np.load('dataset/images_{}_{}.npz'.format(IMAGE_ROW, IMAGE_COL))
    sample1 = loaded['sample1'].reshape((-1, 1, IMAGE_ROW, IMAGE_COL))
    sample2 = loaded['sample2'].reshape((-1, 1, IMAGE_ROW, IMAGE_COL))
    sample_list = [sample1, sample2]
    test_size = 400  # from 2000
    valid_size = 100  # from 2000
else:  # FVC2002
    loaded = np.load('dataset/fvc_{}_{}.npz'.format(IMAGE_ROW, IMAGE_COL))
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
net = TrainedAlexnet()
start_epoch = 0
if args.CONT is not None:
    net.load_state_dict(torch.load('results/{}-model.pkl'.format(args.TAG)))
    start_epoch = args.CONT
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

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

        net = net.train()
        scheduler.step()
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
        del features, labels

        net = net.eval()
        with torch.no_grad():
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
            del features, labels

        with torch.no_grad():
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
            del features, labels

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
                      '\tTrain(L={:.3f}'.format(train_loss),
                      ' | A={:.3f})'.format(train_acc),
                      '\tValid(L={:.3f}'.format(valid_loss),
                      ' | A={:.3f})'.format(valid_acc),
                      '\tTest(L={:.3f}'.format(test_loss),
                      ' | A={:.3f})'.format(test_acc),
                      '\t> Best' if saved else '')
        if epoch > min_epochs:
            if valid_loss - min(plot_valid_loss[-5:]) > max_loss_diff:
                break
        plot(plot_train_loss, plot_valid_loss,
             plot_test_loss, plot_train_acc,
             plot_valid_acc, plot_test_acc, tag=args.TAG)
#######################################################################################
# TEST BEST MODEL
#######################################################################################
test_loader = DataLoader(ImageDataset(test_x, test_y), batch_size=batch_size, shuffle=True)

net.load_state_dict(torch.load('results/{}-model.pkl'.format(args.TAG)))
net = net.eval()
test_correct = 0
true_list = list()
false_list = list()
csv_diff_list = list()
csv_label_list = list()
with torch.no_grad():
    for features, labels in test_loader:  # For each batch, do:
        features = torch.autograd.Variable(features.float())
        labels = torch.autograd.Variable(labels.long())
        if args.GPU:
            features = features.cuda()
            labels = labels.cuda()
        outputs = net(features)
        diff = outputs[:, 1] - outputs[:, 0]
        csv_diff_list += diff.tolist()
        csv_label_list += labels.tolist()
        true_list += diff[labels == 1].tolist()
        false_list += diff[labels == 0].tolist()
        test_correct += torch.sum(torch.argmax(outputs, 1) == labels)
    del features, labels
print_and_log('>>> Test acc on best model =', str(test_correct.item() / test_x.shape[0]))
print_and_log('FNMR=', str(100 * sum(np.array(true_list) < 0) / test_x.shape[0] * 2), '%')
print_and_log('FMR=', str(100 * sum(np.array(false_list) > 0) / test_x.shape[0] * 2), '%')
plot_hist(true_list, false_list, bin_num=100, tag=args.TAG)

import csv

with open('results/24-alexnet-diff.csv', 'w') as csv_file:
    wr = csv.writer(csv_file)
    wr.writerow(['lbl', 'diff'])
    for lbl, diff in zip(csv_label_list, csv_diff_list):
        wr.writerow([lbl, diff])
