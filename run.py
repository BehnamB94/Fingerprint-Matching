import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from modules.dataset import ImageDataset
from modules.net import Cnn
from modules.tools import plot, make_xy, make_train_xy, check_data, check_sample

batch_size = 700
learning_rate = 1e-4
max_loss_diff = 0.04
min_epochs = 20
max_epochs = 50

IMAGE_ROW = 181
IMAGE_COL = 181

if len(sys.argv) > 1:
    RUN_TAG = sys.argv[1]
else:
    RUN_TAG = 'TEST'
print(RUN_TAG, '\nrunning with learning rate = {}'.format(learning_rate), 'and batch size = {}'.format(batch_size))

#######################################################################################
# PREPARE DATA
#######################################################################################

# NIST-DB4
# loaded = np.load('dataset/images_181_181.npz')
# sample1 = loaded['sample1'].reshape((-1, 1, IMAGE_ROW, IMAGE_COL))
# sample2 = loaded['sample2'].reshape((-1, 1, IMAGE_ROW, IMAGE_COL))
# sample_list = [sample1, sample2]
# test_size = 400  # from 2000
# valid_size = 100  # from 2000
# check_sample(sample1[-20:], sample2[-20:])

# FVC2002
loaded = np.load('dataset/fvc_181_181.npz')
loaded = loaded['DB1']
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
valid_x, valid_y = make_xy([s[-valid_size:] for s in sample_list])
for i in range(len(sample_list)):
    sample_list[i] = sample_list[i][:-valid_size]

# PREPARE TRAIN SET
train_x, train_y = make_train_xy(sample_list)
# check_data(train_x[-20:], train_y[-20:])
# exit()
np.random.seed()

print('Data Prepared:\n'
      '\tTRAIN:{}\n'
      '\tVALID:{}\n'
      '\tTEST:{}'.format(train_x.shape[0], valid_x.shape[0], test_x.shape[0]))

#######################################################################################
# DO TRAIN
#######################################################################################
train_loader = DataLoader(ImageDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(ImageDataset(valid_x, valid_y), batch_size=batch_size, shuffle=True)

net = Cnn()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

plot_train_loss, plot_train_acc, plot_valid_loss, plot_valid_acc = [], [], [], []
for epoch in range(max_epochs):
    train_loss_list = list()
    valid_loss_list = list()
    train_correct = 0
    valid_correct = 0
    for features, labels in train_loader:  # For each batch, do:
        features = torch.autograd.Variable(features.float())
        labels = torch.autograd.Variable(labels.long())
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
        # valid_loss_list.append(.5)
        outputs = net(features)
        valid_correct += torch.sum(torch.argmax(outputs, 1) == labels)
        loss = loss_fn(outputs, labels)
        valid_loss_list.append(loss.data.item())

    saved = False
    train_acc = train_correct.item() / train_x.shape[0]
    valid_acc = valid_correct.item() / valid_x.shape[0]
    train_loss = sum(train_loss_list) / len(train_loss_list)
    valid_loss = sum(valid_loss_list) / len(valid_loss_list)

    plot_train_acc.append(train_acc)
    plot_train_loss.append(train_loss)
    plot_valid_loss.append(valid_loss)
    plot_valid_acc.append(valid_acc)

    if valid_loss == min(plot_valid_loss):
        torch.save(net.state_dict(), 'best_model.pkl')
        saved = True
    print(epoch + 1,
          '\ttrain loss={:.3f}'.format(train_loss),
          '\ttrain acc={:.3f}'.format(train_acc),
          '\tvalid loss={:.3f}'.format(valid_loss),
          '\tvalid acc={:.3f}'.format(valid_acc),
          '\t> saved as best model!' if saved else '')
    if epoch > min_epochs:
        if valid_loss - min(plot_valid_loss[-5:]) > max_loss_diff:
            break

#######################################################################################
# TEST BEST MODEL
#######################################################################################
test_loader = DataLoader(ImageDataset(test_x, test_y), batch_size=batch_size, shuffle=True)

net.load_state_dict(torch.load('best_model.pkl'))
net.eval()
test_correct = 0
for features, labels in test_loader:  # For each batch, do:
    features = torch.autograd.Variable(features.float())
    labels = torch.autograd.Variable(labels.long())
    outputs = net(features)
    test_correct += torch.sum(torch.argmax(outputs, 1) == labels)
print('\ttest acc on best model =', test_correct.item() / test_x.shape[0])
plot(plot_train_loss, plot_valid_loss, plot_train_acc, plot_valid_acc, tag=RUN_TAG)
