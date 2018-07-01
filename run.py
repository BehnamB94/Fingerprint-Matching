import numpy as np
import torch
from torch.utils.data import DataLoader

from modules.dataset import ImageDataset
from modules.net import Cnn
from modules.tools import plot

batch_size = 700
learning_rate = 5e-4
max_epochs = 50
test_size = 400  # from 2000
valid_size = 100  # from 2000

IMAGE_ROW = 97
IMAGE_COL = 93

RUN_TAG = 'shuffle-dropout2-{}'.format(batch_size)

print('running with learning rate = {}'.format(learning_rate), 'and batch size = {}'.format(batch_size))

#######################################################################################
# PREPARE DATA
#######################################################################################
loaded = np.load('dataset/images.npz')
sample1 = loaded['sample1'].reshape((-1, 1, IMAGE_ROW, IMAGE_COL))
sample2 = loaded['sample2'].reshape((-1, 1, IMAGE_ROW, IMAGE_COL))

# SHUFFLE DATA
np.random.seed(0)
ind = np.random.permutation(range(sample1.shape[0])).astype(np.int)
sample1 = sample1[ind]
sample2 = sample2[ind]
np.random.seed()

# PREPARE TEST SET
test_x = np.concatenate([sample1[-test_size:], sample2[-test_size:]], axis=1)
fake_test_x = np.copy(test_x)
fake_test_x[:, 0, :, :] = fake_test_x[::-1, 0, :, :]
test_x = np.concatenate([test_x, fake_test_x], axis=0)
test_y = np.array([1] * test_size + [0] * test_size)
sample1 = sample1[:-test_size]
sample2 = sample2[:-test_size]

# PREPARE VALIDATION SET
valid_x = np.concatenate([sample1[-valid_size:], sample2[-valid_size:]], axis=1)
fake_valid_x = np.copy(valid_x)
fake_valid_x[:, 0, :, :] = fake_valid_x[::-1, 0, :, :]
valid_x = np.concatenate([valid_x, fake_valid_x], axis=0)
valid_y = np.array([1] * valid_size + [0] * valid_size)
sample1 = sample1[:-valid_size]
sample2 = sample2[:-valid_size]

# PREPARE TRAIN SET
train_size = sample1.shape[0]
train_x = np.zeros((train_size * 3, 2, IMAGE_ROW, IMAGE_COL), dtype=np.uint8)
train_y = np.array([1] * (train_size * 3) + [0] * (train_size * 3))
train_x[:train_size, 0, :, :] \
    = train_x[train_size:-train_size, 0, :, :] \
    = train_x[train_size:-train_size, 1, :, :] \
    = sample1[:, 0, :, :]
train_x[:train_size, 1, :, :] \
    = train_x[-train_size:, 0, :, :] \
    = train_x[-train_size:, 1, :, :] \
    = sample2[:, 0, :, :]
train_x[train_size:, 1, :20, :] = 255
fake_train_x = np.copy(train_x)
fake_train_x[:, 0, :, :] = fake_train_x[::-1, 0, :, :]
train_x = np.concatenate([train_x, fake_train_x], axis=0)
# add reverse
train_x = np.concatenate([train_x, train_x[:, ::-1, :, :]], axis=0)
train_y = np.concatenate([train_y, train_y], axis=0)

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

best_valid_acc = -1
plot_train_loss, plot_train_acc, plot_valid_acc = [], [], []
for epoch in range(max_epochs):
    loss_list = list()

    train_correct = 0
    valid_correct = 0
    for features, labels in train_loader:  # For each batch, do:
        features = torch.autograd.Variable(features.float())
        labels = torch.autograd.Variable(labels)
        outputs = net(features)
        train_correct += torch.sum(torch.argmax(outputs, 1) == labels)
        loss = loss_fn(outputs, labels)
        loss_list.append(loss.data.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for features, labels in valid_loader:  # For each batch, do:
        features = torch.autograd.Variable(features.float())
        labels = torch.autograd.Variable(labels)
        outputs = net(features)
        valid_correct += torch.sum(torch.argmax(outputs, 1) == labels)

    saved = False
    train_acc = train_correct.item() / train_x.shape[0]
    valid_acc = valid_correct.item() / valid_x.shape[0]
    train_loss = sum(loss_list) / len(loss_list)

    plot_train_loss.append(train_loss)
    plot_train_acc.append(train_acc)
    plot_valid_acc.append(valid_acc)

    if valid_acc > best_valid_acc:
        torch.save(net.state_dict(), 'best_model.pkl')
        best_valid_acc = valid_acc
        saved = True
    print(epoch + 1, '\ttrain loss={:.3f}'.format(train_loss),
          '\ttrain acc={:.3f}'.format(train_acc),
          '\tvalid acc={:.3f}'.format(valid_acc),
          '\t> saved as best model!' if saved else '')
    if valid_acc > .55 and best_valid_acc - valid_acc > 0.05: break

#######################################################################################
# TEST BEST MODEL
#######################################################################################
test_loader = DataLoader(ImageDataset(test_x, test_y), batch_size=batch_size, shuffle=True)

net.load_state_dict(torch.load('best_model.pkl'))
net.eval()
test_correct = 0
for features, labels in test_loader:  # For each batch, do:
    features = torch.autograd.Variable(features.float())
    labels = torch.autograd.Variable(labels)
    outputs = net(features)
    test_correct += torch.sum(torch.argmax(outputs, 1) == labels)
print('\ttest acc on best model =', test_correct.item() / test_x.shape[0])
plot(plot_train_loss, plot_train_acc, plot_valid_acc, tag=RUN_TAG)
