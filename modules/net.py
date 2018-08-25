import torch
import torch.nn as nn
from torchvision.models import alexnet, inception_v3


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),

            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LogSigmoid(),
            nn.Dropout(p=.5),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=.5),

            nn.Linear(64, 2),
        )

    def forward(self, data):
        out = self.convolution(data)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class MyCnn(nn.Module):
    def __init__(self):
        super(MyCnn, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),

            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * 128, 128),
            nn.ReLU(),
            nn.Dropout(p=.5),

            # nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Dropout(p=.5),

            nn.Linear(128, 2),  # (64, 2),
        )

    def forward(self, data):
        out1 = self.convolution(data[:, 0, :, :].unsqueeze_(1))
        out2 = self.convolution(data[:, 1, :, :].unsqueeze_(1))
        out1 = out1.view(out1.size(0), -1)
        out2 = out2.view(out2.size(0), -1)
        out = torch.cat((out1, out2), 1)
        return self.fc(out)


class NewCnn(nn.Module):
    def __init__(self):
        super(NewCnn, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, ),  # 179
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 90

            nn.Conv2d(8, 16, kernel_size=3),  # 88
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 44

            nn.Conv2d(16, 16, kernel_size=3),  # 42
            nn.ReLU(),

            nn.Conv2d(16, 16, kernel_size=3),  # 40
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 20

        )

        self.fc = nn.Sequential(
            nn.Linear(19 * 19 * 16 * 2, 128),
            nn.ReLU(),
            nn.Dropout(p=.5),

            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Dropout(p=.5),

            nn.Linear(16, 2),
        )

    def forward(self, data):
        out1 = self.convolution(data[:, 0, :, :].unsqueeze_(1))
        out2 = self.convolution(data[:, 1, :, :].unsqueeze_(1))
        out1 = out1.view(out1.size(0), -1)
        out2 = out2.view(out2.size(0), -1)
        out = torch.cat((out1, out2), 1)
        return self.fc(out)


class TrainedAlexnet(nn.Module):
    def __init__(self):
        super(TrainedAlexnet, self).__init__()
        self.alex = nn.Sequential(*list(alexnet(pretrained=True).children())[:-1])
        self.fully_connected = nn.Sequential(
            nn.Linear(256 * 6 * 6 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 2),
        )

    def forward(self, data):
        im1 = data[:, 0, :, :].unsqueeze_(1)
        out1 = self.alex(torch.cat((im1, im1, im1), 1))
        out1 = out1.view(out1.size(0), -1)

        im2 = data[:, 1, :, :].unsqueeze_(1)
        out2 = self.alex(torch.cat((im2, im2, im2), 1))
        out2 = out2.view(out2.size(0), -1)

        out = torch.cat((out1, out2), 1)
        return self.fully_connected(out)


class TrainedInception(nn.Module):
    def __init__(self):
        super(TrainedInception, self).__init__()
        self.inception = nn.Sequential(*list(inception_v3(pretrained=True).children())[:-1])
        self.fully_connected = nn.Linear(2048, 2)

    def forward(self, data):
        im1 = data[:, 0, :, :].unsqueeze_(1)
        out1 = self.inception(torch.cat((im1, im1, im1), 1))
        out1 = out1.view(out1.size(0), -1)

        im2 = data[:, 1, :, :].unsqueeze_(1)
        out2 = self.inception(torch.cat((im2, im2, im2), 1))
        out2 = out2.view(out2.size(0), -1)

        out = torch.cat((out1, out2), 1)
        return self.fully_connected(out)


class Cnn4(nn.Module):
    def __init__(self):
        super(Cnn4, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(48, 128, kernel_size=5, stride=1, groups=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(128, 192, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(192, 128, kernel_size=3, stride=1, groups=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(14 * 14 * 128 * 2, 2048),
            nn.ReLU(),
            nn.Dropout(p=.5),

#            nn.Linear(25088 * 2, 2048),
#            nn.ReLU(),
#            nn.Dropout(p=.5),

            nn.Linear(2048, 2),
        )

    def forward(self, data):
        out1 = self.convolution(data[:, 0, :, :].unsqueeze_(1))
        out2 = self.convolution(data[:, 1, :, :].unsqueeze_(1))
        out1 = out1.view(out1.size(0), -1)
        out2 = out2.view(out2.size(0), -1)
        out = torch.cat((out1, out2), 1)
        return self.fc(out)
