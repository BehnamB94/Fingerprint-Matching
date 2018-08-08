import torch
import torch.nn as nn


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
            nn.Conv2d(1, 4, kernel_size=3),  # 179
            nn.ReLU(),

            nn.Conv2d(4, 8, kernel_size=3, ),  # 177
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 89

            nn.Conv2d(8, 8, kernel_size=3),  # 87
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 44

            nn.Conv2d(8, 8, kernel_size=3),  # 42
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 21
        )

        self.fc = nn.Sequential(
            nn.Linear(20 * 20 * 8 * 2, 100),
            nn.ReLU(),
            nn.Dropout(p=.5),

            nn.Linear(100, 2),
        )

    def forward(self, data):
        out1 = self.convolution(data[:, 0, :, :].unsqueeze_(1))
        out2 = self.convolution(data[:, 1, :, :].unsqueeze_(1))
        out1 = out1.view(out1.size(0), -1)
        out2 = out2.view(out2.size(0), -1)
        out = torch.cat((out1, out2), 1)
        return self.fc(out)
