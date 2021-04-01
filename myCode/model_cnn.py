import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNmodel(nn.Module):
    def __init__(self):
        super(CNNmodel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=9,kernel_size=2, stride=1, padding=1, out_channels=18)
        self.bn1 = nn.BatchNorm1d(num_features=18, momentum=0.95).train()
        self.max_pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

        self.conv2 = nn.Conv1d(in_channels=18, kernel_size=2, stride= 1, padding=1, out_channels=36)
        self.bn2 = nn.BatchNorm1d(num_features=36, momentum=0.95).train()
        self.max_pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

        self.conv3 = nn.Conv1d(in_channels=36, kernel_size=2, stride=1, padding=1, out_channels=72)
        self.bn3 = nn.BatchNorm1d(num_features=72, momentum=0.95).train()
        self.max_pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)


        self.conv4 = nn.Conv1d(in_channels=72, kernel_size=2, stride=1, padding=1,out_channels=144)
        self.bn4 = nn.BatchNorm1d(num_features=144, momentum=0.95).train()
        self.max_pool4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

        self.hidden1 = nn.Dropout(0.2)

        self.dense = nn.Linear(10*144,2)

    def forward(self, x):
        x = x.float()
        c1 = F.tanh(self.conv1(x))
        bn1 = self.bn1(c1)
        m1 = self.max_pool1(bn1)

        c2 = F.tanh(self.conv2(m1))
        bn2 = self.bn2(c2)
        m2 = self.max_pool2(bn2)

        c3 = F.tanh(self.conv3(m2))
        bn3 = self.bn3(c3)
        m3 = self.max_pool3(bn3)

        c4 = F.tanh(self.conv4(m3))
        bn4 = self.bn4(c4)
        m4 = self.max_pool4(bn4)
        m4 = self.hidden1(m4)

        m4 = torch.reshape(m4, (-1, 10*144))

        res = self.dense(m4)
        return res
