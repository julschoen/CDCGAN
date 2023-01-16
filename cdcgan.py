import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128, k=100):
        super(Discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(1, d//2, 4, 2, 1)
        self.conv1_2 = nn.Conv2d(10, d//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, k, 3, 1, 0)
        self.fill = torch.zeros([10, 10, 28, 28])
        for i in range(10):
            self.fill[i, i, :, :] = 1

        self.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.2)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.2)
            nn.init.constant_(m.bias.data, 0)

    # forward method
    def forward(self, input, label):
        label = self.fill[label]
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)

        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = torch.sigmoid(self.conv4(x))

        return x