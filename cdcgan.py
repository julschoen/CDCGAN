import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm

def conv(nc,ndf,kernel,stride,padding,bias,spectral):
    if spectral:
        return SpectralNorm(nn.Conv2d(nc, ndf, kernel, stride=stride, padding=padding, bias=bias))
    else:
        return nn.Conv2d(nc, ndf, kernel, stride=stride, padding=padding, bias=bias)

class Discriminator(nn.Module):
    # initializers
    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.p = params

        ch = 3 if params.cifar else 1
        final_kernel = 4 if params.cifar else 3
        d = params.filter
        im_size = 32 if params.cifar else 28

        self.conv1_1 = conv(ch, d//2, 4, 2, 1, False, params.spectral_norm)
        self.conv1_bn = nn.BatchNorm2d(d//2)

        self.conv1_2 = conv(10, d//2, 4, 2, 1, False, params.spectral_norm)

        self.conv2 = conv(d, d*2, 4, 2, 1, False, params.spectral_norm)
        self.conv2_bn = nn.BatchNorm2d(d*2)

        self.conv3 = conv(d*2, d*4, 4, 2, 1, False, params.spectral_norm)
        self.conv3_bn = nn.BatchNorm2d(d*4)

        self.conv4 = nn.Conv2d(d * 4, params.k, final_kernel, 1, 0, bias=False)

        self.fill = torch.zeros([10, 10, im_size, im_size]).to(params.device)
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

        x = self.conv1_1(input)
        if not self.p.spectral_norm:
            x = self.conv1_bn(x)
        x = F.leaky_relu(x)

        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)

        x = self.conv2(x)
        if not self.p.spectral_norm:
            x = self.conv2_bn(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv3(x)
        if not self.p.spectral_norm:
            x = self.conv3_bn(x)
        x = F.leaky_relu(x, 0.2)
        
        x = torch.sigmoid(self.conv4(x))
        return x