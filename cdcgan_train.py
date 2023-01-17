import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from carbontracker.tracker import CarbonTracker

import os
import numpy as np
from cdcgan import Discriminator
from mmd import mix_rbf_mmd2, mix_rbf_cmmd2


class Trainer():
    def __init__(self, params, train_loader):
        self.p = params

        self.losses = []
        self.test_losses = []
        self.model = Discriminator(self.p).to(self.p.device)
        self.train_loader = train_loader
        self.gen = self.inf_train_gen()

        if not os.path.isdir(self.p.log_dir):
            os.mkdir(self.p.log_dir)

        if self.p.cifar:
            if self.p.init_ims:
                self.ims, _ = next(self.gen)
                self.ims = self.ims.to(self.p.device)
            else:
                self.ims = torch.randn(10*self.p.num_ims,3,32,32).to(self.p.device)
        else:
            if self.p.init_ims:
                self.ims, _ = next(self.gen)
                self.ims = self.ims.to(self.p.device)
            else:
                self.ims = torch.randn(10*self.p.num_ims,1,28,28).to(self.p.device)
        self.ims = torch.nn.Parameter(self.ims)
        self.labels = torch.arange(10).repeat(self.p.num_ims,1).T.flatten()
        
        self.sigma_list = [1, 2, 4, 8, 16, 24, 32, 64]

        # setup optimizer
        self.optD = torch.optim.Adam(self.model.parameters(), lr=self.p.lr)
        self.optIms = torch.optim.Adam([self.ims], lr=self.p.lrIms)

        if not os.path.isdir('./cdc_carbon'):
            os.mkdir('./cdc_carbon')
        self.tracker = CarbonTracker(epochs=self.p.niter, log_dir='./cdc_carbon/')

    def inf_train_gen(self):
        while True:
            for data in self.train_loader:
                yield data

    def log_interpolation(self, step):
        path = os.path.join(self.p.log_dir, 'images')
        if not os.path.isdir(path):
            os.mkdir(path)
        torchvision.utils.save_image(
            vutils.make_grid(torch.sigmoid(self.ims), padding=2, normalize=True)
            , os.path.join(path, f'{step}.png'))

    def shuffle(self):
        indices = torch.randperm(self.ims.shape[0])
        self.ims = torch.nn.Parameter(torch.index_select(self.ims, dim=0, index=indices))
        self.labels = torch.index_select(self.labels, dim=0, index=indices)

    def save(self):
        path = os.path.join(self.p.log_dir, 'checkpoints')
        if not os.path.isdir(path):
            os.mkdir(path)
        file_name = os.path.join(path, 'data.pt')
        torch.save(torch.sigmoid(self.ims.cpu()), file_name)

        file_name = os.path.join(path, 'labels.pt')
        torch.save(self.labels.cpu(), file_name)

    def train(self):
        for p in self.model.parameters():
                    p.requires_grad = False
        self.ims.requires_grad = False

        for t in range(self.p.niter):
            self.tracker.epoch_start()
            for p in self.model.parameters():
                p.requires_grad = True
            for _ in range(1):
                for p in self.model.parameters():
                    p.data.clamp_(-0.01, 0.01)

                data, labels = next(self.gen)

                self.model.zero_grad()
                encX = self.model(data.to(self.p.device), labels)
                self.shuffle()
                encY = self.model(torch.sigmoid(self.ims), self.labels)

                if self.p.cmmd:
                    mmd2_D = mix_rbf_cmmd2(encX, encY, labels, self.labels, self.sigma_list)
                else:
                    mmd2_D = mix_rbf_mmd2(encX, encY, self.sigma_list)
                mmd2_D = F.relu(mmd2_D)
                errD = -torch.sqrt(mmd2_D)
                errD.backward()
                self.optD.step()


            for p in self.model.parameters():
                p.requires_grad = False

            self.shuffle()
            self.ims.requires_grad = True
            data, labels = next(self.gen)

            self.optIms.zero_grad()

            encX = self.model(data.to(self.p.device), labels)
            encY = self.model(torch.sigmoid(self.ims), self.labels)

            if self.p.cmmd:
                mmd2_G = mix_rbf_cmmd2(encX, encY, labels, self.labels, self.sigma_list)
            else:
                mmd2_G = mix_rbf_mmd2(encX, encY, self.sigma_list)
            mmd2_G = F.relu(mmd2_G)

            errG = torch.sqrt(mmd2_G)
            errG.backward()
            self.optIms.step()
            self.ims.requires_grad = False

            self.tracker.epoch_end()

            if ((t+1)%100 == 0) or (t==0):
                self.log_interpolation(t)
                print('[{}|{}] ErrD: {:.4f}, ErrG: {:.4f}'.format(t+1, self.p.niter, errD.item(), errG.item()))


        self.tracker.stop()
        self.save()
