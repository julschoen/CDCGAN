import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform
import torchvision
from carbontracker.tracker import CarbonTracker

import os
import numpy as np
import itertools

from cdcgan import Discriminator as DCGAN
from biggan import Discriminator as BigGAN
from mmd import mix_rbf_mmd2, mix_rbf_cmmd2
from flows import (
    AffineConstantFlow, ActNorm, AffineHalfFlow, 
    SlowMAF, MAF, IAF, Invertible1x1Conv, NSF_AR, NSF_CL,
    NormalizingFlow, NormalizingFlowModel,
)
from fid import calculate_frechet_distance as FID


class Trainer():
    def __init__(self, params, train_loader):
        self.p = params

        self.losses = []
        if self.p.biggan and self.p.cifar:
            self.model = BigGAN().to(self.p.device)
        else:
            self.model = DCGAN(self.p).to(self.p.device)
            
        self.train_loader = train_loader
        self.gen = self.inf_train_gen()

        if self.p.norm_flow:
            flows = [AffineHalfFlow(dim=self.p.k, parity=i%2) for i in range(9)]
            prior = TransformedDistribution(MultivariateNormal(torch.zeros(self.p.k).to(self.p.device), torch.eye(self.p.k).to(self.p.device)), SigmoidTransform().inv)
            self.norm_flow = NormalizingFlowModel(prior, flows, self.p).to(self.p.device)
            self.normOpt = torch.optim.Adam(self.norm_flow.parameters(), lr=1e-4, weight_decay=1e-5)

        if not os.path.isdir(self.p.log_dir):
            os.mkdir(self.p.log_dir)

        if self.p.cifar:
            if self.p.init_ims:
                self.ims = torch.load('means.pt')
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
            vutils.make_grid(torch.tanh(self.ims), nrow=self.p.num_ims, padding=2, normalize=True)
            , os.path.join(path, f'{step}.png'))

    def shuffle(self):
        indices = torch.randperm(self.ims.shape[0])
        self.ims = torch.nn.Parameter(torch.index_select(self.ims, dim=0, index=indices.to(self.ims.device)))
        self.labels = torch.index_select(self.labels, dim=0, index=indices.to(self.labels.device))

    def save(self):
        path = os.path.join(self.p.log_dir, 'checkpoints')
        if not os.path.isdir(path):
            os.mkdir(path)
        file_name = os.path.join(path, 'data.pt')
        torch.save(torch.tanh(self.ims.cpu()), file_name)

        file_name = os.path.join(path, 'labels.pt')
        torch.save(self.labels.cpu(), file_name)

    def flow(self):
        for p in self.norm_flow.parameters():
            p.requires_grad = True

        data, labels = next(self.gen)
        enc = self.model(data.to(self.p.device), labels.to(self.p.device))

        zs, prior_logprob, log_det = self.norm_flow(enc.squeeze())
        logprob = prior_logprob + log_det
        loss = -torch.sum(logprob) # NLL

        if loss > 0:
            self.norm_flow.zero_grad()
            loss.backward()
            self.normOpt.step()

        for p in self.norm_flow.parameters():
            p.requires_grad = False

        return loss.detach().item()

    def total_variation_loss(self, img, weight=1, four=True):
        bs_img, c_img, h_img, w_img = img.size()

        tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
        tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()

        tv = weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

        if four:
            tv_h4 = torch.pow(img[:,:,:-1,:]-img[:,:,1:,:], 2).sum()
            tv_w4 = torch.pow(img[:,:,:,:-1]-img[:,:,:,1:], 2).sum()
            tv = tv + weight*(tv_h4+tv_w4)/(bs_img*c_img*h_img*w_img)
            tv = tv/2

        return tv

    def train(self):
        for p in self.model.parameters():
                    p.requires_grad = False
        self.ims.requires_grad = False

        for t in range(self.p.niter):
            self.tracker.epoch_start()

            if self.p.norm_flow:
                nf_loss = self.flow()

            for p in self.model.parameters():
                p.requires_grad = True
            for _ in range(self.p.iterD):
                if not self.p.biggan:
                    for p in self.model.parameters():
                        p.data.clamp_(-0.01, 0.01)

                data, labels = next(self.gen)

                self.model.zero_grad()
   
                encX = self.model(data.to(self.p.device), labels.to(self.p.device))
                encY = self.model(torch.tanh(self.ims), self.labels.to(self.p.device))

                if self.p.fid:
                    errD = -FID(encX.squeeze(), encY.squeeze())
                else:
                    if self.p.norm_flow:
                        encX, _, _ = self.norm_flow(encX.squeeze())
                        encY, _, _ = self.norm_flow(encY.squeeze())
                        encX = encX[-1].reshape(encX[0].shape[0],-1,1,1)
                        encY = encY[-1].reshape(encY[0].shape[0],-1,1,1)

                    mmd2_D = mix_rbf_mmd2(encX, encY, self.sigma_list)
                    mmd2_D = F.relu(mmd2_D)
                    errD = -torch.sqrt(mmd2_D)
                errD.backward()
                self.optD.step()


            for p in self.model.parameters():
                p.requires_grad = False

            self.ims.requires_grad = True
            for _ in range(self.p.iterIms):
                data, labels = next(self.gen)

                self.optIms.zero_grad()

                encX = self.model(data.to(self.p.device), labels.to(self.p.device))
                encY = self.model(torch.tanh(self.ims), self.labels.to(self.p.device))

                if self.p.fid:
                    errG = FID(encX.squeeze(), encY.squeeze())
                else:
                    if self.p.norm_flow:
                        encX, _, _ = self.norm_flow(encX.squeeze())
                        encY, _, _ = self.norm_flow(encY.squeeze())
                        encX = encX[-1].reshape(encX[0].shape[0],-1,1,1)
                        encY = encY[-1].reshape(encY[0].shape[0],-1,1,1)

                    if self.p.class_wise:
                        errG = 0
                        for i in range(10):
                            X = encX[labels == i]
                            Y = encY[self.labels == i]

                            if X.shape[0] < Y.shape[0]:
                                Y = Y[:X.shape[0]]
                            elif X.shape[0] > Y.shape[0]:
                                X = X[:Y.shape[0]]

                            l = mix_rbf_mmd2(X, Y, self.sigma_list)
                            errG = errG + torch.sqrt(F.relu(l))
                    else:
                        mmd2_G = mix_rbf_mmd2(encX, encY, self.sigma_list)
                        mmd2_G = F.relu(mmd2_G)

                        errG = torch.sqrt(mmd2_G)

                if self.p.corr:
                    corr = self.total_variation_loss(torch.tanh(self.ims))
                    errG = errG + corr * self.p.corr_coef
                
                errG.backward()
                self.optIms.step()
            self.ims.requires_grad = False

            self.tracker.epoch_end()

            if self.p.norm_flow:
                self.losses.append((errD.item(), errG.item(), nf_loss))
            else:
                self.losses.append((errD.item(), errG.item()))


            if ((t+1)%100 == 0) or (t==0):
                self.log_interpolation(t)
                s = '[{}|{}] ErrD: {:.4f}, ErrG: {:.4f}'.format(t+1, self.p.niter, errD.item(), errG.item())

                if self.p.norm_flow:
                    s = s+ ', Flow: {:.4f}'.format(nf_loss)
                if self.p.corr:
                    s =  s+ ', Corr: {:.4f}'.format(corr.item())
                
                print(s, flush=True)


        self.tracker.stop()
        self.save()
