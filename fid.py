import torch
from torch.autograd import Function
import numpy as np
from sqrtm import sqrtm

def calculate_frechet_distance(act1, act2, eps=1e-6):
    mu1 = torch.mean(act1, dim=0)
    sigma1 = torch.cov(act1.T)

    mu2 = torch.mean(act2, dim=0)
    sigma2 = torch.cov(act2.T)
    diff = mu1 - mu2

    # Product might be almost singular
    covmean = sqrtm(torch.mm(sigma1, sigma2))

    if not torch.isfinite(covmean).all():
        offset = torch.eye(sigma1.shape[0]) * eps
        covmean = sqrtm(torch.mm((sigma1 + offset), (sigma2 + offset)))

    # Numerical error might give slight imaginary component
    if torch.is_complex(covmean):
        covmean = covmean.real

    tr_covmean = torch.trace(covmean)

    print(tr_covmean)

    return (torch.inner(diff, diff) + torch.trace(sigma1)
            + torch.trace(sigma2) - 2 * tr_covmean)