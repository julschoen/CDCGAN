import argparse
import torch
from torchvision import datasets, transforms
from cdcgan_train import Trainer

class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        sample = sample-sample.min()
        sample = sample/sample.max()
        return sample

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # General Training
    parser.add_argument('--batch-size', type=int, default= 100, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--niter', type=int, default=20000, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--lrIms', type=float, default=1e-3, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--num_ims', type=int, default=10)
    parser.add_argument('--cifar', type=bool, default=True)
    parser.add_argument('--cmmd', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--init_ims', type=bool, default=False)
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--norm_flow', type=bool, default=False)
    parser.add_argument('--biggan', type=bool, default=False)
    parser.add_argument('--iterIms', type=int, default=1)
    parser.add_argument('--iterD', type=int, default=1)
    parser.add_argument('--reloadD', type=bool, default=False)

    # Model Params
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--filter', type=int, default=128)
    args = parser.parse_args()

    train_kwargs = {'batch_size': args.batch_size, 'shuffle':True}

    transform=transforms.Compose([
        transforms.ToTensor(),
        Normalize()
        ])
    if args.cifar:
        dataset1 = datasets.CIFAR10('../data/', train=True, download=True,
                           transform=transform)
    else:
        dataset1 = datasets.MNIST('../data/', train=True, download=True,
                           transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)

    trainer = Trainer(args, train_loader)
    trainer.train()
    

if __name__ == '__main__':
    main()