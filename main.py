import argparse
import torch
from torchvision import datasets, transforms
from cdcgan_train import Trainer

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # General Training
    parser.add_argument('--batch-size', type=int, default= 100, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--niter', type=int, default=20000, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--lrIms', type=float, default=5e-3, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--num_ims', type=int, default=10)
    parser.add_argument('--cifar', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--init_ims', type=bool, default=False)
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--norm_flow', type=bool, default=False)
    parser.add_argument('--biggan', type=bool, default=False)
    parser.add_argument('--iterIms', type=int, default=1)
    parser.add_argument('--iterD', type=int, default=1)
    parser.add_argument('--class_wise', type=bool, default=False)
    parser.add_argument('--fid', type=bool, default=False)
    parser.add_argument('--corr', type=bool, default=False)
    parser.add_argument('--corr_coef', type=float, default=1)
    parser.add_argument('--repulsion', type=bool, default=False)
    parser.add_argument('--var', type=bool, default=False)
    parser.add_argument('--lr_schedule', type=bool, default=False)
    parser.add_argument('--spectral_norm', type=bool, default=False)

    # Model Params
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--filter', type=int, default=512)
    args = parser.parse_args()

    train_kwargs = {'batch_size': args.batch_size, 'shuffle':True}

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), 
            (0.5, 0.5, 0.5))
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