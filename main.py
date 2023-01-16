import argparse
import torch
from torchvision import datasets, transforms
from cdcgan_train import Trainer

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # General Training
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--niter', type=int, default=2000, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--num_ims', type=int, default=10)

    # Model Params
    parser.add_argument('--k', type=int, default=100, help='VAE Latent Space Size')
    args = parser.parse_args()

    train_kwargs = {'batch_size': args.num_ims*10, 'shuffle':True}
    test_kwargs = {'batch_size': args.test_batch_size}

    transform=transforms.Compose([
        transforms.ToTensor()
        ])
    dataset1 = datasets.MNIST('./', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    trainer = Trainer(args, train_loader, test_loader)
    trainer.train()
    

if __name__ == '__main__':
    main()