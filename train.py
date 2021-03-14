import net as N
import utils as U
from torch.utils.data import DataLoader
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pylab as plt
import shutil
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256,
                    help='batchsize')
parser.add_argument('--b', type=int, default=1e-7,
                    help='value of the regularizer applied on the spatial gradients')
parser.add_argument('--c', type=int, default=4,
                    help='maximum displacement along consecutive pixels')
parser.add_argument('--img_channels', type=int, default=2,
                    help='channels of images')
parser.add_argument('--net_step', type=int, default=3,
                    help='number of steps for the multi-step network')
parser.add_argument('--save_folder', type=str, default='./models',
                    help='where to save the trained models')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of training epochs')

args = parser.parse_args()

transform=transforms.Compose([
    transforms.ToTensor()
    ])

dataset1 = datasets.MNIST('./data', train=True, download=True,
                   transform=transform)
dataset2 = datasets.MNIST('./data', train=False,
                   transform=transform)

mov_train_loader = torch.utils.data.DataLoader(dataset1, shuffle=True, batch_size=args.batch_size)
mov_test_loader = torch.utils.data.DataLoader(dataset2, shuffle=True, batch_size=args.batch_size)

ref_train_loader = torch.utils.data.DataLoader(dataset1, shuffle=True, batch_size=args.batch_size)
ref_test_loader = torch.utils.data.DataLoader(dataset2, shuffle=True, batch_size=args.batch_size)

model = U.to_cuda(N.DisplNet(args.img_channels,args.net_step))
criterion = U.to_cuda(torch.nn.MSELoss())

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = 10

if os.path.exists(args.save_folder):
    shutil.rmtree(args.save_folder)
os.mkdir(args.save_folder)


for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        iter_ = 0
        for batch_idx, (data, target) in enumerate(zip(mov_train_loader, ref_train_loader)):
            mov_data, ref_data = data, target
            mov_data[0] = U.to_cuda(mov_data[0])
            ref_data[0] = U.to_cuda(ref_data[0])
            optimizer.zero_grad()

            deformable, deformed, deformed_images = model(mov_data[0], ref_data[0])

            mse_loss = 0.0
            for s in range(0, args.net_step):
               mse_loss = mse_loss+criterion(deformed_images[s], ref_data[0])
            mse_loss = mse_loss/float(args.net_step)
            def_loss = args.b*torch.sum(torch.abs(deformable))
            loss = mse_loss + def_loss

            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            if iter_ % 100 == 0:
                print('Train (Epoch {}) [{}/{}]\tLoss: {:.4f}'.format(
                      epoch, batch_idx, int(len(mov_train_loader.dataset)/args.batch_size), loss.item()))
            iter_ += 1

        print('\nTrain set: Average loss: {:.4f}'.format(total_loss/float(batch_idx)))


        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(zip(mov_test_loader, ref_test_loader)):
                mov_data, ref_data = data, target
                mov_data[0] = U.to_cuda(mov_data[0])
                ref_data[0] = U.to_cuda(ref_data[0])

                deformable, deformed, deformed_images = model(mov_data[0], ref_data[0])

                mse_loss = 0.0
                for s in range(0, args.net_step):
                   mse_loss = mse_loss+criterion(deformed_images[s], ref_data[0])
                mse_loss = mse_loss/float(args.net_step)
                def_loss = args.b*torch.sum(torch.abs(deformable))

                test_loss += mse_loss + def_loss
        print('Test set: Average loss: {:.4f}\n'.format(
            test_loss/float(batch_idx)))

        torch.save(model.state_dict(), "{}/model_{}.pt".format(args.save_folder,epoch))

