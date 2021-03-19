import net as N
import utils as U
from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pylab as plt
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1,
                    help='batchsize')
parser.add_argument('--model_id', type=str, default='10',
                    help='specific model that you want to load')
parser.add_argument('--c', type=int, default=4,
                    help='maximum displacement along consecutive pixels')
parser.add_argument('--img_channels', type=int, default=2,
                    help='channels of images')
parser.add_argument('--net_step', type=int, default=3,
                    help='number of steps for the multi-step network')
parser.add_argument('--save_folder', type=str, default='./models',
                    help='where to save the trained models')

args = parser.parse_args()


model = U.to_cuda(N.DisplNet(args.img_channels,args.net_step))
model.load_state_dict(torch.load('./models/model_{}.pt'.format(args.model_id)))

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


mov_inputs, mov_classes = next(iter(mov_test_loader))  
mov_inp = U.to_cuda(mov_inputs[0]).unsqueeze(0) #torch.stack( (mov_inputs[0].unsqueeze(0),mov_inputs[0].unsqueeze(0),mov_inputs[0].unsqueeze(0)), 1).squeeze(2)
ref_inputs, ref_classes = next(iter(ref_test_loader))  
ref_inp = U.to_cuda(ref_inputs[0]).unsqueeze(0) #torch.stack( (ref_inputs[0].unsqueeze(0),ref_inputs[0].unsqueeze(0),ref_inputs[0].unsqueeze(0)), 1).squeeze(2)

print('Convert from digit ', mov_classes[0], 'to digit ', ref_classes[0])
deformable, sgrid, deformed_images = model(mov_inp, ref_inp)

mov_inp = mov_inp.permute(0,2,3,1)
ref_inp = ref_inp.permute(0,2,3,1)
deformed = deformed_images[-1].permute(0,2,3,1)

deformed, sgrid = deformed.data.cpu().numpy(), sgrid.data.cpu().numpy()

###PLOT####
xx, yy = np.meshgrid(range(mov_inp.shape[1]), range(mov_inp.shape[2]))
dx, dy = np.squeeze(sgrid[:,0,:,:]) + xx, np.squeeze(sgrid[:,1,:,:]) + yy


plt.figure(figsize=(10,4))
plt.subplot(1, 4, 1)
plt.imshow(np.squeeze(mov_inp.data.cpu().numpy()),cmap='gray')
plt.title('Moving Image')
plt.subplot(1, 4, 2)
plt.imshow(np.squeeze(deformed),cmap='gray')
plt.contour(dx, 50, alpha=0.5, linewidths=0.5)
plt.contour(dy, 50, alpha=0.5, linewidths=0.5)
plt.title('Forward Deformation \n applied on Moving Image')
plt.subplot(1, 4, 3)
plt.imshow(np.squeeze(ref_inp.data.cpu().numpy()),cmap='gray')
plt.title('Reference Image')
plt.tight_layout()
plt.savefig('example-2d-output.png')
