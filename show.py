import nets as N
import utils as U
from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pylab as plt
import numpy as np

model = U.to_cuda(N.MultiStepNet(6))
model.load_state_dict(torch.load('mnist_cnn.pt'))

transform=transforms.Compose([
        transforms.ToTensor()
        ])

dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)

mov_train_loader = torch.utils.data.DataLoader(dataset1, shuffle=True, batch_size=256)
mov_test_loader = torch.utils.data.DataLoader(dataset2, shuffle=True, batch_size=256)

ref_train_loader = torch.utils.data.DataLoader(dataset1, shuffle=True, batch_size=256)
ref_test_loader = torch.utils.data.DataLoader(dataset2, shuffle=True, batch_size=256)


mov_inputs, mov_classes = next(iter(mov_test_loader))  
mov_inp = torch.stack( (mov_inputs[0].unsqueeze(0),mov_inputs[0].unsqueeze(0),mov_inputs[0].unsqueeze(0)), 1).squeeze(2)
ref_inputs, ref_classes = next(iter(ref_test_loader))  
ref_inp = torch.stack( (ref_inputs[0].unsqueeze(0),ref_inputs[0].unsqueeze(0),ref_inputs[0].unsqueeze(0)), 1).squeeze(2)

print('mov', 'ref', mov_classes[0], ref_classes[0])

deformable, affine, reg_info = model(U.to_cuda(mov_inp), U.to_cuda(ref_inp))
deformed, ref_def, grid, grid_inv = reg_info

mov_inp = mov_inp.permute(0,2,3,1)
ref_inp = ref_inp.permute(0,2,3,1)
deformed = deformed.permute(0,2,3,1)
ref_def = ref_def.permute(0,2,3,1)

deformed, ref_def, grid, grid_inv = deformed.data.cpu().numpy(), ref_def.data.cpu().numpy(), grid.data.cpu().numpy(), grid_inv.data.cpu().numpy()


###PLOT####
xx, yy = np.meshgrid(range(mov_inp.shape[1]), range(mov_inp.shape[2]))
dx, dy = np.squeeze(grid[:,0,:,:]) + xx, np.squeeze(grid[:,1,:,:]) + yy
dxi, dyi = np.squeeze(grid_inv[:,0,:,:]) + xx, np.squeeze(grid_inv[:,1,:,:]) + yy


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
plt.imshow(np.squeeze(ref_def),cmap='gray')
plt.contour(dxi, 50, alpha=0.5, linewidths=0.5)
plt.contour(dyi, 50, alpha=0.5, linewidths=0.5)
plt.title('Inverse Deformation \n applied on Deformed Image')
plt.subplot(1, 4, 4)
plt.imshow(np.squeeze(ref_inp.data.cpu().numpy()),cmap='gray')
plt.title('Reference Image')
plt.tight_layout()
plt.savefig('example-2d-output.png')
