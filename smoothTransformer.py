from skimage import io
import numpy as np
import torch
import utils as U
import torch.nn.functional as F

def integralImage(x):
    x_s = torch.cumsum(x[:,0,:,:], dim=2)
    y_s = torch.cumsum(x[:,1,:,:], dim=1)
    out = torch.stack( [x_s, y_s], 1)
    return out-1

def repeat(x, n_repeats):
    rep = torch.ones(n_repeats).int().unsqueeze(0)
    x = x.unsqueeze(1).int()
    x = x*rep
    return x.flatten()

def logisticGrowth(x, maxgrad):
    out = maxgrad / (1 + (maxgrad-1)*torch.exp(-x))
    return out


def resample2D(im, sampling_grid, height, width, samples, channels):
    x_s, y_s = sampling_grid[:,0,:,:], sampling_grid[:,1,:,:]
    x = x_s.flatten()
    y = y_s.flatten()

    height_f = float(height)
    width_f = float(width)
    out_height = int(height_f)
    out_width = int(width_f)
    zero = int(0)
    max_y = int(height-1)
    max_x = int(width-1)

    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1
    x0 = torch.clamp(x0, zero, max_x)
    x1 = torch.clamp(x1, zero, max_x)
    y0 = torch.clamp(y0, zero, max_y)
    y1 = torch.clamp(y1, zero, max_y)

    dim2 = width
    dim1 = width*height
    base = repeat(torch.arange(samples)*dim1, out_height*out_width)
    base = U.to_cuda(base)

    base_y0 = base + y0*dim2
    base_y1 = base + y1*dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    im_flat = torch.reshape(im.permute(0,2,3,1), (-1,3))
    Ia = torch.index_select(im_flat,0,idx_a.long())
    Ib = torch.index_select(im_flat,0,idx_b.long())
    Ic = torch.index_select(im_flat,0,idx_c.long())
    Id = torch.index_select(im_flat,0,idx_d.long())

    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()

    wa = ((x1_f-x) * (y1_f-y)).unsqueeze(1)
    wb = ((x1_f-x) * (y-y0_f)).unsqueeze(1)
    wc = ((x-x0_f) * (y1_f-y)).unsqueeze(1)
    wd = ((x-x0_f) * (y-y0_f)).unsqueeze(1)


    s_out = torch.stack([wa*Ia, wb*Ib, wc*Ic, wd*Id], dim=0)
    output = s_out.sum(dim=0)

    output = output.view(samples, height, width, channels)
    output = output.permute(0,3,1,2)

    return output


def smoothTransformer2D(inp):

    if len(inp) == 2:
        [im, defgrad] = inp
        defgrad = logisticGrowth(defgrad, 2.0)
        sampling_grid = integralImage(defgrad)
    else:
        [im, defgrad, previous_sgrid] = inp
        defgrad = logisticGrowth(defgrad, 2.0)
        sampling_grid = previous_sgrid + defgrad


    base_grid = U.to_cuda(integralImage(torch.ones(defgrad.shape[0], defgrad.shape[1], defgrad.shape[2], defgrad.shape[3]))*0.5)

    samples = im.shape[0]
    channels = im.shape[1]
    height = im.shape[2]
    width = im.shape[3]

    mov_def = resample2D(im, sampling_grid_norm, height, width, samples, channels)

    return mov_def, sampling_grid

