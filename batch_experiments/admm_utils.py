import torch
from skimage.restoration import denoise_tv_chambolle
import numpy as np
import math
import bm3d
import prox_tv
from matplotlib.pyplot import imread, imsave
from skimage.restoration import denoise_nl_means, estimate_sigma


def A_inpainting(num_ratio, img_dim):
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    num_measurements = np.round(img_dim * num_ratio).astype('int')
    chosen_ind = np.random.permutation(img_dim)[:num_measurements]
    A_diag= torch.zeros(img_dim).type(dtype)
    A_diag[chosen_ind] = 1
    # mask = torch.load('../../vase_mask128.pt').type(dtype)
    # mask = torch.ones(1, 3, 128, 128).type(dtype)
    # mask[:,:,50:70, 80:100] = 0
    # A_diag = mask.view(-1)
    # chosen_ind = (A_diag==1)
    def A(x):
        return x[chosen_ind]
    def At(b):
        b_ = torch.zeros(img_dim, device=b.device)
        b_[chosen_ind] = b
        return b_
    return A, At, A_diag


def l1_prox(input, lamda):
    t = torch.abs(input) - lamda
    t = t * (t > 0).type(t.dtype)
    return torch.sign(input) * t


def tv_prox(input, lamda):
    numpy_input = input.detach().cpu().numpy()
    numpy_input = numpy_input[0]
    result = []
    for i in numpy_input:
        result.append(prox_tv.tv1_2d(i, w=lamda, n_threads=16, method='pd')[np.newaxis])
    result = np.concatenate(result, 0)
    result = result[np.newaxis]
    return input.new(result)

def bm3d_prox(input, lamda):
    numpy_input = input.detach().cpu().numpy()
    numpy_input = numpy_input[0].transpose((1,2,0))
    result = bm3d.bm3d(numpy_input, lamda)
    result = result.transpose((2,0,1))[np.newaxis]
    return input.new(result)

def nlm_prox(input, lamda):
    numpy_input = input.detach().cpu().numpy()
    numpy_input = numpy_input[0].transpose((1,2,0))
    result = denoise_nl_means(numpy_input, multichannel=True, sigma=lamda, patch_distance=2, h=0.05)
    result = result.transpose((2,0,1))[np.newaxis]
    return input.new(result)

def linf_proj(input, center, bound):
    numpy_input = input.clone().detach()
    numpy_center = center.clone().detach()
    inp_minus_center = numpy_input - numpy_center

    no_change_part = ((inp_minus_center<=bound) * (inp_minus_center>=-bound))
    above_part = (inp_minus_center>bound)
    below_part = (inp_minus_center<-bound)

    upper_value = numpy_center + bound
    lower_value = numpy_center - bound

    result = no_change_part * numpy_input + above_part * upper_value + below_part * lower_value
    return result.detach()

# def projection_simplex_sort(v):
#     v = v.clone().detach()
#     v_flatten = v.reshape(-1)
#     n = v_flatten.shape[0]
#     u = torch.sort(v_flatten, descending=True)[0]
#     cssv = torch.cumsum(u, dim=0) - 1.
#     ind = torch.arange(n, device=u.device) + 1.
#     cond = (u - cssv / ind.float() > 0).long()
#     rho = torch.nonzero(cond).max() + 1
#     theta = cssv[rho - 1] / rho
#     w = torch.clamp(v - theta, min=0)
#     return w.detach()
    
# def proj_l1(v):
#     u = torch.abs(v)
#     if torch.sum(u) <= 1.:
#         return v
#     w = projection_simplex_sort(u)
#     w *= torch.sign(v)
#     return w

# def linf_prox(x, shrinkage_param):
#     return x - shrinkage_param * proj_l1(x / shrinkage_param)

# def linf_prox_x_b(x, b, shrinkage_param):
#     return linf_prox(x-b, shrinkage_param) + b
    
# if __name__ == '__main__':
#     test_v = torch.tensor([1,1])
#     print(linf_prox(test_v,1))
