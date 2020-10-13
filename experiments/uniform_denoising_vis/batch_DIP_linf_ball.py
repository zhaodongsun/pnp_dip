import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from matplotlib.pyplot import imread, imsave
from skimage.transform import resize
import time
import sys

sys.path.append('../')

from admm_utils import *
from torch import optim
from models import *
import math
import glob

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def run(f_name, specific_result_dir, noise_sigma, num_iter, rho, sigma_0, L):

    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    img = imread(f_name)[:,:,:3]
    if img.dtype == 'uint8':
        img = img.astype('float32') / 255  # scale to [0, 1]
    elif img.dtype == 'float32':
        img = img.astype('float32')
    else:
        raise TypeError()

    # img = np.clip(resize(img, (256, 256)), 0, 1)
    imsave(specific_result_dir + 'true.png', img)
    img = img.transpose((2, 0, 1))
    x_true = torch.from_numpy(img).unsqueeze(0).type(dtype)


    b = x_true
    b = b + noise_sigma * (2 * torch.rand(b.shape) - 1).type(dtype) # add uniform noise with mean 0
    b_clipped = torch.clamp(b, 0, 1)
    imsave(specific_result_dir + 'corrupted.png',
           b_clipped.reshape(1, 3, 512, 512)[0].permute((1, 2, 0)).cpu().numpy()) ## CHANGED

    def fn(x):
        return torch.norm(x - b) ** 2 / 2

    G = skip(3, 3,
            num_channels_down=[16, 32, 64, 128, 128],
            num_channels_up=[16, 32, 64, 128, 128],  # [16, 32, 64, 128, 128],
            num_channels_skip=[0, 0, 0, 0, 0],
            filter_size_up=3, filter_size_down=3, filter_skip_size=1,
            upsample_mode='nearest',  # downsample_mode='avg',
            need1x1_up=False,
            need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').type(dtype)
    
    z = torch.zeros((1, 3, 512, 512)).type(dtype).normal_()
    z_saved = z.detach().clone()
    noise = z.detach().clone()

    x = G(z).clone().detach()
    scaled_lambda_ = torch.zeros_like(x, requires_grad=False).type(dtype)

    x.requires_grad, z.requires_grad = False, True

    # since we use exact minimization over x, we don't need the grad of x
    z.requires_grad = False
    opt_z = optim.Adam(G.parameters(), lr=L)

    sigma_0 = torch.tensor(sigma_0).type(dtype)
    Gz = G(z)

    record = {"psnr_gt": [],
              "mse_gt": [],
              "total_loss": [],
              "prior_loss": [],
              "fidelity_loss": [],
              "cpu_time": [],
              }

    results = None
    for t in range(num_iter):
        # z = z_saved + noise.normal_() / 30
        # for x
        with torch.no_grad():
            x = linf_proj(Gz.detach() - scaled_lambda_, b, noise_sigma)

        # for z (GD)
        opt_z.zero_grad()
        Gz = G(z)
        loss_z = torch.norm(b- Gz) ** 2 / 2 + (rho / 2) * torch.norm(x - G(z) + scaled_lambda_) ** 2
        loss_z.backward()
        opt_z.step()

        # for dual var(lambda)
        with torch.no_grad():
            Gz = G(z).detach()
            x_Gz = x - Gz
            scaled_lambda_.add_(sigma_0 * x_Gz)

        if results is None:
            results = Gz.detach()
        else:
            results = results * 0.99 + Gz.detach() * 0.01

        psnr_gt = peak_signal_noise_ratio(x_true.cpu().numpy(), results.cpu().numpy())
        mse_gt = np.mean((x_true.cpu().numpy() - results.cpu().numpy()) ** 2)
        fidelity_loss = fn(torch.tensor(results).cuda()).detach()
        
        if (t + 1) % 500 == 0:
            imsave(specific_result_dir + 'iter%d_PSNR_%.2f.png' % (t, psnr_gt), results[0].cpu().numpy().transpose((1, 2, 0)))

        record["psnr_gt"].append(psnr_gt)
        record["mse_gt"].append(mse_gt)
        record["fidelity_loss"].append(fidelity_loss.item())
        record["cpu_time"].append(time.time())

        if (t + 1) % 10 == 0:
            print('Img %d Iteration %5d PSRN_gt: %.2f MSE_gt: %e' % (f_num, t + 1, psnr_gt, mse_gt))
    np.savez(specific_result_dir + 'record', **record)

# torch.manual_seed(500)

dataset_dir = '../../data_visual/'
results_dir = '../../data_visual/results/DIP_linf_ball/'

os.makedirs(results_dir)
f_name_list = glob.glob('../../data_visual/*.png')

for f_num, f_name in enumerate(f_name_list):

    specific_result_dir = results_dir+str(f_num)+'/'
    os.makedirs(specific_result_dir)
    run(f_name=f_name,
        specific_result_dir=specific_result_dir,
        noise_sigma=50/255,
        num_iter=5000,
        rho=1,
        sigma_0=1,
        L=0.001)
