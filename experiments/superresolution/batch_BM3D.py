import os
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

def run(f_name, specific_result_dir, noise_sigma, num_iter, rho, sigma_0, shrinkage_param, prior, num_ratio):
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    img = imread(f_name)
    if img.dtype == 'uint8':
        img = img.astype('float32') / 255  # scale to [0, 1]
    elif img.dtype == 'float32':
        img = img.astype('float32')
    else:
        raise TypeError()
    img = np.clip(resize(img, (128, 128)), 0, 1)
    imsave(specific_result_dir + 'true.png', img)
    img = img.transpose((2, 0, 1))
    x_true = torch.from_numpy(img).unsqueeze(0).type(dtype)

    A, At, A_diag, down_img= A_superresolution(2, x_true.shape)

    b = A(x_true.reshape(-1, ))
    b = torch.clamp(b + noise_sigma * torch.randn(b.shape).type(dtype), 0, 1)
    imsave(specific_result_dir+'corrupted.png', down_img(x_true).cpu().numpy()[0].transpose((1,2,0)))

    def fn(x):
        return torch.norm(A(x.reshape(-1)) - b) ** 2 / 2

    v = torch.zeros_like(x_true).type(dtype).uniform_()
    x = v.clone().detach()
    scaled_lambda_ = torch.zeros_like(x, requires_grad=False).type(dtype)

    sigma_0 = torch.tensor(sigma_0).type(dtype)
    prox_op = eval(prior)

    inv = A_diag + rho
    inv = 1 / inv
    At_b = At(b)

    record = {"psnr_gt": [],
              "mse_gt": [],
              "total_loss": [],
              "prior_loss": [],
              "fidelity_loss": [],
              "cpu_time": [],
              }

    for t in range(num_iter):
        # for x (exact min)
        x = inv * (At_b - rho * (scaled_lambda_ - v).view(-1))
        x = x.view(*v.shape)

        # for v (exact min with prox op)
        if shrinkage_param > 0:
            v = prox_op(x + scaled_lambda_, shrinkage_param / rho)
        else:
            v = x + scaled_lambda_

        # for dual var(lambda)
        scaled_lambda_.add_(sigma_0 * rho * (x - v))

        results = x.detach()
        results = torch.clamp(results, 0, 1)
        if (t + 1) % 250 == 0:
            imsave(specific_result_dir + 'iter%d_PSNR_%.2f.png' % (t, psnr_gt), results[0].cpu().numpy().transpose((1, 2, 0)))

        psnr_gt = peak_signal_noise_ratio(x_true.cpu().numpy(), results.detach().cpu().numpy())
        mse_gt = np.mean((x_true.cpu().numpy() - results.detach().cpu().numpy()) ** 2)
        fidelity_loss = fn(results).detach()

        record["psnr_gt"].append(psnr_gt)
        record["mse_gt"].append(mse_gt)
        record["fidelity_loss"].append(fidelity_loss.item())
        record["cpu_time"].append(time.time())

        if (t + 1) % 10 == 0:
            print('Img %d Iteration %5d PSRN_gt: %.2f MSE_gt: %e' % (f_num, t + 1, psnr_gt, mse_gt))
    np.savez(specific_result_dir + 'record', **record)

# torch.manual_seed(500)
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

dataset_dir = '../../data/'
results_dir = '../../data/results/bm3d_sr/'

os.makedirs(results_dir)
f_name_list = glob.glob('../../data/*.jpg')

for f_num, f_name in enumerate(f_name_list):

    specific_result_dir = results_dir+str(f_num)+'/'
    os.makedirs(specific_result_dir)
    run(f_name=f_name,
        specific_result_dir=specific_result_dir,
        noise_sigma=10/255,
        num_iter=5000,
        rho=1,
        sigma_0=1,
        shrinkage_param=0.05,
        prior='bm3d_prox',
        num_ratio=0.5)
