import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from matplotlib.pyplot import imread, imsave
from skimage.transform import resize
import time
import sys
import glob

sys.path.append('../')

from admm_utils import *
from torch import optim
from models import *
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def run(f_name, specific_result_dir, noise_sigma, num_iter, GD_lr):
    img = imread(f_name)[:,:,:3]
    if img.dtype == 'uint8':
        img = img.astype('float32') / 255  # scale to [0, 1]
    elif img.dtype == 'float32':
        img = img.astype('float32')
    else:
        raise TypeError()
    img = np.clip(resize(img, (128, 128)), 0, 1)
    imsave(specific_result_dir + 'true.png', img)
    if len(img.shape) == 2:
        img = img[:,:,np.newaxis]
        num_channels = 1
    else:
        num_channels = 3

    img = img.transpose((2, 0, 1))
    x_true = torch.from_numpy(img).unsqueeze(0).type(dtype)

    # A = torch.zeros(num_measurements, x_true.numel()).normal_().type(dtype) / math.sqrt(num_measurements)
    A, At, _, down_img= A_superresolution(2, x_true.shape)
    b = A(x_true.reshape(-1,))
    b = torch.clamp(b + noise_sigma * torch.randn(b.shape).type(dtype), 0, 1)
    imsave(specific_result_dir+'corrupted.png', down_img(x_true).cpu().numpy()[0].transpose((1,2,0)))

    def fn(x): return torch.norm(A(x.reshape(-1)) - b) ** 2 / 2

    # G = skip(3, 3,
    #            num_channels_down = [16, 32, 64, 128, 128, 128],
    #            num_channels_up =   [16, 32, 64, 128, 128, 128],
    #            num_channels_skip =    [4, 4, 4, 4, 4, 4],
    #            filter_size_up = [7, 7, 5, 5, 3, 3],filter_size_down = [7, 7, 5, 5, 3, 3],  filter_skip_size=1,
    #            upsample_mode='bilinear', # downsample_mode='avg',
    #            need1x1_up=False,
    #            need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').type(dtype)
    # G = skip(3, 3,
    #          num_channels_down=[128, 128, 128, 128, 128],
    #          num_channels_up=[128, 128, 128, 128, 128],#[16, 32, 64, 128, 128],
    #          num_channels_skip=[4, 4, 4, 4, 4],
    #          filter_size_up=3, filter_size_down=3, filter_skip_size=1,
    #          upsample_mode='bilinear',  # downsample_mode='avg',
    #          need1x1_up=True,
    #          need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').type(dtype)
    G = get_net(3, 'skip', 'reflection',
              skip_n33d=128, 
              skip_n33u=128, 
              skip_n11=4, 
              num_scales=5,
              upsample_mode='bilinear').type(dtype)
    z = torch.zeros_like(x_true).type(dtype).normal_()

    z.requires_grad = False
    opt = optim.Adam(G.parameters(), lr=GD_lr)

    record = {"psnr_gt": [],
              "mse_gt": [],
              "total_loss": [],
              "prior_loss": [],
              "fidelity_loss": [],
              "cpu_time": [],
              }

    results = None
    for t in range(num_iter):
        x = G(z)
        fidelity_loss = fn(x)

        # prior_loss = (torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])))
        total_loss = fidelity_loss #+ 0.01 * prior_loss
        opt.zero_grad()
        total_loss.backward()
        opt.step()


        if results is None:
            results = x.detach().cpu().numpy()
        else:
            results = results * 0.99 + x.detach().cpu().numpy() * 0.01

        psnr_gt = peak_signal_noise_ratio(x_true.cpu().numpy(), results)
        mse_gt = np.mean((x_true.cpu().numpy() - results) ** 2)

        if (t + 1) % 250 == 0:
            if num_channels == 3:
                imsave(specific_result_dir + 'iter%d_PSNR_%.2f.png'%(t, psnr_gt), results[0].transpose((1,2,0)))
            else:
                imsave(specific_result_dir + 'iter%d_PSNR_%.2f.png'%(t, psnr_gt), results[0, 0], cmap='gray')


        record["psnr_gt"].append(psnr_gt)
        record["mse_gt"].append(mse_gt)
        record["fidelity_loss"].append(fidelity_loss.item())
        record["cpu_time"].append(time.time())
        if (t + 1) % 10 == 0:
            print('Img %d Iteration %5d   PSRN_gt: %.2f MSE_gt: %e' % (f_num, t + 1, psnr_gt, mse_gt))
    np.savez(specific_result_dir+'record', **record)

# torch.manual_seed(500)
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

dataset_dir = '../../data/'
results_dir = '../../data/results/DIP_sr/'
os.makedirs(results_dir)
f_name_list = glob.glob('../../data/*.jpg')

for f_num, f_name in enumerate(f_name_list):

    specific_result_dir = results_dir+str(f_num)+'/'
    os.makedirs(specific_result_dir)
    run(f_name = f_name,
        specific_result_dir = specific_result_dir,
        noise_sigma = 10 / 255,
        num_iter = 5000,
        GD_lr=0.01)
