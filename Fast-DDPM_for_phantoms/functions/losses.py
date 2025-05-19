import torch
import math
import time
from medpy import metric
import numpy as np
np.bool = np.bool_


def calculate_psnr(img1, img2):
    # img1: img
    # img2: gt
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mse = np.mean((img1 - img2)**2)
    psnr = 20 * math.log10(255.0 / math.sqrt(mse))

    return psnr


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    # a: a_T in DDIM
    # 1-a: 1-a_T in DDIM 
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # X_T
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)



def sr_noise_estimation_loss(model,
                          x_bw: torch.Tensor,
                          x_md: torch.Tensor,
                          x_fw: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    # a: a_T in DDIM
    # 1-a: 1-a_T in DDIM 
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # X_T
    x = x_md * a.sqrt() + e * (1.0 - a).sqrt()

    output = model(torch.cat([x_bw, x_fw, x], dim=1), t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)



def sg_noise_estimation_loss(model,
                          x_img: torch.Tensor,
                          x_gt: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    # a: a_T in DDIM
    # 1-a: 1-a_T in DDIM 
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # X_T
    x = x_gt * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(torch.cat([x_img, x], dim=1), t.float())

    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def sg_combo_mse_kl_loss(model,
                          x_img: torch.Tensor,
                          x_gt: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    # a: a_T in DDIM
    # 1-a: 1-a_T in DDIM
    #print('x_img', x_img.shape)
    #print ('t', t.shape)
    #print('e', e.shape)
    #print('b', b.shape)
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    #a_t = (1-b).cumprod(dim=0).index_select(0, t)
    #print('a', a.shape)
    #print('a_t', a_t.shape)
    # X_T
    x = x_gt * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(torch.cat([x_img, x], dim=1), t.float())
    #print('output', output.shape)

    mse =  (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

    var_e = e.var(dim=(1, 2, 3), unbiased=False)
    var_output = output.var(dim=(1, 2, 3), unbiased=False)

    kl_div = 0.5 * (var_output / var_e - 1 - torch.log(var_output / var_e)).sum(dim=0).mean(dim=0)

    total_loss = mse + 0.2 * kl_div

    return total_loss


loss_registry = {
    'simple': noise_estimation_loss,
    'sr': sr_noise_estimation_loss,
    'sg': sg_combo_mse_kl_loss
}