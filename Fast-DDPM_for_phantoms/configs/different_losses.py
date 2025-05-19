import torch
import math
import time
from medpy import metric
import numpy as np
np.bool = np.bool_

def sg_noise_estimation_loss_with_kl(model, x_img, x_gt, t, e, b, keepdim=False):
    # Вычисление параметра a
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # Генерация зашумленного изображения x
    x = x_gt * a.sqrt() + e * (1.0 - a).sqrt()
    # Предсказание шума моделью
    output = model(torch.cat([x_img, x], dim=1), t.float())

    # MSE Loss
    mse_loss = F.mse_loss(output, e, reduction='none')
    if not keepdim:
        mse_loss = mse_loss.sum(dim=(1, 2, 3)).mean(dim=0)

    # Вычисление параметров для KL-дивергенции
    alpha_t = a
    alpha_t_prev = (1 - b).cumprod(dim=0).index_select(0, torch.maximum(t - 1, torch.zeros_like(t))).view(-1, 1, 1, 1)
    beta_t = b.index_select(0, t).view(-1, 1, 1, 1)  # Дисперсия на шаге t

    # Истинное апостериорное среднее
    mu_true = ((x - (1 - alpha_t).sqrt() * e) / alpha_t.sqrt()) * (1 - alpha_t_prev).sqrt() + \
              x_gt * (alpha_t_prev * beta_t).sqrt() / (1 - alpha_t)

    # Предсказанное среднее моделью
    mu_theta = ((x - (1 - alpha_t).sqrt() * output) / alpha_t.sqrt()) * (1 - alpha_t_prev).sqrt() + \
               x_gt * (alpha_t_prev * beta_t).sqrt() / (1 - alpha_t)

    # KL-дивергенция между двумя нормальными распределениями
    kl_div = 0.5 * ((mu_true - mu_theta).pow(2) / beta_t).sum(dim=(1, 2, 3))
    if not keepdim:
        kl_div = kl_div.mean(dim=0)

    # Итоговая функция потерь
    loss = mse_loss + kl_div
    return loss