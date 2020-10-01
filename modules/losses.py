import torch
import math

import torch.nn as nn
from modules.vgg import VGG19

__all__ = ['kl_criterion', 'kl_bern_criterion']


def kl_criterion(mu, logvar):
    KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return KLD.mean()


def kl_bern_criterion(x):
    KLD = torch.mul(x, torch.log(x + 1e-20) - math.log(0.5)) + torch.mul(1 - x, torch.log(1 - x + 1e-20) - math.log(1 - 0.5))
    return KLD.mean()


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, use_gpu):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        if use_gpu:
            self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
