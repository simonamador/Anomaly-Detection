# Code from https://github.com/ci-ber/PHANES/

import torch
import torch.nn as nn

from scipy.ndimage.filters import gaussian_filter

import numpy as np
import lpips
import exposure

class Anomaly:
    def __init__(self, device):
        self.ft_dif = lpips.LPIPS()
        self.device = device

    def normalize(self, x, a):
        p_a = np.percentile(x, a)
        num = x - np.min(x)
        den = p_a - np.min(x)
        
        out = np.zeros((x.shape[0], x.shape[1]))
        n_x = np.divide(num, den, out=out, where=den!=0)
        return n_x.clip(0,1)

    def anomaly(self, x, y):
        sal = self.saliency_map(x, y)
        anomalies = []
        for batch in range(x.size(0)):
            x_b = self.normalize(x[batch].cpu().detach().numpy().squeeze(),99)
            y_b = self.normalize(y[batch].cpu().detach().numpy().squeeze(),99)
            
            eq_x = exposure.equalize_adapthist(x_b)
            eq_y = exposure.equalize_adapthist(y_b)            
            dif = abs(eq_x - eq_y)
            norm95 = self.normalize(dif, 95)
            anomalies.append(torch.from_numpy(norm95).type(torch.float))

        return sal, torch.cat(anomalies,0)

    def saliency_map(self, x, y):
        saliencies = []
        for batch in range(x.shape(0)):
            saliency = self.ft_dif(2*x[batch]-1, 2*y[batch]-1)
            saliency = gaussian_filter(saliency, sigma = 1.2)
            saliencies.append(saliency)
        return torch.cat(saliencies, 0)
    
    def mask_generation(self, x, th=95):
        for batch in range(x.shape[0]):
            x_b = self.normalize(x[batch][0].cpu().detach().numpy())
            p_th = np.perecentile(x_b, th)

            m  = x_b
            m[m>p_th] = 1
            m[m<1] = 0
            
            f_m = m
            f_m = gaussian_filter(np.asarray(f_m), sigma=1.2)
            f_m = torch.Tensor(f_m).to(x.device)
            
            f_m[f_m>0.1]=1
            f_m[f_m<1]=0

            f_m.append(np.expand_dims(f_m, 0))
        return f_m