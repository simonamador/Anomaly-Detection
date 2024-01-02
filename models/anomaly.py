# Code from https://github.com/ci-ber/PHANES/

import torch
import torch.nn as nn

from scipy.ndimage.filters import gaussian_filter

import numpy as np
import lpips
from skimage import exposure
import operator

class Anomaly:
    def __init__(self, device):
        self.ft_dif = lpips.LPIPS(pretrained=True, net='squeeze', eval_mode=True, spatial=True,
                                     lpips=True).to(device)
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
        for batch in range(x.shape[0]):
            x_b = self.normalize(x[batch].cpu().detach().numpy().squeeze(),99)
            y_b = self.normalize(y[batch].cpu().detach().numpy().squeeze(),99)
            
            eq_x = exposure.equalize_adapthist(x_b)
            eq_y = exposure.equalize_adapthist(y_b)            
            dif = abs(eq_x - eq_y)
            norm95 = self.normalize(dif, 95)
            anomalies.append(torch.Tensor(norm95).type(torch.float).to(self.device))

        return sal, torch.stack(anomalies)

    def saliency_map(self, x, y):
        saliencies = []
        for batch in range(x.shape[0]):
            saliency = self.ft_dif(2*x[batch:batch+1]-1, 2*y[batch:batch+1]-1)
            saliency = gaussian_filter(saliency.cpu().detach().numpy().squeeze(), sigma = 1.2)
            saliencies.append(saliency)
            
        return torch.tensor(np.asarray(saliencies)).type(torch.float).to(x.device)
    
    def mask_generation(self, x, th=95):
        masks = []
        for batch in range(x.shape[0]):
            # x_b = self.normalize(x[batch].cpu().detach().numpy(), 100)
            x_b = x[batch].cpu().detach().numpy()
            p_th = np.percentile(x_b, th)

            m  = x_b
            m[m>p_th] = 1
            m[m<1] = 0
            
            f_m = m
            f_m = gaussian_filter(np.asarray(f_m), sigma=1.2)
            
            f_m[f_m>0.1]=1
            f_m[f_m<1]=0
            f_m = np.expand_dims(f_m, 0)
            masks.append(f_m)
        return torch.tensor(np.asarray(masks)).type(torch.float).to(self.device)
    
    def zero_pad(self, x, n):
        results = []
        target = (n, n)
        for batch in range(x.shape[0]):
            img = x[batch][0].cpu().detach().numpy()
            if (img.shape > np.array(target)).any():
                target_shape2 = np.min([target, img.shape],axis=0)
                start = tuple(map(lambda a, da: a//2-da//2, img.shape, target_shape2))
                end = tuple(map(operator.add, start, target_shape2))
                slices = tuple(map(slice, start, end))
                img = img[tuple(slices)]
            offset = tuple(map(lambda a, da: a//2-da//2, target, img.shape))
            slices = [slice(offset[dim], offset[dim] + img.shape[dim]) for dim in range(img.ndim)]
            result = np.zeros(target)
            result[tuple(slices)] = img
            result = np.expand_dims(result, 0)
            results.append(result)
        return torch.tensor(np.asarray(results)).type(torch.float).to(self.device)
