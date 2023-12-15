import torch

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage import exposure

def mask_builder(input, recon, lpips_vgg, device,mask=False):
    
    def normalize(x, a):
        p = np.percentile(x, a)

        num = x-np.min(x)
        den = p-np.min(x)

        out = np.zeros((x.shape[0], x.shape[1]))
        x = np.divide(num, den, out=out, where=den!=0)

        return x.clip(0, 1)
    
    eq_input = exposure.equalize_adapthist(normalize(input,95))
    eq_recon = exposure.equalize_adapthist(normalize(recon,95))
    dif = abs(eq_input - eq_recon)

    norm95 = normalize(dif, 95)

    input = np.expand_dims(np.expand_dims(normalize(input,95), axis=0), axis=0)
    recon = np.expand_dims(np.expand_dims(normalize(recon,95), axis=0), axis=0)
    input = torch.from_numpy(input).type(torch.float).to(device)
    recon = torch.from_numpy(recon).type(torch.float).to(device)
    
    saliency = lpips_vgg(2*input-1, 2*recon-1, normalize=True).cpu().detach().numpy().squeeze()
    saliency = gaussian_filter(saliency, sigma=1.2)

    m = norm95*(saliency-0.25).clip(0,1)

    if mask == True:
        m95 = np.percentile(m, 95)
        m[m>m95] = 1
        m[m<1] = 0
        f_m = gaussian_filter(m, sigma=1.2)
        f_m[f_m>0.1] = 1
        f_m[f_m<1] = 0
        return f_m, saliency
    else:
        return m