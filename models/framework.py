from anomaly import Anomaly
import aotgan.aotgan as inpainting
import torch.nn as nn
import torch

# Code inspired from https://github.com/ci-ber/PHANES/

class Framework(nn.Module):
    def __init__(self, n, z_dim, method, device, model='default', ga: bool=False):
        self.z = z_dim
        self.ga = ga
        self.method = method

        self.anomap = Anomaly(device)
        self.refineG = inpainting.InpaintGenerator().to(device)
        self.refineD = inpainting.InpaintDiscriminator().to(device)

        if ga:
            from ga_vae import Encoder, Decoder
            self.encoder = Encoder(n, n, z_dim, method)
            self.decoder = Decoder(n, n, z_dim)
        else:
            from vae import Encoder, Decoder
            self.encoder = Encoder(n, n, z_dim, model = model)
            self.decoder = Decoder(n, n, z_dim, model = model)

    def forward(self, x_im, x_ga = None):
        if self.ga:
            if x_ga is None:
                raise NameError('Missing GA as input.')
            else:
                if self.method == 'bVAE':
                    z, mu, log_var = self.encoder(x_im, x_ga)
                else: 
                    z = self.encoder(x_im, x_ga)
        else:
            if self.method == 'bVAE':
                    z, mu, log_var = self.encoder(x_im)
            else: 
                z = self.encoder(x_im)

        x_recon = self.decode(z)

        anom, saliency = self.anomap.anomaly(x_recon.detach(), x_im)
        anom = anom*(saliency-0.25)

        masks = self.anomap.mask_generation(anom, 95)

        x_ref = (x_im.detach()*(1-masks).float()) + masks

        y_ref = self.G(x_ref, masks)
        y_ref = torch.clamp(y_ref, 0, 1)

        anom_det = x_ref * y_ref

        if self.method == "beta-VAE":
            return anom_det, {"mu": mu, "log_var": log_var, "x_recon": x_recon, "anom": anom, "mask": masks, "saliency": saliency}
        else:
            return anom_det, {"x_recon": x_recon, "anom": anom, "mask": masks, "saliency": saliency}