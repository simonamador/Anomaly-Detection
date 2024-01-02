from models.anomaly import Anomaly
import models.aotgan.aotgan as inpainting
import copy
import torch.nn as nn
import torch

# Code inspired from https://github.com/ci-ber/PHANES/

class Framework(nn.Module):
    def __init__(self, n, z_dim, method, device, model, ga):
        super(Framework, self).__init__()
        self.z = z_dim
        self.ga = ga
        self.method = method

        self.anomap = Anomaly(device)
        self.refineG = inpainting.InpaintGenerator().to(device)
        self.refineD = inpainting.Discriminator().to(device)

        if ga:
            from models.ga_vae import Encoder, Decoder
            self.encoder = Encoder(n, n, z_dim, method)
            self.decoder = Decoder(n, n, z_dim)
        else:
            from models.vae import Encoder, Decoder
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

        x_recon = self.decoder(z)
        saliency, anom = self.anomap.anomaly(x_recon, x_im)
        
        anom = anom*(saliency-0.25)

        masks = self.anomap.mask_generation(anom, 99)

        x_ref = copy.deepcopy(x_im.detach())
        x_ref = (x_ref*(1-masks).float()) + masks

        y_ref = self.refineG(x_ref, masks)
        y_ref = torch.clamp(y_ref, 0, 1)
        y_ref = self.anomap.zero_pad(y_ref, x_ref.shape[2])

        y_fin = x_im * (1-masks) + masks * y_ref

        if self.method == "beta-VAE":
            return y_fin, {"mu": mu, "log_var": log_var, "x_recon": x_recon, "anom": anom, "mask": masks, "saliency": saliency, "x_ref": x_ref, "y_ref": y_ref}
        else:
            return y_fin, {"x_recon": x_recon, "anom": anom, "mask": masks, "saliency": saliency, "x_ref": x_ref, "y_ref": y_ref}