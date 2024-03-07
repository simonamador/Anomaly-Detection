from models.anomaly import Anomaly
import models.aotgan.aotgan as inpainting
import copy
import torch.nn as nn
import torch

# Code inspired from https://github.com/ci-ber/PHANES/

class Framework(nn.Module):
    def __init__(self, n, z_dim, method, device, model, ga, ga_n, th=99, cGAN = False):
        super(Framework, self).__init__()
        self.z = z_dim
        self.ga = ga
        self.method = method
        self.th = th

        self.anomap = Anomaly(device)
        self.refineG = inpainting.InpaintGenerator().to(device)
        self.refineD = inpainting.Discriminator().to(device)

        if ga:
            from models.ga_vae import Encoder, Decoder
            self.encoder = Encoder(n, n, z_dim, method, model=model, ga_n=ga_n)
            self.decoder = Decoder(n, n, int(z_dim/2) + (ga_n if method in ['ordinal_encoding','one_hot_encoding', 'bpoe'] else 0))
        else:
            from models.vae import Encoder, Decoder
            self.encoder = Encoder(n, n, z_dim, model = model)
            self.decoder = Decoder(n, n, int(z_dim/2), model = model)

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
        
        anom1 = anom*saliency

        masks = self.anomap.mask_generation(anom1, th=self.th)

        x_ref = copy.deepcopy(x_im.detach())
        x_ref = (x_ref*(1-masks).float()) + masks

        y_ref = self.refineG(x_ref, masks, x_ga)
        # y_ref = self.refineG(x_ref, masks)
        y_ref = torch.clamp(y_ref, 0, 1)
        y_ref = self.anomap.zero_pad(y_ref, x_ref.shape[2])

        y_fin = x_im * (1-masks) + masks * y_ref

        if self.method == "beta-VAE":
            return y_fin, {"mu": mu, "log_var": log_var, "x_recon": x_recon, "anom": anom, "mask": masks, "saliency": saliency, "x_ref": x_ref, "y_ref": y_ref}
        else:
            return y_fin, {"x_recon": x_recon, "anom": anom, "mask": masks, "saliency": saliency, "x_ref": x_ref, "y_ref": y_ref, 'anom1':anom1}