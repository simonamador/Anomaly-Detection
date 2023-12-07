import monai.losses as losses 
from monai.networks.layers import gaussian_1d, separable_filtering
import torch.nn as nn
import torch

def kld_loss(mu, log_var):
    kld = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())
    return kld
    
l1_loss = nn.L1Loss()

l1_error = nn.L1Loss(reduction="none")

l2_loss = nn.MSELoss()

ssim_loss = losses.SSIMLoss(spatial_dims=2)

ms_ssim_loss = losses.MultiScaleLoss(loss = losses.SSIMLoss(spatial_dims = 2, win_size = 5),
                                     scales = [0.5, 1.0, 2.0, 4.0, 8.0])

perceptual_loss = losses.PerceptualLoss(spatial_dims = 2, network_type = 'radimagenet_resnet50')
    
# class l1_ssim_loss(nn.Module):
#     def __init__(self, alpha = 0.84):
#         self.ssim = ssim_loss
#         self.win = gaussian_1d
#         self.alpha = alpha
#     def forward(self, x, y):
#         A = self.alpha * self.ssim.forward(x, y)
#         B = (1-self.alpha) * torch.abs(
#             (separable_filtering(y, [self.win(1.5).to(y)] * (x.ndim -2))) - 
#             (separable_filtering(x, [self.win(1.5).to(y)] * (x.ndim -2))))
#         return 100 * (A + B)

# mixed_loss = l1_ssim_loss()