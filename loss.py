from pytorch_msssim import MS_SSIM, SSIM, gaussian_filter, _fspecial_gauss_1d
import torch.nn as nn
import lpips

def kld_loss(mu, log_var):
    klds = -0.5 * (1 + log_var - mu ** 2 - log_var.exp())
    return klds.sum(1).mean(0, True)

class ssim_loss():
    def __init__(self,):
        self.loss = SSIM(data_range=1.0, win_size = 5, size_average=True, channel=1)
    def forward(self, x, y):
        return 100 * (1 - self.loss(x, y))

class ms_ssim_loss():
    def __init__(self,):
        self.loss = MS_SSIM(data_range=1.0, win_size = 5, size_average=True, channel=1)
    def forward(self, x, y):
        return 100 * (1 - self.loss(x, y))
    
class l1_loss():
    def __init__(self,):
        self.loss = nn.L1Loss()
    def forward(self, x, y):
        return self.loss(x, y)
    
class l2_loss():
    def __init__(self,):
        self.loss = nn.MSELoss()
    def forward(self, x, y):
        return self.loss(x, y)
    
class l1_ssim_loss():
    def __init__(self, alpha = 0.84):
        self.l1 = nn.L1Loss(reduction='none')
        self.ssim = SSIM(data_range=1.0, win_size = 5, size_average=True, channel=1)
        self.win = _fspecial_gauss_1d(size = 5, sigma = 1.5)
        self.alpha = alpha
    def forward(self, x, y):
        return 100 * (self.alpha * self.ssim + (1-self.alpha) * self.l1 * gaussian_filter(self.win))
    
class perceptual_loss():
    def __init__(self,): 
        self.loss = lpips.LPIPS(net='alex')
    def forward(self, x, y):
        return self.loss.forward(x, y)