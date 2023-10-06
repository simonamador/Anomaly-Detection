import torch
import torch.nn as nn
import torch.distributions as dist
import math

class Basic(nn.Module):
    def __init__(self, input, output,transpose=False):
        super(Basic, self).__init__()

        if transpose == False:
            self.conv_relu_norm = nn.Sequential(
                nn.Conv2d(input, output, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(output)
            )
        else:
            self.conv_relu_norm = nn.Sequential(
                nn.ConvTranspose2d(input, output, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(output)
            )
    def forward(self,x):
        return self.conv_relu_norm(x)

class ResDown(nn.Module):
     def __init__(self, input, output):
         super(ResDown, self).__init__()
         
         self.basic1 = Basic(input,input)
         self.basic2 = Basic(input,output)

         self.res = Basic(input,output)

     def forward(self,x):
         residual = self.res(x)
         x = self.basic2(self.basic1(x))
         return residual + x
        
class ResUp(nn.Module):
    def __init__(self, input, output):
        super(ResUp, self).__init__()
        
        self.basic1 = Basic(input,output,transpose=True)
        self.basic2 = Basic(output,output,transpose=False)

        self.res = Basic(input,output,transpose=True)

    def forward(self,x):
        residual = self.res(x)
        x = self.basic2(self.basic1(x))
        return residual + x

'''class SA(nn.Module):
    dum

class RESA(nn.Module):
    dum
'''

class Encoder(nn.Module):
    def __init__(
            self, 
            h,
            w,
            z_dim,
            model='default'
        ):

        ch = 16

        super(Encoder,self).__init__()

        self.step0 = Basic(1,ch)

        if model == 'default':
            self.step1 = Basic(ch,ch * 2)
            self.step2 = Basic(ch * 2,ch * 4)
            self.step3 = Basic(ch * 4,ch * 8)
            self.step4 = Basic(ch * 8,ch * 16)
        elif model == 'residual':
            self.step1 = ResDown(ch,ch * 2)
            self.step2 = ResDown(ch * 2,ch * 4)
            self.step3 = ResDown(ch * 4,ch * 8)
            self.step4 = ResDown(ch * 8,ch * 16)
            '''elif model == 'self-attention':
                self.step1 = SA(ch,ch * 2)
                self.step2 = SA(ch * 2,ch * 4)
                self.step3 = SA(ch * 4,ch * 8)
            elif model == 'full':
                self.step1 = RESA(ch,ch * 2)
                self.step2 = RESA(ch * 2,ch * 4)
            self.step3 = RESA(ch * 4,ch * 8)'''
        else:
            raise AttributeError("Model is not valid")
        
        self.flat_n = math.ceil(h/16) * math.ceil(w/16) * ch * 16
        self.linear = nn.Linear(self.flat_n,z_dim)
    def forward(self,x):
        x = self.step0(x)
        x = self.step1(x)
        x = self.step2(x)
        x = self.step3(x)
        x = self.step4(x)
        x = x.view(-1, self.flat_n)
        z_params = self.linear(x)

        mu, log_std = torch.chunk(z_params, 2, dim=1)
        std = torch.exp(log_std)
        z_dist = dist.Normal(mu, std)
        z_sample = z_dist.rsample()
        return z_sample
    
class Decoder(nn.Module):
    def __init__(
            self, 
            h, 
            w, 
            z_dim, 
            model='default'
            ):
        super(Decoder, self).__init__()

        self.ch = 16
        self.hshape = math.ceil(h / 16)
        self.wshape = math.ceil(w / 16)

        self.z_develop = self.hshape * self.wshape * self.ch * 16
        self.linear = nn.Linear(z_dim, self.z_develop)

        if model == 'default':
            self.step0 = Basic(self.ch* 16, self.ch * 8)
            self.step1 = Basic(self.ch* 8, self.ch * 4)
            self.step2 = Basic(self.ch * 4, self.ch * 2)
            self.step3 = Basic(self.ch * 2, self.ch)
        elif model == 'residual':
            self.step0 = Basic(self.ch * 16, self.ch * 8)
            self.step1 = ResUp(self.ch * 8, self.ch * 4)
            self.step2 = ResUp(self.ch * 4, self.ch * 2)
            self.step3 = ResUp(self.ch * 2, self.ch)
            '''elif model == 'self-attention':
                self.step1 = SA(ch,ch * 2)
                self.step2 = SA(ch * 2,ch * 4)
                self.step3 = SA(ch * 4,ch * 8)
            elif model == 'all':
                self.step1 = RESA(ch,ch * 2)
                self.step2 = RESA(ch * 2,ch * 4)
            self.step3 = RESA(ch * 4,ch * 8)'''
        else:
            raise AttributeError("Model is not valid")
        
        self.step4 = nn.Conv2d(self.ch, 1, 1)

    def forward(self,z):
        x = self.linear(z)
        x = x.view(-1, self.ch * 16, self.hshape, self.wshape)
        x = self.step0(x)
        x = self.step1(x)
        x = self.step2(x)
        x = self.step3(x)
        recon = self.step4(x)
        return recon
    
if __name__ == '__main__':
    from torchsummary import summary
    emodel = Encoder(160, 192, 512)
    dmodel = Decoder(160, 192, 512)
    summary(emodel, (1, 160, 192, 96), device='cpu')
    summary(dmodel, (1, 512), device='cpu')