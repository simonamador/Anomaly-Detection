import torch
import torch.nn as nn
import torch.distributions as dist

# Author: @simonamador

# The following code builds an autoencoder model for unsupervised learning applications in MRI anomaly detection.
# The model can be build in 4 types:
# * Default: 2d Convolutions followed by a flattening and linear transformation
# * Residual: Includes residual blocks in between convolutions
# * Self-attention: Includes self-attention modules in between convolutions
# * Full: Includes both residual blocks and self-attention modules in between convolutions

# Basic class conducts a basic convolution-ReLU activation-batch normalization block.
# Inputs: input channels, output channels. Optional: kernel size, stride, transpose (true or false, default is false).
class Basic(nn.Module):
    def __init__(self, input, output, k_size=3,stride=1,padding=0,transpose=False):
        super(Basic, self).__init__()

        if transpose == False:
            self.conv_relu_norm = nn.Sequential(
                nn.Conv2d(input, output, k_size, padding=padding,stride=stride),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(output)
            )
        else:
            self.conv_relu_norm = nn.Sequential(
                nn.ConvTranspose2d(input, output, k_size, padding=padding,stride=stride),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(output)
            )
    def forward(self,x):
        return self.conv_relu_norm(x)

# ResDown class conducts the residual block for the encoder.
class ResDown(nn.Module):
     def __init__(self, input, output,k_size,stride):
         super(ResDown, self).__init__()
         
         self.basic1 = Basic(input,input,padding=1)
         self.basic2 = Basic(input,output,k_size=k_size,stride=stride)

         self.res = Basic(input,output,k_size=k_size,stride=stride)

     def forward(self,x):
         residual = self.res(x)
         x = self.basic2(self.basic1(x))
         return residual + x

# ResUp conducts the residual block for the decoder        
class ResUp(nn.Module):
    def __init__(self, input, output,k_size,stride):
        super(ResUp, self).__init__()
        
        self.basic1 = Basic(input,output,k_size=k_size, stride=stride, transpose=True)
        self.basic2 = Basic(output,output,padding=1)

        self.res = Basic(input,output,k_size=k_size, stride=stride, transpose=True)

    def forward(self,x):
        residual = self.res(x)
        x = self.basic2(self.basic1(x))
        return residual + x

'''class SA(nn.Module):
    dum

class RESA(nn.Module):
    dum
'''

# Encoder class builds encoder model depending on the model type.
# Inputs: H, y (x and y size of the MRI slice),z_dim (length of the output z-parameters), model (the model type)
class Encoder(nn.Module):
    def __init__(
            self, 
            h,
            w,
            z_dim,
            model='default'
        ):

        ch = 16
        k_size = 4
        stride = 2
        self.model = model

        super(Encoder,self).__init__()

        self.step0 = Basic(1,ch,k_size=k_size, stride=stride)

        if model == 'default' or model == 'bVAE':
            self.step1 = Basic(ch,ch * 2, k_size=k_size, stride=stride)
            self.step2 = Basic(ch * 2,ch * 4, k_size=k_size, stride=stride)
            self.step3 = Basic(ch * 4,ch * 8, k_size=k_size, stride=stride)
        elif model == 'residual':
            self.step1 = ResDown(ch,ch * 2, k_size=k_size, stride=stride)
            self.step2 = ResDown(ch * 2,ch * 4, k_size=k_size, stride=stride)
            self.step3 = ResDown(ch * 4,ch * 8, k_size=k_size, stride=stride)
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
        
        n_h = int(((h-k_size)/(stride**4)) - (k_size-1)/(stride**3) - (k_size-1)/(stride**2) - (k_size-1)/stride + 1)
        n_w = int(((w-k_size)/(stride**4)) - (k_size-1)/(stride**3) - (k_size-1)/(stride**2) - (k_size-1)/stride + 1)
        self.flat_n = n_h * n_w * ch * 8
        self.linear = nn.Linear(self.flat_n,z_dim)
    def forward(self,x):
        x = self.step0(x)
        x = self.step1(x)
        x = self.step2(x)
        x = self.step3(x)
        x = x.view(-1, self.flat_n)
        z_params = self.linear(x)

        mu, log_std = torch.chunk(z_params, 2, dim=1)
        std = torch.exp(log_std)

        z_dist = dist.Normal(mu, std)
        z_sample = z_dist.rsample()
        if self.model != 'bVAE':
            return z_sample
        else:
            return z_sample, mu, log_std
   
# Decoder class builds decoder model depending on the model type.
# Inputs: H, y (x and y size of the MRI slice),z_dim (length of the input z-vector), model (the model type) 

# Note: z_dim in Encoder is not the same as z_dim in Decoder, as the z_vector has half the size of the z_parameters.

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
        self.k_size = 4
        self.stride = 2
        self.hshape = int(((h-self.k_size)/(self.stride**4)) - (self.k_size-1)/(self.stride**3) - (self.k_size-1)/(self.stride**2) - (self.k_size-1)/self.stride + 1)
        self.wshape = int(((w-self.k_size)/(self.stride**4)) - (self.k_size-1)/(self.stride**3) - (self.k_size-1)/(self.stride**2) - (self.k_size-1)/self.stride + 1)

        self.z_develop = self.hshape * self.wshape * 8 * self.ch
        self.linear = nn.Linear(z_dim, self.z_develop)

        if model == 'default' or model == 'bVAE':
            self.step1 = Basic(self.ch* 8, self.ch * 4, k_size=self.k_size, stride=self.stride, transpose=True)
            self.step2 = Basic(self.ch * 4, self.ch * 2, k_size=self.k_size, stride=self.stride, transpose=True)
            self.step3 = Basic(self.ch * 2, self.ch, k_size=self.k_size, stride=self.stride, transpose=True)
        elif model == 'residual':
            self.step1 = ResUp(self.ch * 8, self.ch * 4, k_size=self.k_size, stride=self.stride)
            self.step2 = ResUp(self.ch * 4, self.ch * 2, k_size=self.k_size, stride=self.stride)
            self.step3 = ResUp(self.ch * 2, self.ch, k_size=self.k_size, stride=self.stride)
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
        
        self.step4 = Basic(self.ch, 1, k_size=self.k_size, stride=self.stride, transpose=True)
        self.activation = nn.ReLU()

    def forward(self,z):
        x = self.linear(z)
        x = x.view(-1, self.ch * 8, self.hshape, self.wshape)
        x = self.step1(x)
        x = self.step2(x)
        x = self.step3(x)
        x = self.step4(x)
        recon = self.activation(x)
        return recon
    

if __name__ == '__main__':
    from torchsummary import summary
    emodel = Encoder(158, 158, 1024,model='default')
    dmodel = Decoder(158, 158, 512,model='default')
    summary(emodel, (1, 158, 158), device='cpu')
    summary(dmodel, (1, 512), device='cpu')