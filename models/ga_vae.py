import torch
import torch.nn as nn
import torch.distributions as dist
from models.vae import Basic

# Author: @simonamador

# The following code builds an autoencoder model for unsupervised learning applications in MRI anomaly detection.

# Encoder class builds encoder model depending on the model type.
# Inputs: H, y (x and y size of the MRI slice),z_dim (length of the output z-parameters), model (the model type)
class Encoder(nn.Module):
    def __init__(
            self, 
            h,
            w,
            z_dim,
            model: str = 'default',
            method: str = 'multiplication'
        ):

        method_type = ['multiplication','concat']

        if method not in method_type:
            raise ValueError('Invalid method to include gestational age. Expected one of: %s' % method_type)

        ch = 16
        k_size = 4
        stride = 2
        self.method = method
        self.model = model

        # Reduce dimension size by 1 to account for the concatenation of GA
        if method == 'concat':
            z_dim = z_dim-1

        super(Encoder,self).__init__()

        self.step0 = Basic(1,ch,k_size=k_size, stride=stride)

        self.step1 = Basic(ch,ch * 2, k_size=k_size, stride=stride)
        self.step2 = Basic(ch * 2,ch * 4, k_size=k_size, stride=stride)
        self.step3 = Basic(ch * 4,ch * 8, k_size=k_size, stride=stride)

        n_h = int(((h-k_size)/(stride**4)) - (k_size-1)/(stride**3) - (k_size-1)/(stride**2) - (k_size-1)/stride + 1)
        n_w = int(((w-k_size)/(stride**4)) - (k_size-1)/(stride**3) - (k_size-1)/(stride**2) - (k_size-1)/stride + 1)
        self.flat_n = n_h * n_w * ch * 8
        self.linear = nn.Linear(self.flat_n,z_dim)

    def normalize(self,x):
        return x/40

    def forward(self,x,ga):
        ga = self.normalize(ga)

        x = self.step0(x)
        x = self.step1(x)
        x = self.step2(x)
        x = self.step3(x)
        x = x.view(-1, self.flat_n)
        z_params = self.linear(x)

        if self.method == 'concat':
            z_params = torch.cat((z_params,ga), 1)

        mu, log_std = torch.chunk(z_params, 2, dim=1)
        std = torch.exp(log_std)
        z_dist = dist.Normal(mu, std)

        z_sample = z_dist.rsample()

        if self.method == 'multiplication':
            z_sample = z_sample *ga

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
            ):
        super(Decoder, self).__init__()

        self.ch = 16
        self.k_size = 4
        self.stride = 2
        self.hshape = int(((h-self.k_size)/(self.stride**4)) - (self.k_size-1)/(self.stride**3) - (self.k_size-1)/(self.stride**2) - (self.k_size-1)/self.stride + 1)
        self.wshape = int(((w-self.k_size)/(self.stride**4)) - (self.k_size-1)/(self.stride**3) - (self.k_size-1)/(self.stride**2) - (self.k_size-1)/self.stride + 1)

        self.z_develop = self.hshape * self.wshape * 8 * self.ch
        self.linear = nn.Linear(z_dim, self.z_develop)
        self.step1 = Basic(self.ch* 8, self.ch * 4, k_size=self.k_size, stride=self.stride, transpose=True)
        self.step2 = Basic(self.ch * 4, self.ch * 2, k_size=self.k_size, stride=self.stride, transpose=True)
        self.step3 = Basic(self.ch * 2, self.ch, k_size=self.k_size, stride=self.stride, transpose=True)        
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