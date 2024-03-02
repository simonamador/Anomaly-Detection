import torch
import torch.nn as nn
import torch.distributions as dist
from models.vae import Basic

# Author: @GuillermoTafoya

# The following code builds an autoencoder model for unsupervised learning applications in MRI anomaly detection.

# Encoder class builds encoder model depending on the model type.
# Inputs: H, y (x and y size of the MRI slice),z_dim (length of the output z-parameters), model (the model type)
class Encoder(nn.Module):
    def __init__(
            self, 
            h,
            w,
            z_dim,
            method,
            model: str = 'default',
            ga_n = 100
        ):

        method_type = ['multiplication','concat', 'concat_sample', 'ordinal_encoding', 'one_hot_encoding', 'bpoe']

        if method not in method_type:
            raise ValueError('Invalid method to include gestational age. Expected one of: %s' % method_type)

        ch = 16
        k_size = 4
        stride = 2
        self.method = method
        self.model = model
        self.size = ga_n

        # Reduce dimension size by 1 to account for the concatenation of GA
        if method == 'concat':
            z_dim = z_dim-1

        if method == 'concat_sample':
            z_dim = z_dim-2

        super(Encoder,self).__init__()

        self.step0 = Basic(1,ch,k_size=k_size, stride=stride)

        self.step1 = Basic(ch,ch * 2, k_size=k_size, stride=stride)
        self.step2 = Basic(ch * 2,ch * 4, k_size=k_size, stride=stride)
        self.step3 = Basic(ch * 4,ch * 8, k_size=k_size, stride=stride)

        n_h = int(((h-k_size)/(stride**4)) - (k_size-1)/(stride**3) - (k_size-1)/(stride**2) - (k_size-1)/stride + 1)
        n_w = int(((w-k_size)/(stride**4)) - (k_size-1)/(stride**3) - (k_size-1)/(stride**2) - (k_size-1)/stride + 1)
        self.flat_n = n_h * n_w * ch * 8
        self.linear = nn.Linear(self.flat_n,z_dim)

    def standardize(self,x):
        mean_of_TD_GA = 30.2988
        std_of_TD_GA = 4.0039
        return (x-mean_of_TD_GA)/std_of_TD_GA

    def normalize(self,x):
        return x/40
    
    def calculate_ga_index(self,ga):
        # Map GA to the nearest increment starting from 20 (assuming a range of 20-40 GA)
        increment = (40-20)/self.size
        ga_mapped = torch.round((ga - 20) / increment)
        return ga_mapped
    
    def create_one_hot_vector(self, gas):

        device = gas.device
        batch_size = gas.size(0)
        ga_indices = self.calculate_ga_index(gas)
        vectors = torch.zeros(batch_size, self.size, device=device)

        for i in range(batch_size):
            idx = ga_indices[i].long()
            if idx >= self.size:
                idx = self.size-1
            elif idx < 0:
                idx = 0
            vectors[i, idx] = 1  

        return vectors
   
    def create_ordinal_vector(self, gas):
        # https://link.springer.com/chapter/10.1007/978-3-030-32251-9_82
        
        device = gas.device
        batch_size = gas.size(0)
        ga_indices = self.calculate_ga_index(gas)
        vectors = torch.zeros(batch_size, self.size, device=device)

        for i in range(batch_size):
            idx = ga_indices[i].long()
            if idx > self.size:
                idx = self.size
            elif idx < 0:
                idx = 1
            vectors[i, :idx] = 1  

        return vectors
    
    def create_bi_partitioned_ordinal_vector(self, gas):
        # Adjusting the threshold for the nearest 0.1 increment
        threshold_index = self.size//2
        device = gas.device
        batch_size = gas.size(0)
        ga_indices = self.calculate_ga_index(gas)
        vectors = torch.full((batch_size, self.size), -1, device=device)  # Default fill with -1

        for i in range(batch_size):
            idx = ga_indices[i].long()
            if idx > self.size:
                idx = self.size
            elif idx < 0:
                idx = 1
            

            if idx >= threshold_index:  # GA >= 30
                new_idx = (idx-threshold_index)*2
                vectors[i, :new_idx] = 1  # First 100 elements to 1 (up to GA == 30)
                vectors[i, new_idx:] = 0  # The rest to 0
            else:  # GA < 30
                new_idx = idx*2
                vectors[i, :new_idx] = 0  # First 100 elements to 0
                # The rest are already set to -1

        return vectors

    def forward(self,x,ga): 
        if self.method == 'ordinal_encoding':
            # Create the ordinal encoding vector for GA
            ga = self.create_ordinal_vector(ga)
        elif self.method == 'one_hot_encoding':
            ga = self.create_one_hot_vector(ga)
        elif self.method == 'bpoe':
            ga = self.create_bi_partitioned_ordinal_vector(ga)
        else:
            #ga = self.standardize(ga)
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

        # ------ Instead of concatenating to the params, perform the operation on the sample ------
        if self.method in ['concat_sample', 'ordinal_encoding', 'one_hot_encoding', 'bpoe']:
            z_sample = torch.cat((z_sample,ga), 1)

        if self.method == 'multiplication':
            z_sample = z_sample * ga

        if self.model == 'bVAE':
            return z_sample, mu, log_std
        
        return z_sample
   
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
    

class Transformer(nn.Module):
    