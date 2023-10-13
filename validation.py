import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from model import Encoder, Decoder

import os
import argparse
import numpy as np
import nibabel as nib

from collections import OrderedDict

import matplotlib.pyplot as plt

class val_dataset(Dataset):
    def __init__(self, root_dir, view):
        self.root_dir = root_dir
        self.view = view

    def __len__(self):
        if self.view == 'L':
            size = 110
        elif self.view == 'A':
            size = 158
        else:
            size = 126
        return size
    
    def __getitem__(self, idx):
        raw = nib.load(self.root_dir).get_fdata()
        if self.view == 'L':
            n_img = raw[idx,:158,:]
        elif self.view == 'A':
            n_img = raw[:110,idx,:]
        else:
            n_img = raw[:110,:158,idx]
        
        num = n_img-np.min(n_img)
        den = np.max(n_img)-np.min(n_img)
        out = np.zeros((n_img.shape[0], n_img.shape[1]))
    
        n_img = np.divide(num, den, out=out, where=den!=0)

        n_img *= 255
        n_img = np.expand_dims(n_img,axis=0)
        n_img = torch.from_numpy(n_img).type(torch.float)

        return n_img

parser = argparse.ArgumentParser()

parser.add_argument('--view',
        dest='view',
        choices=['L','A','S'],
        required=True,
        help='''
        The view of the image input for the model. Options:
        "L" Left view
        "A" Axial view
        "S" Sagittal view''')

parser.add_argument('--model',
        dest='model',
        default='default_AE_SSIM',
        choices=['default_AE_L2','default_AE_SSIM','residual_AE_L2','residual_AE_SSIM'],
        required=False,
        help='''
        Model name. Ex. default_AE_L2
        ''')

parser.add_argument('--date',
        dest='date',
        default='20231011',
        required=False,
        help='''
        Date of model training.
        ''')

parser.add_argument('--ga',
        dest='ga',
        type = int,
        default=31,
        choices = range(22,39),
        required=False,
        help='''
        GA to validate.
        ''')

args = parser.parse_args()

print(args)

view = args.view
model = args.model
date = args.date
ga = args.ga

path = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/'
model_path = path + '/Results/' + view + '_' + model + '_' + date + '/Saved_models/'

if view == 'L':
    w = 158
    h = 126
elif view == 'A':
    w = 110
    h = 126
else:
    w = 110
    h = 158

encoder = Encoder(w,h,512)
decoder = Decoder(w,h,256)

cpe = torch.load(model_path+'encoder_best.pth')
cpd = torch.load(model_path+'decoder_best.pth')

cpe_new = OrderedDict()
cpd_new = OrderedDict()

for k, v in cpe['encoder'].items():
    name = k[7:]
    cpe_new[name] = v

for k, v in cpd['decoder'].items():
    name = k[7:]
    cpd_new[name] = v

encoder.load_state_dict(cpe_new)
decoder.load_state_dict(cpd_new)

print('Encoder and Decoder loaded.')

print('-'*20)
print('Beginning validation:')
print('-'*20)


images = os.listdir(path + 'healthy_dataset/recon-by-GA/GA_' + str(ga))
loss = nn.MSELoss()
errors = []

for idx,image in enumerate(images):
    print('-'*20)
    print(f'Currently in image {idx} of {len(images)}')
    val_path = path + 'healthy_dataset/recon-by-GA/GA_' + str(ga) + '/' + image

    val_set = val_dataset(val_path,view)

    loader = DataLoader(val_set, batch_size=1)
    for id, slice in enumerate(loader):
        z = encoder(slice)
        recon = decoder(z)

        error = loss(recon,slice)
        errors.append(error.detach().numpy())

        recon = recon.detach().cpu().numpy().squeeze()
        input = slice.cpu().numpy().squeeze()
        
        if (idx == 0) & (id == 70):
            fig = plt.figure()
            fig.add_subplot(3,1,1)
            plt.imshow(input, cmap='gray')
            plt.axis('off')
            fig.add_subplot(3,1,2)
            plt.imshow(recon, cmap='gray')
            plt.axis('off')
            fig.add_subplot(3,1,3)
            plt.imshow((recon-input)**2, cmap='hot')
            plt.axis('off')
            plt.show()

error = np.mean(errors)
print('-'*20)
print(f'Mean error: {error}')