import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn

from model import Encoder, Decoder
from process import perceptual_loss, threshold, img_dataset

from train import img_dataset
import matplotlib.pyplot as plt

import os
import argparse
import numpy as np

from collections import OrderedDict

# Author: @simonamador

parser = argparse.ArgumentParser()

parser.add_argument('--model_type',
    dest='type',
    choices=['default', 'residual', 'bVAE', 'self-attention','full'],
    required=False,
    default='default',
    help='''
    Type of model to train. Available options:
    "defalut" Default VAE using convolution blocks
    "residual: VAE which adds residual blocks between convolutions''')

parser.add_argument('--model_view',
    dest='view',
    choices=['L', 'A', 'S'],
    required=False,
    default='L',
    help='''
    The view of the image input for the model. Options:
    "L" Left view
    "A" Axial view
    "S" Sagittal view''')

parser.add_argument('--loss',
    dest='loss',
    default='L2',
    choices=['L2', 'SSIM', 'MS_SSIM', 'Mixed'],
    required=False,
    help='''
    Loss function:
    L2 = Mean square error.
    SSIM = Structural similarity index.
    ''')

parser.add_argument('--batch',
    dest='batch',
    type=int,
    default=64,
    choices=range(1, 512),
    required=False,
    help='''
    Number of batch size.
    ''')

parser.add_argument('--date',
    dest='date',
    default='20231102',
    required=False,
    help='''
    Date of model training.
    ''')

parser.add_argument('--anomaly',
    dest='anomaly',
    default='healthy',
    choices = ['healthy', 'vm'],
    required=False,
    help='''
    Extra model name info.
    ''')

parser.add_argument('--extra',
    dest='extra',
    default=False,
    required=False,
    help='''
    Extra model name info.
    ''')

parser.add_argument('--z_dim',
    dest='z',
    type=int,
    default=256,
    required=False,
    help='''
    z dimension.
    ''')


args = parser.parse_args()

print(args)
print('-'*25)


model = args.type
view = args.view
batch = args.batch
loss_type = args.loss
date = args.date
extra = args.extra
anomaly = args.anomaly
z_dim = args.z


path = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/'

print('-'*20)
print('Beginning anomaly maps generation:')
print('-'*20)

if extra:
    model_name = extra + view + '_' + model + '_AE_' + loss_type + '_b' +str(batch) + '_' + date
else:
    model_name = view + '_' + model + '_AE_' + loss_type + '_b' +str(batch) + '_' + date

model_path = path + '/Results/' + model_name + '/Saved_models/'
h = w = 158

#158, 126, 110
if view == 'L':
    ids = np.arange(start=40,stop=70)
elif view == 'A':
    ids = np.arange(start=64,stop=94)
else:
    ids = np.arange(start=48,stop=78)

encoder = Encoder(w,h,z_dim*2)
decoder = Decoder(w,h,z_dim)

cpe = torch.load(model_path+'encoder_best.pth', map_location=torch.device('cpu'))
cpd = torch.load(model_path+'decoder_best.pth', map_location=torch.device('cpu'))

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

if anomaly == 'vm':
    img_path = 'Ventriculomegaly/recon_img/'
    images = os.listdir(path + img_path)
else:
    img_path = 'healthy_dataset/test/'
    images = os.listdir(path + img_path)

loss = nn.MSELoss(reduction = 'none')

save_path = path + 'Results/Visual_' + anomaly + '/Per_loss_Otsu_' + model_name + '/'
if not os.path.exists(save_path):
        os.mkdir(save_path)

for idx,image in enumerate(images):
    print('-'*20)
    print(f'Currently in image {idx+1} of {len(images)}')
    val_path = path + img_path + image

    val_set = img_dataset(val_path,view)

    val_set = Subset(val_set,ids)

    loader = DataLoader(val_set, batch_size=1)

    for id, slice in enumerate(loader):
        z = encoder(slice)
        recon = decoder(z)
        error = perceptual_loss(recon,slice, model_path)
        n_error, th = threshold(error.detach().cpu().numpy().squeeze())
        
        fig, axs = plt.subplots(2,3)
        axs[0,0].imshow(slice.detach().cpu().numpy().squeeze(), cmap = "gray")
        axs[0,0].axis("off")
        axs[1,0].imshow(recon.detach().cpu().numpy().squeeze(), cmap = "gray")
        axs[1,0].axis("off")
        axs[0,1].imshow(n_error, cmap = "hot")
        axs[0,1].axis("off")
        axs[1,1].hist(n_error)
        axs[0,2].imshow(th, cmap = "hot")
        axs[0,2].axis("off")
        axs[1,2].hist(th)

        plt.savefig(save_path+image[:-4]+'_sl'+str(id)+'.png')
        plt.close()