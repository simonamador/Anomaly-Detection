import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from model import Encoder, Decoder
from train import img_dataset
import matplotlib.pyplot as plt

import os
import sys
import numpy as np

from collections import OrderedDict

# Author: @simonamador

batch = 64
view = 'L'
date = str(sys.argv[1])
anomaly = str(sys.argv[2])
model = 'default_AE_L2'
z = 512

path = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/'

print('-'*20)
print('Beginning validation:')
print('-'*20)

if anomaly == 'VM':
    images = os.listdir(path + 'Ventriculomegaly/recon_img/')
    extract_path = path + 'Ventriculomegaly/recon_img/'
elif anomaly == 'healthy':
    images = os.listdir(path + 'healthy_dataset/test/')
    extract_path = path + 'healthy_dataset/test/'

model_path = path + '/Results/HighZ' + view + '_' + model + '_b' +str(batch) + '_' + date + '/Saved_models/'

if view == 'L':
    w = 158
    h = 126
    ids = np.arange(start=40,stop=70)
elif view == 'A':
    w = 110
    h = 126
    ids = np.arange(start=64,stop=94)
else:
    w = 110
    h = 158
    ids = np.arange(start=48,stop=78)

encoder = Encoder(w,h,z*2)
decoder = Decoder(w,h,z)

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

for idx,image in enumerate(images):
    print('-'*20)
    print(f'Currently in image {idx+1} of {len(images)}')
    val_path = extract_path + image

    val_set = img_dataset(val_path,view)

    val_set = Subset(val_set,ids)

    loader = DataLoader(val_set, batch_size=1)

    loss = nn.MSELoss(reduction="none")
    for id, slice in enumerate(loader):
        if id == 20:
            z = encoder(slice)
            recon = decoder(z)

            error = loss(recon,slice)
            print(torch.mean(error))
            
            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2)
            ax1.imshow(slice.detach().cpu().numpy().squeeze(), cmap = "gray")
            ax1.axis("off")
            ax2.imshow(recon.detach().cpu().numpy().squeeze(), cmap = "gray")
            ax2.axis("off")
            ax3.hist(slice.detach().cpu().numpy().squeeze())
            ax4.hist(recon.detach().cpu().numpy().squeeze())
            ax5.imshow(error.detach().cpu().numpy().squeeze(), cmap = "hot")
            ax5.axis("off")
            ax6.imshow(error.detach().cpu().numpy().squeeze(), cmap = "hot")
            ax6.axis("off")
            plt.show()
    print('-'*20)