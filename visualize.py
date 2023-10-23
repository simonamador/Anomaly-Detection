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
view = 'A'
date = str(sys.argv[1])
anomaly = str(sys.argv[2])
model = 'default_AE_L2'

path = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/'

print('-'*20)
print('Beginning validation:')
print('-'*20)

if anomaly == 'VM':
    images = os.listdir(path + 'Ventriculomegaly/recon_img/')

model_path = path + '/Results/' + view + '_' + model + '_b' +str(batch) + '_' + date + '/Saved_models/'

if view == 'L':
    w = 158
    h = 126
    ids = np.arange(start=12,stop=99)
elif view == 'A':
    w = 110
    h = 126
    ids = np.arange(start=16,stop=143)
else:
    w = 110
    h = 158
    ids = np.arange(start=12,stop=115)

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

for idx,image in enumerate(images):
    print('-'*20)
    print(f'Currently in image {idx+1} of {len(images)}')
    val_path = path + 'Ventriculomegaly/recon_img/' + image

    val_set = img_dataset(val_path,view)

    val_set = Subset(val_set,ids)

    loader = DataLoader(val_set, batch_size=1)
    for id, slice in enumerate(loader):
        z = encoder(slice)
        recon = decoder(z)

        recon = recon.detach().cpu().numpy().squeeze()
        input = slice.cpu().numpy().squeeze()
        error = (recon-input)**2
        if id == 60:
            fig, (ax1, ax2, ax3) = plt.subplots(3,1)
            ax1.imshow(input, cmap = "gray")
            ax2.imshow(recon, cmap = "gray")
            ax3.imshow(error, cmap = "hot")
            plt.show()
    print('-'*20)