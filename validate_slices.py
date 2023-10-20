import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from model import Encoder, Decoder
from train import img_dataset

import os
import sys
import numpy as np

from collections import OrderedDict

# Author: @simonamador

path = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/'

views = ['L', 'A', 'S']
date = str(sys.argv[1])

print('-'*20)
print('Beginning validation:')
print('-'*20)

images = os.listdir(path + 'healthy_dataset/test/')

for view in views:
    loss = nn.MSELoss()
    model_path = path + '/Results/' + view + '_default_AE_L2_' + date + '/Saved_models/'

    if view == 'L':
        w = 158
        h = 126
        l = 110
    elif view == 'A':
        w = 110
        h = 126
        l = 158
    else:
        w = 110
        h = 158
        l = 126

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

    id_list = np.linspace(1,l,l)
    ids = "subject"
    for n in id_list:
        ids = ids+', p'+(str(int(n)))

    errors = []

    writer = open(path+'Results/Slice_validation_'+view+'_'+date+'.txt', 'w')
    writer.write(ids+"\n")

    for idx,image in enumerate(images):
        print('-'*20)
        print(f'Currently in image {idx+1} of {len(images)}')
        val_path = path + 'healthy_dataset/test' + '/' + image

        val_set = img_dataset(val_path,view)

        loader = DataLoader(val_set, batch_size=1)

        writer.write(str(int(idx+1)))

        for id, slice in enumerate(loader):
            z = encoder(slice)
            recon = decoder(z)

            error = loss(recon,slice)
            error = error.detach().numpy()
            writer.write(", "+str(error))

        print('-'*20)
        writer.write("\n")
writer.close()
print('-'*20)
print('Finished.')