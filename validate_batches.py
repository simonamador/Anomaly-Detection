import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from model import Encoder, Decoder
from train import img_dataset

import os
import sys
import numpy as np

from collections import OrderedDict

batches = [1,8,16,32,64]
view = 'L'
date = str(sys.argv[1])
model = 'default_AE_L2'

path = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/'

print('-'*20)
print('Beginning validation:')
print('-'*20)

images = os.listdir(path + 'healthy_dataset/test/')

writer = open(path+'Results/Batch_size_'+view+'_'+date+'.txt', 'w')

for batch in batches:
    loss = nn.MSELoss()
    model_path = path + '/Results/' + view + '_' + model + '_b' +str(batch) + '_' + date + '/Saved_models/'

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

    errors = []
    writer.write("Batch size "+str(batch)+", ")
    
    for idx,image in enumerate(images):
        print('-'*20)
        print(f'Currently in image {idx+1} of {len(images)}')
        val_path = path + 'healthy_dataset/test' + '/' + image

        val_set = img_dataset(val_path,view)

        loader = DataLoader(val_set, batch_size=batch)
        for id, slice in enumerate(loader):
            z = encoder(slice)
            recon = decoder(z)

            error = loss(recon,slice)
            errors.append(error.detach().numpy())

            recon = recon.detach().cpu().numpy().squeeze()
            input = slice.cpu().numpy().squeeze()

        error = np.mean(errors)
        print('-'*20)
        writer.write(", "+str(error))
    writer.write("\n")
writer.close()
print('-'*20)
print('Finished.')