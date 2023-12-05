import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn

from parser_module import settings_parser
from model import Encoder, Decoder
from train import img_dataset
from post_processing import threshold
from pytorch_msssim import ssim

import os
import numpy as np

from collections import OrderedDict

# Author: @simonamador

parser = settings_parser()
args = parser.parse_args()

model = args.type
view = args.view
batch = args.batch
loss_type = args.loss
date = args.date
extra = args.extra
anomaly = args.anomaly
z_dim = args.z
path = args.path

print('-'*20)
print('Beginning validation:')
print('-'*20)

if extra:
    model_name = extra + view + '_' + model + '_AE_' + loss_type + '_b' +str(batch) + '_' + date
else:
    model_name = view + '_' + model + '_AE_' + loss_type + '_b' +str(batch) + '_' + date

model_path = path + '/Results/' + model_name + '/Saved_models/'

h = w = 158

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

mae = nn.L1Loss(reduction = 'none')
mse = nn.MSELoss(reduction = 'none')

writer = open(path+'Results/Validations/' + anomaly + '_' + model_name +'.txt', 'w')
writer.write('Case_ID, Slide_ID, mae, mse, ssim'+'\n')

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

        MSE = np.mean(threshold(mse(slice, recon).detach().cpu().numpy().squeeze()))
        MAE = np.mean(threshold(mae(slice, recon).detach().cpu().numpy().squeeze()))
        SSIM = ssim(slice, recon)
        writer.write(image[:-4]+', '+str(id+1)+', '+str(MAE)+', '+str(MSE)+', '+str(SSIM.item())+'\n')

        
    print('-'*20)

print('-'*20)
print('Finished.')