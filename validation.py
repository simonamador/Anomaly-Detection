import torch.nn as nn

from utils.process import val_loader, threshold
from utils.load_model import load_model
from utils.loss import ssim_loss
from config.parser_module import settings_parser

import os
import numpy as np

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

encoder, decoder = load_model(model_path, h, w, z_dim)

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

    loader = val_loader(val_path, view)

    for id, slice in enumerate(loader):
        z = encoder(slice)
        recon = decoder(z)

        MSE = np.mean(threshold(mse(slice, recon).detach().cpu().numpy().squeeze()))
        MAE = np.mean(threshold(mae(slice, recon).detach().cpu().numpy().squeeze()))
        SSIM = 1-ssim_loss(slice, recon)
        writer.write(image[:-4]+', '+str(id+1)+', '+str(MAE)+', '+str(MSE)+', '+str(SSIM.item())+'\n')
        
    print('-'*20)

print('-'*20)
print('Finished.')