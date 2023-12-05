import torch
from torch.utils.data import DataLoader, Subset

import os
import numpy as np
from collections import OrderedDict
import cv2
import matplotlib.pyplot as plt

from model import Encoder, Decoder
from utils import img_dataset, perceptual_loss
from config import settings_parser

def mask_generation(input, recon):

    clahe = cv2.createCLAHE(clipLimit=2,
	    tileGridSize=(11, 11))
  
    eq_input = clahe.apply((255*input).astype(np.uint8))
    eq_recon = clahe.apply((255*recon).astype(np.uint8))
    dif = abs(eq_recon - eq_input)

    p95 = np.percentile(dif, 95)
    if p95>0:
        norm95 = (dif*1.0 / p95).clip(0, 1)
    else:
        norm95 = dif.clip(0, 1)

    input = np.expand_dims(np.expand_dims(input, axis=0).repeat(3, axis=0),axis=0)
    recon = np.expand_dims(np.expand_dims(recon, axis=0).repeat(3, axis=0),axis=0)
    input = torch.from_numpy(input).type(torch.float)
    recon = torch.from_numpy(recon).type(torch.float)
    
    per = perceptual_loss
    slpips = per(input, recon)
    m = norm95*slpips.item()
    m = m * 255/np.max(m)
    m95 = np.percentile(m, 95)
    ret, b_m = cv2.threshold(m,m95-1,255,cv2.THRESH_BINARY)
    return b_m

parser = settings_parser()
args = parser.parse_args()

model = args.type
view = args.view
batch = args.batch
loss_type = args.loss
date = args.date
extra = args.extra
z_dim = args.z
path = args.path

print('-'*20)
print('Beginning mask generation:')
print('-'*20)

if extra:
    model_name = extra + view + '_' + model + '_AE_' + loss_type + '_b' +str(batch) + '_' + date
else:
    model_name = view + '_' + model + '_AE_' + loss_type + '_b' +str(batch) + '_' + date

model_path = path + '/Results/' + model_name + '/Saved_models/'
mask_path = path + '/Masks/' + model_name + '/'

if not os.path.exists(mask_path):
        os.mkdir(mask_path)

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

img_path = 'healthy_dataset/recon_img/'
images = os.listdir(path + img_path)

for idx,image in enumerate(images):
    print('-'*20)
    print(f'Currently in image {image}')
    source_path = path + img_path + image

    source_set = img_dataset(source_path,view)

    source_set = Subset(source_set,ids)

    loader = DataLoader(source_set, batch_size=1)

    for id, slice in enumerate(loader):
        z = encoder(slice)
        recon = decoder(z)

        slice = slice.detach().cpu().numpy().squeeze()
        recon = recon.detach().cpu().numpy().squeeze()

        mask = mask_generation(slice, recon)

        plt.imsave(mask_path+image+'_s'+str(id)+'.png', mask, cmap="gray")