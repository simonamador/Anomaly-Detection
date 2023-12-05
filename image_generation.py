import torch
from torch.utils.data import DataLoader, Subset
import os
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

from vae_model.model import Encoder, Decoder
from utils.process import img_dataset
from config import settings_parser

parser = settings_parser()
args = parser.parse_args()

print('-'*25)


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
train_path = path + '/Train_Refinement/' + model_name + '/'
if not os.path.exists(train_path):
        os.mkdir(train_path)
test_path = path + '/Test_Refinement/' + model_name + '/'
if not os.path.exists(test_path):
        os.mkdir(test_path)

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

        plt.imsave(train_path+image+'_s'+str(id)+'.png', recon, cmap="gray")
        plt.imsave(test_path+image+'_s'+str(id)+'.png', slice, cmap="gray")