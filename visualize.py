import torch
import lpips

from config.parser_module import settings_parser
from utils.process import mask_builder, val_loader 
from utils.loss import l1_error
from utils.load_model import load_model

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os

# Author: @simonamador

parser = settings_parser()
args = parser.parse_args()

print('-'*25)

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
print('Beginning anomaly maps generation:')
print('-'*20)

if extra:
    model_name = extra + view + '_' + model + '_AE_' + loss_type + '_b' +str(batch) + '_' + date
else:
    model_name = view + '_' + model + '_AE_' + loss_type + '_b' +str(batch) + '_' + date

model_path = path + '/Results/' + model_name + '/Saved_models/'

h = w = 158

#158, 126, 110
encoder, decoder = load_model(model_path, h, w, z_dim)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

lpips_vgg = lpips.LPIPS(pretrained = True, net = 'alex', eval_mode = True, spatial = True, lpips = True).to(device)


if anomaly == 'vm':
    img_path = 'Ventriculomegaly/recon_img/'
    images = os.listdir(path + img_path)
else:
    img_path = 'healthy_dataset/test/'
    images = os.listdir(path + img_path)

save_path = path + 'Results/Visual_' + anomaly + '/' + model_name + '/'
if not os.path.exists(save_path):
        os.mkdir(save_path)

for idx,image in enumerate(images):
    print('-'*20)
    print(f'Currently in image {idx+1} of {len(images)}')
    val_path = path + img_path + image

    loader = val_loader(val_path, view)

    for id, slice in enumerate(loader):
        z = encoder(slice)
        recon = decoder(z)
        error = l1_error(slice, recon)

        slice = slice.detach().cpu().numpy().squeeze()
        recon = recon.detach().cpu().numpy().squeeze()

        th = mask_builder(slice, recon, lpips_vgg, device)

        masked_img = np.repeat(np.expand_dims(recon,axis=-1), 3, axis =-1)
        mask_r = np.expand_dims(th,axis=-1)/255*0.56
        mask_g = np.expand_dims(th,axis=-1)/255*0.93
        mask_b = np.expand_dims(th,axis=-1)/255*0.56
        mask = np.concatenate((mask_r,mask_b,mask_g),axis=-1)

        for ch in range(3):
             for x in range(158):
                  for y in range(158):
                       if mask[x,y,ch]>0:
                            masked_img[x,y,ch] = mask[x,y,ch]

        # error = perceptual_loss(recon,slice) * l1_error(recon,slice)
        # n_error, th = threshold(error.detach().cpu().numpy().squeeze())
        
        fig, axs = plt.subplots(2,3)
        axs[0,0].imshow(slice, cmap = "gray")
        axs[0,0].axis("off")
        axs[1,0].imshow(recon, cmap = "gray")
        axs[1,0].axis("off")
        axs[0,1].imshow(error.detach().cpu().numpy().squeeze(), cmap = "hot")
        axs[0,1].axis("off")
        axs[1,1].hist(error.detach().cpu().numpy().squeeze())
        axs[0,2].imshow(th, cmap = "hot")
        axs[0,2].axis("off")
        axs[1,2].imshow(masked_img)

        plt.savefig(save_path+image[:-4]+'_sl'+str(id)+'.png')
        plt.close()