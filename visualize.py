import torch.nn as nn

from config import settings_parser
from utils.process import perceptual_loss, threshold, val_loader 
from utils.load_model import load_model

import matplotlib.pyplot as plt
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

    loader = val_loader(val_path, view)

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