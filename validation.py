import torch
import lpips

from utils.process import mask_builder
from utils.config import val_loader, load_model, settings_parser
from utils.loss import ssim_loss, l1_loss, l2_loss

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
base = args.model

print('-'*20)
print('Beginning validation:')
print('-'*20)

if extra:
    model_name = extra + view + '_' + model + '_AE_' + loss_type + '_b' +str(batch) + '_' + date
elif base == 'ga_VAE':
    model_name = view + '_' + model + '_AE_' + loss_type + '_b' +str(batch) + '_' + date + 'ga_VAE'
else:
    model_name = view + '_' + model + '_AE_' + loss_type + '_b' +str(batch) + '_' + date

model_path = path + '/Results/' + model_name + '/Saved_models/'

h = w = 158

encoder, decoder = load_model(model_path, base, args.ga_method, h, w, z_dim, model=model)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

encoder = torch.nn.DataParallel(encoder).to(device)
decoder = torch.nn.DataParallel(decoder).to(device)

lpips_vgg = lpips.LPIPS(pretrained = True, net = 'squeeze', eval_mode = True, spatial = True, lpips = True).to(device)

if anomaly == 'vm':
    img_path = 'Ventriculomegaly/recon_img/'
    images = os.listdir(path + img_path)
else:
    img_path = 'healthy_dataset/test/'
    images = os.listdir(path + img_path)

writer = open(path+'Results/Validations/' + anomaly + '_' + model_name +'.txt', 'w')
writer.write('Case_ID, Slide_ID, mae, mse, ssim, anomaly'+'\n')

for idx,image in enumerate(images):
    print('-'*20)
    print(f'Currently in image {idx+1} of {len(images)}')
    val_path = path + img_path + image

    if anomaly == 'vm':
        loader = val_loader(val_path, view, image[:-4], data='vm')
    else:
        loader = val_loader(val_path, view, image[:-4])

    for id, slice in enumerate(loader):
        img = slice['image'].to(device)
        if base == 'ga_VAE':
            ga = slice['ga'].to(device)
            z = encoder(img, ga)
        else:
            z = encoder(img)

        recon = decoder(z)

        MSE = l2_loss(img, recon).item()
        MAE = l1_loss(img, recon).item()
        SSIM = 1-ssim_loss(img, recon)

        anomaly_map = np.mean(mask_builder(
            img.detach().cpu().numpy().squeeze(), 
            recon.detach().cpu().numpy().squeeze(), 
            lpips_vgg, device))

        writer.write(image[:-4]+', '+str(id+1)+', '+str(MAE)+
                     ', '+str(MSE)+', '+str(SSIM.item())+', '+
                     str(anomaly_map.item())+'\n')
        
    print('-'*20)

print('Finished.')