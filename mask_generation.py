import torch
import os
import matplotlib.pyplot as plt
import lpips

from utils.load_model import load_model
from utils.process import val_loader, mask_builder
from config.parser_module import settings_parser

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
mask_path = path + '/Masks/' + model_name + 'VM/'

if not os.path.exists(mask_path):
        os.mkdir(mask_path)

h = w = 158

encoder, decoder = load_model(model_path, h, w, z_dim)

img_path = 'Ventriculomegaly/recon_img/'
images = os.listdir(path + img_path)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

lpips_vgg = lpips.LPIPS(pretrained = True, net = 'alex', eval_mode = True, spatial = True, lpips = True).to(device)

for idx,image in enumerate(images):
    print('-'*20)
    print(f'Currently in image {image}')
    source_path = path + img_path + image

    loader = val_loader(source_path, view)

    for id, slice in enumerate(loader):
        z = encoder(slice)
        recon = decoder(z)

        slice = slice.detach().cpu().numpy().squeeze()
        recon = recon.detach().cpu().numpy().squeeze()

        mask = mask_builder(slice, recon, lpips_vgg, device)

        plt.imsave(mask_path+image+'_s'+str(id)+'.png', mask, cmap="gray")