import os
import matplotlib.pyplot as plt

from utils.load_model import load_model
from utils.process import val_loader
from config.parser_module import settings_parser

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
train_path = path + '/Refinement/Train_Refinement/' + model_name + '/'
if not os.path.exists(train_path):
        os.mkdir(train_path)
test_path = path + '/Refinement/Test_Refinement/' + model_name + '/'
if not os.path.exists(test_path):
        os.mkdir(test_path)

h = w = 158

encoder, decoder = load_model(model_path, w, h, z_dim)

img_path = 'healthy_dataset/Raw/'
images = os.listdir(path + img_path)

for idx,image in enumerate(images):
    print('-'*20)
    print(f'Currently in image {image}')
    source_path = path + img_path + image

    loader = val_loader(source_path,view)

    for id, slice in enumerate(loader):
        z = encoder(slice)
        recon = decoder(z)

        slice = slice.detach().cpu().numpy().squeeze()
        recon = recon.detach().cpu().numpy().squeeze()

        plt.imsave(train_path+image+'_s'+str(id)+'.png', recon, cmap="gray")
        plt.imsave(test_path+image+'_s'+str(id)+'.png', slice, cmap="gray")