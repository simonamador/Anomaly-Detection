import torch
import numpy as np
import matplotlib.pyplot as plt
from model import Encoder, Decoder
from torch.utils.data import DataLoader
from collections import OrderedDict

path = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/'
model_path = path + '/Results/L_default_AE_20231011/Saved_models/'

encoder = Encoder(158,126,512)
decoder = Decoder(158,126,256)

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

print('Encoder and Decoder loaded.')

print('Loading new image')

val_set = np.load(path+'healthy_dataset/L_view_e/test.npy')

print(f'h = {val_set.shape[1]}, w = {val_set.shape[2]}')
print('Reconstructing image')

val_set = np.expand_dims(val_set,axis=1)
val_set = torch.from_numpy(val_set).type(torch.float)
loader = DataLoader(val_set, batch_size=1)

images = iter(loader)
img = next(images)
img = next(images)
img = next(images)
# img = next(images)
# img = next(images)

z = encoder(img)
recon = decoder(z)

recon = recon.detach().cpu().numpy().squeeze()
input = img.cpu().numpy().squeeze()

error = np.mean((recon - input)**2)
print(error)

plt.imshow(input, cmap='gray')
plt.show()
plt.imshow(recon, cmap='gray')
plt.show()

print('Done.')