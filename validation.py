import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from model import Encoder, Decoder
from train import img_dataset
import os
import argparse
import numpy as np

from collections import OrderedDict
 
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--view',
        dest='view',
        choices=['L','A','S'],
        required=True,
        help='''
        The view of the image input for the model. Options:
        "L" Left view
        "A" Axial view
        "S" Sagittal view''')

parser.add_argument('--model',
        dest='model',
        default='default_AE_SSIM',
        choices=['default_AE_L2','default_AE_SSIM','residual_AE_L2','residual_AE_SSIM'],
        required=False,
        help='''
        Model name. Ex. default_AE_L2
        ''')

parser.add_argument('--date',
        dest='date',
        default='20231011',
        required=False,
        help='''
        Date of model training.
        ''')

parser.add_argument('--ga',
        dest='ga',
        type = int,
        default=31,
        choices = range(22,39),
        required=False,
        help='''
        GA to validate.
        ''')

args = parser.parse_args()

print(args)

view = args.view
# model = args.model
date = args.date
# ga = args.ga
models = ['default_AE_L2','default_AE_SSIM']
gas = range(22,40)

path = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/'

print('-'*20)
print('Beginning validation:')
print('-'*20)

writer = open(path+'Results/GA_val_'+view+'_'+date+'_2.txt', 'w')
writer.write('GA, L2 error, SSIM error'+'\n')

loss = nn.MSELoss()

for ga in gas: 
    images = os.listdir(path + 'healthy_dataset/recon-by-GA/GA_' + str(ga))
    m_error = []
    for model in models:

        model_path = path + '/Results/' + view + '_' + model + '_' + date + '/Saved_models/'

        if view == 'L':
            w = 158
            h = 126
        elif view == 'A':
            w = 110
            h = 126
        else:
            w = 110
            h = 158

        encoder = Encoder(w,h,512)
        decoder = Decoder(w,h,256)

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

        errors = []
        for idx,image in enumerate(images):
            print('-'*20)
            print(f'Currently in image {idx} of {len(images)}')
            val_path = path + 'healthy_dataset/recon-by-GA/GA_' + str(ga) + '/' + image

            val_set = img_dataset(val_path,view)

            loader = DataLoader(val_set, batch_size=8, shuffle=True)
            for id, slice in enumerate(loader):
                z = encoder(slice)
                recon = decoder(z)

                error = loss(recon,slice)
                errors.append(error.detach().numpy())

                recon = recon.detach().cpu().numpy().squeeze()
                input = slice.cpu().numpy().squeeze()
                
                if (idx == 0) & (id == 10):
                    fig = plt.figure()
                    fig.add_subplot(2,1,1)
                    plt.hist(input[0])
                    fig.add_subplot(2,1,2)
                    plt.hist(recon[0])
                    plt.show()

                    fig = plt.figure()
                    fig.add_subplot(3,1,1)
                    plt.imshow(input[0], cmap='gray')
                    plt.axis('off')
                    fig.add_subplot(3,1,2)
                    plt.imshow(recon[0], cmap='gray')
                    plt.axis('off')
                    fig.add_subplot(3,1,3)
                    plt.imshow((recon[0]-input[0])**2, cmap='hot')
                    plt.axis('off')
                    plt.show()

        error = np.mean(errors)
        print('Model: ' + model)
        print(f'Mean error: {error}')
        m_error.append(error)
        print('-'*20)
    writer.write(str(ga)+', '+str(m_error[0])+', '+str(m_error[1])+'\n')
writer.close()
print('-'*20)
print('Finished.')