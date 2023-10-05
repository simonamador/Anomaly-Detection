import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

'''
from monai.utils import set_determinism
from monai.visualize import plot_2d_or_3d_image
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    AddChannelD,
    Compose,
    LoadImageD,
    ScaleIntensityD,
    EnsureTypeD,
)'''

from model import Encoder, Decoder
from glob import glob
import random
import os
import time

def paths(im_sz, z_dim):

    date = time.strftime('%Y%m%d', time.localtime(time.time()))
    results_path = '../Results'
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    folder_name = "/{0}_insize{1}_z{2}_denseAE_sigmoid".format(date, im_sz, z_dim)
    tensorboard_path = results_path + folder_name + '/Tensorboard'
    saved_model_path = results_path + folder_name + '/Saved_models/'
    log_path = results_path + folder_name + '/log'
    if not os.path.exists(results_path + folder_name):
        os.mkdir(results_path + folder_name)
        os.mkdir(tensorboard_path)
        os.mkdir(saved_model_path)
        os.mkdir(log_path)
    return tensorboard_path, saved_model_path, log_path

def validation(dataloader,encoder,decoder):
    encoder.eval()
    decoder.eval()

    loss = nn.MSELoss()
    ae_loss = 0.0

    with torch.no_grad():
        for data in dataloader:
            img = data['im']

            z = encoder(img)
            x_recon = decoder(z)

            ed_loss = loss(x_recon, img)
            ae_loss += ed_loss
        val_loss = ae_loss / len(dataloader)

    return val_loss

def train(trainloader,valloader,h,w,z_dim,device_ids,epochs):
    encoder = Encoder(h,w,z_dim=z_dim)
    decoder = Decoder(h,w,z_dim=z_dim)

    loss = nn.MSELoss()

    optimizer = optim.Adam([{'params': encoder.parameters()},
                               {'params': decoder.parameters()}], lr=1e-4, weight_decay=1e-5)

    tensor_path, model_path, log_path = paths(f'{h}-{w}',z_dim)
    writer = SummaryWriter(tensor_path)

    step = 0
    best_loss = 100

    for epoch in range(epochs):
        print('-'*15)
        print(f'epoch {epoch+1}/{epochs}')
        encoder.train()
        decoder.train()

        ae_loss_epoch = 0.0

        for data in trainloader:
            img = data['im']

            z = encoder(img)
            x_recon = decoder(z)

            ed_loss = loss(x_recon,img)

            optimizer.zero_grad()
            ed_loss.backward()
            optimizer.step()

            ae_loss_epoch += ed_loss.item()

            writer.add_scalar('step_train_loss',ed_loss, step)
            
            step +=1

        tr_loss = ae_loss_epoch / len(trainloader)
        val_loss = validation(valloader, encoder, decoder)

        print('train_loss: {:.4f}'.format(tr_loss))
        print('val_loss: {:.4f}'.format(val_loss))
        writer.add_scalars('train and val loss per epoch', {'train_loss': tr_loss,
                                                            'val_loss': val_loss
                                                            }, epoch + 1)

        if (epoch + 1) % 50 == 0 or (epoch + 1) == epochs:
            torch.save({
                'epoch': epoch + 1,
                'encoder': encoder.state_dict(),
            }, model_path + f'/encoder_{epoch + 1}.pth')

            torch.save({
                'epoch': epoch + 1,
                'decoder': decoder.state_dict(),
            }, model_path + f'/decoder_{epoch + 1}.pth')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'encoder': encoder.state_dict(),
            }, model_path + f'/encoder_best.pth')

            torch.save({
                'epoch': epoch + 1,
                'decoder': decoder.state_dict(),
            }, model_path + f'/decoder_best.pth')
            print(f'saved best model in epoch: {epoch+1}')
    
    writer.close()