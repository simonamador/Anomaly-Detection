import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from model import Encoder, Decoder

import numpy as np
import os
import argparse
import time

def validation(ds,encoder,decoder):
    encoder.eval()
    decoder.eval()

    loss = nn.MSELoss()
    ae_loss = 0.0

    with torch.no_grad():
        for data in ds:
            img = data.to(device)

            z = encoder(img)
            x_recon = decoder(z)

            ed_loss = loss(x_recon, img)
            ae_loss += ed_loss
        val_loss = ae_loss / len(ds)

    return val_loss

def train(train_ds,val_ds,h,w,z_dim,mtype,epochs):
    encoder = Encoder(h,w,z_dim=z_dim,model=mtype)
    decoder = Decoder(h,w,z_dim=z_dim,model=mtype)

    encoder = nn.DataParallel(encoder).to(device)
    decoder = nn.DataParallel(decoder).to(device)

    loss = nn.MSELoss()

    optimizer = optim.Adam([{'params': encoder.parameters()},
                               {'params': decoder.parameters()}], lr=1e-4, weight_decay=1e-5)

    writer = SummaryWriter(tensor_path)

    step = 0
    best_loss = 100

    for epoch in range(epochs):
        print('-'*15)
        print(f'epoch {epoch+1}/{epochs}')
        encoder.train()
        decoder.train()

        ae_loss_epoch = 0.0

        for data in train_ds:
            img = data.to(device)

            z = encoder(img)
            x_recon = decoder(z)

            ed_loss = loss(x_recon,img)

            optimizer.zero_grad()
            ed_loss.backward()
            optimizer.step()

            ae_loss_epoch += ed_loss.item()

            writer.add_scalar('step_train_loss',ed_loss, step)
            
            step +=1

        tr_loss = ae_loss_epoch / len(train_ds)
        val_loss = validation(val_ds, encoder, decoder)

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_type',
        dest='type',
        choices=['default', 'residual', 'self-attention','full'],
        required=True,
        help='''
        Type of model to train. Available options:
        "defalut" Default VAE using convolution blocks
        "residual: VAE which adds residual blocks between convolutions''')
    
    parser.add_argument('--model_view',
        dest='view',
        choices=['L', 'A', 'S'],
        required=True,
        help='''
        The view of the image input for the model. Options:
        "L" Left view
        "A" Axial view
        "S" Sagittal view''')
    
    parser.add_argument('--gpu',
        dest='gpu',
        choices=['0', '1', '2'],
        required=True,
        help='''
        The GPU that will be used for training. Terminals have the following options:
        Hanyang: 0, 1
        Busan: 0, 1, 2
        Sejong 0, 1, 2
        Songpa 0, 1
        Gangnam 0, 1
        ''')
    
    parser.add_argument('--epochs',
        dest='epochs',
        default=1,
        choices=range(1, 100),
        required=False,
        help='''
        Number of epochs for training.
        ''')

    args = parser.parse_args()

    print(args)
    print('-'*25)


    model = args.type
    view = args.view
    gpu = args.gpu
    epochs = args.epochs
    batch_size = 8
    z_dim = 512

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('GPU was correctly assigned.')
    print('-'*25)
    print(device)

    path = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/'

    source_path = path + 'healthy_dataset/' + view + '_view'

    date = time.strftime('%Y%m%d', time.localtime(time.time()))
    results_path = path + 'Results/'
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    folder_name = "/{0}_{1}_AE_{2}".format(view,model,date)
    tensor_path = results_path + folder_name + '/Tensorboard'
    model_path = results_path + folder_name + '/Saved_models/'
    if not os.path.exists(results_path + folder_name):
        os.mkdir(results_path + folder_name)
        os.mkdir(tensor_path)
        os.mkdir(model_path)
    
    print('Directories and paths are correctly initialized.')
    print('-'*25)


    train_set = np.load(source_path+'/train.npy')
    val_set = np.load(source_path+'/test.npy')

    h = train_set.shape[1]
    w = train_set.shape[2]

    train_set = np.expand_dims(train_set,axis=1)
    train_set = torch.from_numpy(train_set).type(torch.float)
    train_final = DataLoader(train_set, shuffle=True, batch_size=batch_size)

    val_set = np.expand_dims(val_set,axis=1)
    val_set = torch.from_numpy(val_set).type(torch.float)
    val_final = DataLoader(val_set, shuffle=True, batch_size=batch_size, num_workers=12)

    print('Data has been properly loaded.')
    print('-'*25)

    print(f"h={h}, w={w}")
    print()
    print('Beginning training.')
    print('.'*50)

    train(train_final,val_final,h,w,z_dim,model,epochs)