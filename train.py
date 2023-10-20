import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import Encoder, Decoder, SSIM_Loss

import nibabel as nib

import numpy as np
import os
import argparse
import time

class img_dataset(Dataset):
    def __init__(self, root_dir, view):
        self.root_dir = root_dir
        self.view = view

    def __len__(self):
        if self.view == 'L':
            size = 110
        elif self.view == 'A':
            size = 158
        else:
            size = 126
        return size
    
    def __getitem__(self, idx):
        raw = nib.load(self.root_dir).get_fdata()
        if self.view == 'L':
            n_img = raw[idx,:158,:]
        elif self.view == 'A':
            n_img = raw[:110,idx,:]
        else:
            n_img = raw[:110,:158,idx]
        
        num = n_img-np.min(n_img)
        den = np.max(n_img)-np.min(n_img)
        out = np.zeros((n_img.shape[0], n_img.shape[1]))
    
        n_img = np.divide(num, den, out=out, where=den!=0)

        n_img *= 255
        n_img = np.expand_dims(n_img,axis=0)
        n_img = torch.from_numpy(n_img).type(torch.float)

        return n_img


def validation(ds,encoder,decoder,loss):
    encoder.eval()
    decoder.eval()

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

def train(train_ds,val_ds,h,w,z_dim,mtype,epochs,loss):
    encoder = Encoder(h,w,z_dim=z_dim,model=mtype)
    decoder = Decoder(h,w,z_dim=int(z_dim/2),model=mtype)

    encoder = nn.DataParallel(encoder).to(device)
    decoder = nn.DataParallel(decoder).to(device)

    optimizer = optim.Adam([{'params': encoder.parameters()},
                               {'params': decoder.parameters()}], lr=1e-4, weight_decay=1e-5)

    writer = open(tensor_path,'w')
    writer.write('Epoch, Train_loss, Val_loss'+'\n')

    step = 0
    best_loss = 10000

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
            
            step +=1

        tr_loss = ae_loss_epoch / len(train_ds)
        val_loss = validation(val_ds, encoder, decoder, loss)
        val_loss = val_loss.item()

        print('train_loss: {:.4f}'.format(tr_loss))
        print('val_loss: {:.4f}'.format(val_loss))

        writer.write(str(epoch+1) + ', ' + str(tr_loss)+ ', ' + str(val_loss) + '\n')

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
        type=int,
        default=50,
        choices=range(1, 15000),
        required=False,
        help='''
        Number of epochs for training.
        ''')
    
    parser.add_argument('--loss',
        dest='loss',
        default='L2',
        choices=['L2', 'SSIM'],
        required=False,
        help='''
        Loss function:
        L2 = Mean square error.
        SSIM = Structural similarity index.
        ''')

    parser.add_argument('--batch',
        dest='batch',
        type=int,
        default=1,
        choices=range(1, 512),
        required=False,
        help='''
        Number of batch size.
        ''')

    args = parser.parse_args()

    print(args)
    print('-'*25)


    model = args.type
    view = args.view
    gpu = args.gpu
    epochs = args.epochs
    batch_size = args.batch
    loss_type = args.loss
    z_dim = 512

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    print('GPU was correctly assigned.')
    print('-'*25)

    path = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/'

    source_path = path + 'healthy_dataset/'

    date = time.strftime('%Y%m%d', time.localtime(time.time()))
    results_path = path + 'Results'
    if not os.path.exists(results_path):
        os.mkdir(results_path)
        
    folder_name = "/{0}_{1}_AE_{2}_b{3}_{4}".format(view,model,loss_type,batch_size,date)
    tensor_path = results_path + folder_name + '/history.txt'
    model_path = results_path + folder_name + '/Saved_models/'
    if not os.path.exists(results_path + folder_name):
        os.mkdir(results_path + folder_name)
        os.mkdir(model_path)
    
    print('Directories and paths are correctly initialized.')
    print('-'*25)

    print('Initializing loss function.')
    print('-'*25)

    if loss_type == 'L2':
        loss = nn.MSELoss()
    elif loss_type == 'SSIM':
        loss = SSIM_Loss()

    print('Loading data.')
    print('-'*25)

    train_id = os.listdir(source_path+'train/')
    test_id = os.listdir(source_path+'test/')

    train_set = img_dataset(source_path+'train/'+train_id[0], view)
    test_set = img_dataset(source_path+'test/'+test_id[0],view)

    for idx,image in enumerate(train_id):
        if idx != 0:
            train_path = source_path + 'train/' + image
            tr_set = img_dataset(train_path,view)
            train_set = torch.utils.data.ConcatDataset([train_set, tr_set])

    for idx,image in enumerate(test_id):
        if idx != 0:
            test_path = source_path + 'test/' + image
            ts_set = img_dataset(test_path,view)
            test_set = torch.utils.data.ConcatDataset([test_set, ts_set])

    train_final = DataLoader(train_set, shuffle=True, batch_size=batch_size,num_workers=12)
    val_final = DataLoader(test_set, shuffle=True, batch_size=batch_size,num_workers=12)

    print('Data has been properly loaded.')
    print('-'*25)

    if view == 'L':
        h = 158
        w = 126
    elif view == 'A':
        h = 110
        w = 126
    else:
        h = 110
        w = 158

    print(f"h={h}, w={w}")
    print()
    print('Beginning training.')
    print('.'*50)

    train(train_final,val_final,h,w,z_dim,model,epochs,loss)