import torch
from torch.nn import DataParallel
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from model import Encoder, Decoder
from process import img_dataset

import loss as loss_lib
import numpy as np
import os
import argparse
import time

# Author: @simonamador

# The following code performs the training of the AE model. The training can be performed for different
# views, model types, loss functions, epochs, and batch sizes. 

# Validation function. Acts as the testing portion of training. Inputs the testing dataloader, encoder and
# decoder models, and the loss function. Outputs the loss of the model on the testing data.
def validation(ds,encoder,decoder,loss,model,beta=None):
    encoder.eval()
    decoder.eval()

    mse = loss_lib.l2_loss()
    mae = loss_lib.l1_loss()

    ae_loss = 0.0
    metric1 = 0.0
    metric2 = 0.0
    metric3 = 0.0 

    with torch.no_grad():
        for data in ds:
            img = data.to(device)

            if model == 'bVAE':
                z, mu, log_var = encoder(img)
                x_recon = decoder(z)
                kld_loss = loss_lib.kld_loss(mu, log_var)
                ed_loss = loss(x_recon,img) + kld_loss*beta
            else:
                z = encoder(img)
                x_recon = decoder(z)
                ed_loss = loss(x_recon,img)

            ae_loss += ed_loss
            metric1 += (100-loss_lib.ssim_loss(x_recon, img))/100
            metric2 += mse(x_recon, img)
            metric3 += mae(x_recon, img)
        ae_loss /= len(ds)
        metric1 /= len(ds)
        metric2 /= len(ds)
        metric3 /= len(ds)        
    
    metrics = (ae_loss, metric1, metric2, metric3)

    return metrics

# Training function. Inputs training dataloader, validation dataloader, h and w values (shape of image),
# size of the z_vector (512), model type, epochs of training and loss function. Trains the model, saves 
# training and testing loss for each epoch, saves the parameters for the best model and the last model.

def train(train_ds,val_ds,h,w,z_dim,mtype,epochs,loss,beta=None):
    # Creates encoder & decoder models.

    encoder = Encoder(h,w,z_dim=z_dim,model=mtype)
    decoder = Decoder(h,w,z_dim=int(z_dim/2),model=mtype)

    encoder = DataParallel(encoder).to(device)
    decoder = DataParallel(decoder).to(device)

    # Sets up the optimizer
    optimizer = optim.Adam([{'params': encoder.parameters()},
                               {'params': decoder.parameters()}], lr=1e-4, weight_decay=1e-5)

    # Initialize the logger
    writer = open(tensor_path,'w')
    writer.write('Epoch, Train_loss, Val_loss, SSIM, MSE, MAE'+'\n')

    step = 0
    best_loss = 10000

    # Trains for all epochs
    for epoch in range(epochs):
        print('-'*15)
        print(f'epoch {epoch+1}/{epochs}')
        encoder.train()
        decoder.train()

        ae_loss_epoch = 0.0

        for data in train_ds:
            img = data.to(device)

            if model == 'bVAE':
                z, mu, log_var = encoder(img)
                x_recon = decoder(z)
                kld_loss = loss_lib.kld_loss(mu, log_var)
                ed_loss = loss(x_recon,img) + kld_loss*beta
            else:
                z = encoder(img)
                x_recon = decoder(z)
                ed_loss = loss(x_recon,img)

            optimizer.zero_grad()
            ed_loss.backward()
            optimizer.step()

            ae_loss_epoch += ed_loss.item()
            
            step +=1

        tr_loss = ae_loss_epoch / len(train_ds)

        if beta is None:
            metrics = validation(val_ds, encoder, decoder, loss, model)
        else:
            metrics = validation(val_ds, encoder, decoder, loss, model, beta=beta)
        val_loss = metrics[0].item()

        print('train_loss: {:.4f}'.format(tr_loss))
        print('val_loss: {:.4f}'.format(val_loss))

        writer.write(str(epoch+1) + ', ' + str(tr_loss)+ ', ' + str(val_loss)+ ', ' + 
                     str(metrics[1].item())+ ', ' + str(metrics[2].item())+ ', ' + 
                     str(metrics[3].item()) + '\n')

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


# Main code
if __name__ == '__main__':

# The code first parses through input arguments --model_type, --model_view, --gpu, --epochs, --loss, --batch.
# Model type: default or residual (for now). Which model is it going to train.
# Model view: which view is the model getting train to (L=saggital,A=frontal,S=axial)
# GPU: Defines which GPU to use
# Epochs: How many epochs to train the model for
# Loss: Which loss function to implement
# Batch: Batch size for training

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_type',
        dest='type',
        choices=['default', 'residual', 'bVAE', 'self-attention','full'],
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
        default='SSIM',
        choices=['L2', 'SSIM', 'MS_SSIM', 'Mixed'],
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
    parser.add_argument('--beta',
        dest='beta',
        type=float,
        default=None,
        choices=[0.1, 1, 10, 100],
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
    beta = None

    if model == 'bVAE':
        if args.beta is None:
            print('Beta value will be assigned 1.')
            beta = 1
        else:
            beta = args.beta

    z_dim = 800                 # Dimension of parameters for latent vector (latent vector size = z_dim/2)
    h = w = 158


# Connect to GPU

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    print('GPU was correctly assigned.')
    print('-'*25)

# Define paths for obtaining dataset and saving models and results.
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

# Defining the loss function.

    if loss_type == 'L2':
        loss = loss_lib.l2_loss()
    elif loss_type == 'SSIM':
        loss = loss_lib.ssim_loss
    elif loss_type == 'MS_SSIM':
        loss = loss_lib.ms_ssim_loss
    elif loss_type == 'Mixed':
        loss = loss_lib.l1_ssim_loss
    elif loss_type == 'perceptual':
        loss = loss_lib.perceptual_loss

# Define h and w (shape of the images), change depending on the view.
    if view == 'L':
        ids = np.arange(start=40,stop=70)
    elif view == 'A':
        ids = np.arange(start=64,stop=94)
    else:
        ids = np.arange(start=48,stop=78)

    print('Loading data.')
    print('-'*25)

# Begin the initialization of the datasets. Creates dataset iterativey for each subject and
# concatenates them together for both training and testing datasets (implements img_dataset class).

    train_id = os.listdir(source_path+'train/')
    test_id = os.listdir(source_path+'test/')

    train_set = img_dataset(source_path+'train/'+train_id[0], view)
    train_set = Subset(train_set,ids)
    test_set = img_dataset(source_path+'test/'+test_id[0],view)
    test_set = Subset(test_set,ids)

    for idx,image in enumerate(train_id):
        if idx != 0:
            train_path = source_path + 'train/' + image
            tr_set = img_dataset(train_path,view, size = h)
            tr_set = Subset(tr_set,ids)
            train_set = torch.utils.data.ConcatDataset([train_set, tr_set])

    for idx,image in enumerate(test_id):
        if idx != 0:
            test_path = source_path + 'test/' + image
            ts_set = img_dataset(test_path,view, size = h)
            ts_set = Subset(ts_set,ids)
            test_set = torch.utils.data.ConcatDataset([test_set, ts_set])

# Dataloaders generated from datasets 
    train_final = DataLoader(train_set, shuffle=True, batch_size=batch_size,num_workers=12)
    val_final = DataLoader(test_set, shuffle=True, batch_size=batch_size,num_workers=12)

    print('Data has been properly loaded.')
    print('-'*25)


    print('Beginning training.')
    print('.'*50)

# Conducts training
    train(train_final,val_final,h,w,z_dim,model,epochs,loss,beta=beta)