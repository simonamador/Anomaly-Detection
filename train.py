import torch
from torch.nn import DataParallel
import torch.optim as optim

import os
import time

from utils.process import loader
from utils import loss as loss_lib
from config.parser_module import settings_parser

# Author: @simonamador

# The following code performs the training of the AE model. The training can be performed for different
# views, model types, loss functions, epochs, and batch sizes. 

# Validation function. Acts as the testing portion of training. Inputs the testing dataloader, encoder and
# decoder models, and the loss function. Outputs the loss of the model on the testing data.
def validation(ds, encoder, decoder, loss, base, model, beta=None):
    encoder.eval()
    decoder.eval()

    ae_loss = 0.0
    metric1 = 0.0
    metric2 = 0.0
    metric3 = 0.0 

    with torch.no_grad():
        for data in ds:
            img = data['image'].to(device)

            if model == 'bVAE':
                if base == 'default':
                    z, mu, log_var = encoder(img)
                    x_recon = decoder(z)
                    kld_loss = loss_lib.kld_loss(mu, log_var)
                    ed_loss = loss(x_recon,img) + (beta * kld_loss)
                else:
                    ga = data['ga'].to(device)
                    z, mu, log_var = encoder(img, ga)
                    x_recon = decoder(z)
                    kld_loss = loss_lib.kld_loss(mu, log_var)
                    ed_loss = loss(x_recon,img) + (beta * kld_loss)
            else:
                if base == 'default':
                    z = encoder(img)
                    x_recon = decoder(z)
                    ed_loss = loss(x_recon,img)
                else:
                    ga = data['ga'].to(device)
                    z = encoder(img, ga)
                    x_recon = decoder(z)
                    ed_loss = loss(x_recon,img)

            ae_loss += ed_loss
            metric1 += 1-loss_lib.ssim_loss(x_recon, img)
            metric2 += loss_lib.l2_loss(x_recon, img)
            metric3 += loss_lib.l1_loss(x_recon, img)
        ae_loss /= len(ds)
        metric1 /= len(ds)
        metric2 /= len(ds)
        metric3 /= len(ds)        
    
    metrics = (ae_loss, metric1, metric2, metric3)

    return metrics

# Training function. Inputs training dataloader, validation dataloader, h and w values (shape of image),
# size of the z_vector, model type, epochs of training and loss function. Trains the model, saves 
# training and testing loss for each epoch, saves the parameters for the best model and the last model.

def train(train_ds, val_ds, h, w, z_dim, base, mtype, epochs, loss, beta=None, ga_method = None):
    # Creates encoder & decoder models.

    if base == 'ga_VAE':
        from models.ga_vae import Encoder, Decoder
        encoder = Encoder(h,w,z_dim=z_dim,model=mtype, method = ga_method)
    else:
        from models.vae import Encoder, Decoder
        encoder = Encoder(h,w,z_dim=z_dim,model=mtype)


    decoder = Decoder(h,w,z_dim=int(z_dim/2))

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
            img = data['image'].to(device)

            if model == 'bVAE':
                if base == 'default':
                    z, mu, log_var = encoder(img)
                else:
                    ga = data['ga'].to(device)
                    z, mu, log_var = encoder(img, ga)
                x_recon = decoder(z)
                kld_loss = loss_lib.kld_loss(mu, log_var)
                ed_loss = loss(x_recon,img) + (beta * kld_loss)
            else:
                if base == 'default':
                    z = encoder(img)
                else:
                    ga = data['ga'].to(device)
                    z = encoder(img, ga)
                x_recon = decoder(z)
                ed_loss = loss(x_recon,img)

            optimizer.zero_grad()
            ed_loss.backward()
            optimizer.step()

            ae_loss_epoch += ed_loss.item()
            
            step +=1

        tr_loss = ae_loss_epoch / len(train_ds)

        if beta is None:
            metrics = validation(val_ds, encoder, decoder, loss, base, model)
        else:
            metrics = validation(val_ds, encoder, decoder, loss, base, model, beta=beta)
        val_loss = metrics[0].item()

        print('train_loss: {:.6f}'.format(tr_loss))
        print('val_loss: {:.6f}'.format(val_loss))

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

# The code first parses through input arguments 
# -- model --model_type, --model_view, --gpu, --epochs, --loss, --batch.

    parser = settings_parser()
    args = parser.parse_args()

    print('Trainining script.')
    print('-'*25)

    base = args.model
    model = args.type
    view = args.view
    ga_method = args.ga_method
    gpu = args.gpu
    epochs = args.epochs
    batch_size = args.batch
    loss_type = args.loss
    path = args.path
    beta = args.beta

    z_dim = args.z * 2                # Dimension of parameters for latent vector (latent vector size = z_dim/2)
    h = w = 158

    if model == 'bVAE':
        if args.beta is None:
            print('Beta value will be assigned 1.')
            beta = 1
        else:
            beta = args.beta

# Connect to GPU

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    print('GPU was correctly assigned.')
    print('-'*25)

# Define paths for obtaining dataset and saving models and results.
    source_path = path + 'healthy_dataset/'

    date = time.strftime('%Y%m%d', time.localtime(time.time()))
    results_path = path + 'Results'
    if not os.path.exists(results_path):
        os.mkdir(results_path)
        
    folder_name = "/{0}_{1}_AE_{2}_b{3}_{4}".format(view,model,loss_type,batch_size,date)
    if base == 'ga_VAE':
        folder_name += 'ga_VAE'
    tensor_path = results_path + folder_name + '/history.txt'
    model_path = results_path + folder_name + '/Saved_models/'
    if not os.path.exists(results_path + folder_name):
        os.mkdir(results_path + folder_name)
        os.mkdir(model_path)
    
    print('Directories and paths are correctly initialized.')
    print('-'*25)

# Defining the loss function.

    if loss_type == 'L2':
        loss = loss_lib.l2_loss
    elif loss_type == 'L1':
        loss = loss_lib.l1_loss
    elif loss_type == 'SSIM':
        loss = loss_lib.ssim_loss
    elif loss_type == 'MS_SSIM':
        loss = loss_lib.ms_ssim_loss
    elif loss_type == 'Mixed':
        loss = loss_lib.mixed_loss
    elif loss_type == 'perceptual':
        loss = loss_lib.perceptual_loss

# Begin the initialization of the datasets. Creates dataset iterativey for each subject and
# concatenates them together for both training and testing datasets (implements img_dataset class).

    print('Loading datasets.')
    print('-'*25)

    train_final, val_final = loader(source_path, view, batch_size, h)

    print('Data has been properly loaded.')
    print('-'*25)

    print('Beginning training.')
    print('.'*50)

# Conducts training
    train(train_final, val_final, h, w, z_dim, base, model, epochs, loss, beta=beta, ga_method = ga_method)