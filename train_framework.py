# Code based on https://github.com/ci-ber/PHANES and https://github.com/researchmm/AOT-GAN-for-Inpainting

import torch
from torch.nn import DataParallel
import torch.optim as optim

import matplotlib.pyplot as plt

from models.framework import Framework
from utils.process import loader
from utils.load_model import load_model
from utils import loss as loss_lib

class Trainer:
    def __init__(self, source_path, model_path, tensor_path,
                 image_path, device, z_dim, ga, method, model, 
                 base, view, n, pretrained, pretrained_path):
        
        self.device = device
        self.ga = ga
        self.model = model
        self.model = Framework(n, z_dim, method, device, model, ga)  
        self.model_path = model_path  
        self.tensor_path = tensor_path 
        self.image_path = image_path   

        if pretrained == True:
            encoder, decoder = load_model(pretrained_path, base, method, n, n, z_dim, model=model)
            self.model.encoder = encoder
            self.model.decoder = decoder

        self.loss_keys = {'L1': 1, 'Style': 250, 'Perceptual': 0.1}
        self.losses = {'L1':loss_lib.l1_loss,
                'Style':loss_lib.Style,
                'Perceptual':loss_lib.Perceptual}
        self.adv_loss = loss_lib.smgan
        self.adv_weight = 0.01

        train_dl, val_dl = loader(source_path, view)

        self.loader = {"tr": train_dl, "ts": val_dl}
        
        self.optimizer_e = optim.Adam(self.model.encoder.parameters(), lr=1e-4, weight_decay=1e-5)
        self.optimizer_d = optim.Adam(self.model.decoder.parameters(), lr=1e-4, weight_decay=1e-5)
        self.optimizer_netG = optim.Adam(self.model.refineG.parameters(), lr=1e-4)
        self.optimizer_netD = optim.Adam(self.model.refineD.parameters(), lr=1e-4)

    def train_inpainting(self, epochs, tensor_path):
        
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        for param in self.model.decoder.parameters():
            param.requires_grad = False
        for param in self.model.refineG.parameters():
            param.requires_grad = True
        for param in self.model.refineD.parameters():
            param.requires_grad = True

        self.model = DataParallel(self.model).to(self.device)
        
        self.writer = open(tensor_path, 'w')
        self.writer.write('Epoch, Train_loss, Val_loss, SSIM, MSE, MAE'+'\n')

        self.best_loss = 10000

        # Trains for all epochs
        for epoch in range(epochs):
            print('-'*15)
            print(f'epoch {epoch+1}/{epochs}')
            self.model.train()

            epoch_refineG_loss, epoch_refineD_loss = 0.0, 0.0,

            for data in self.loader["tr"]:
                img = data['image'].to(self.device)

                if self.ga == True:
                    ga = data['ga'].to(self.device)
                    anomap, res_dic = self.model(img, ga)
                else:
                    anomap, res_dic = self.model(img)
                
                losses = {}
                for name, weight in self.loss_keys.items():
                    losses[name] = weight * self.losses[name](res_dic["y_ref"], img)

                dis_loss, gen_loss = self.adv_loss(self.model.refineD, anomap, img, res_dic["mask"])

                losses['advg'] = gen_loss * self.adv_weight
                
                self.optimizer_netG.zero_grad()
                self.optimizer_netD.zero_grad()
                sum(losses.values()).backward()
                dis_loss.backward()
                self.optimizer_netG.step()
                self.optimizer_netD.step()

                epoch_refineG_loss += sum(losses.values()).cpu().item()
                epoch_refineD_loss += dis_loss.cpu().item()
            
            epoch_refineG_loss /= len(self.loader["ts"])
            epoch_refineD_loss /= len(self.loader["ts"])

            val_loss, metrics, images = self.test()

            self.log(self, epoch, epochs, [epoch_refineG_loss, epoch_refineD_loss], val_loss, metrics, images)

            print('train_loss: {:.6f}'.format(epoch_refineG_loss))
            print('val_loss: {:.6f}'.format(val_loss[0]))

        self.writer.close()

    def test(self):
        self.model.eval()

        refineG_loss = 0
        refineD_loss = 0
        
        with torch.no_grad():
            for data in self.loader["ts"]:
                img = data['image'].to(self.device)

                if self.ga == True:
                    ga = data['ga'].to(self.device)
                    anomap, res_dic = self.model(img, ga)
                else:
                    anomap, res_dic = self.model(img)

                losses = {}
                for name, weight in self.loss_keys.items():
                    losses[name] = weight * self.losses[name](res_dic["y_ref"], img)

                dis_loss, gen_loss = self.adv_loss(self.model.refineD, anomap, img, res_dic["mask"])

                losses['advg'] = gen_loss * self.adv_weight

                refineG_loss += sum(losses.values()).cpu().item()
                refineD_loss += dis_loss.cpu().item()

                mse_loss = loss_lib.l2_loss(res_dic["y_ref"], img)
                mae_loss = loss_lib.l1_error(res_dic["y_ref"], img)
                ssim     = 1 - loss_lib.ssim_loss(res_dic["y_ref"], img)
                anom     = torch.mean(anomap.flatten())

            refineG_loss /= len(self.loader["ts"])
            refineD_loss /= len(self.loader["ts"])
            mse_loss /= len(self.loader["ts"])
            mae_loss /= len(self.loader["ts"])
            ssim /= len(self.loader["ts"])
            anom /= len(self.loader["ts"])    

            images = {"input": images[0], "recon": res_dic["x_recon"][0], "saliency": res_dic["saliency"][0],
                      "mask": res_dic["masks"][0], "ref_recon": res_dic["y_ref"][0], "anomaly": anomap[0]}    
        
        return {'losses': [refineG_loss, refineD_loss],'metrics': [mse_loss, mae_loss, ssim, anom], 'images': images}

    def log(self, epoch, epochs, tr_loss, val_loss, metrics, images):
        model_path = self.model_path
        self.writer.write(str(epoch+1) + ', ' +
                          str(tr_loss[0]) + ', ' +
                          str(tr_loss[1]) + ', ' +
                          str(val_loss[0]) + ', ' +
                          str(val_loss[1]) + ', ' +
                          str(metrics[0].item()) + ', ' +
                          str(metrics[1].item()) + ', ' +
                          str(metrics[2].item()) + ', ' +
                          str(metrics[3].item()) + '\n')

        if (epoch + 1) % 50 == 0 or (epoch + 1) == epochs:
            torch.save({
                'epoch': epoch + 1,
                'encoder': self.model.encoder.state_dict(),
            }, model_path + f'/encoder_{epoch + 1}.pth')

            torch.save({
                'epoch': epoch + 1,
                'decoder': self.model.decoder.state_dict(),
            }, model_path + f'/decoder_{epoch + 1}.pth')

            torch.save({
                'epoch': epoch + 1,
                'refineG': self.model.refineG.state_dict(),
            }, model_path + f'/refineG_{epoch + 1}.pth')

            torch.save({
                'epoch': epoch + 1,
                'refineD': self.model.refineD.state_dict(),
            }, model_path + f'/refineD_{epoch + 1}.pth')

            progress_im = self.plot(images)
            progress_im.savefig(self.image_path+'epoch_'+str(epoch+1)+'.png')

        if val_loss[0] < self.best_loss:
            self.best_loss = val_loss[0]
            torch.save({
                'epoch': epoch + 1,
                'encoder': self.model.encoder.state_dict(),
            }, model_path + f'/best_encoder.pth')

            torch.save({
                'epoch': epoch + 1,
                'decoder': self.model.decoder.state_dict(),
            }, model_path + f'/best_decoder.pth')

            torch.save({
                'epoch': epoch + 1,
                'refineG': self.model.refineG.state_dict(),
            }, model_path + f'/best_refineG.pth')

            torch.save({
                'epoch': epoch + 1,
                'refineD': self.model.refineD.state_dict(),
            }, model_path + f'/best_refineD.pth')

    def plot(images):
        fig, axs = plt.subplots(2,3)
        names = [["input", "recon", "ref_recon"], ["saliency", "anomaly", "mask"]]
        cmap_i = ["gray", "heat"]
        for x in range(2):
            for y in range(3):
                if x == 1 and y == 2:
                    cmap_i[1] = "binary"
                axs[x,y].imshow(images[names[x,y]].detach().cpu().numpy().squeeze(), cmap = cmap_i[x])
                axs[x,y].axis("off")
        return fig