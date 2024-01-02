from utils.config import load_model, val_loader
from utils.loss import ssim_loss, l1_loss, l2_loss
from models.anomaly import Anomaly
from models.framework import Framework

import torch
import matplotlib.pyplot as plt
import scipy.stats as stts
import seaborn as sns
import pandas as pd
import os

class Validator:
    def __init__(self, path, model_path, base, model, view, method, 
                 loss, batch, z_dim, date, n, device):
        if base == 'ga_VAE':
            self.ga = True
            model_name = view + '_' + model + '_AE_' + loss + '_b' +str(batch) + '_' + date + 'ga_VAE'
        else:
            self.ga = False
            model_name = view + '_' + model + '_AE_' + loss + '_b' +str(batch) + '_' + date

        self.view = view
        self.device = device
        self.model = Framework(n, z_dim, method, device, model, self.ga)
        self.model.encoder, self.model.decoder = load_model(model_path, base, method, 
                                                            n, n, z_dim, model=model)
       
        self.hist_path = path+'Results' + model_name + '/history.txt'
        self.val_path = path+'Results/Validations/'+model_name+'/'

        if not os.path.exists(self.val_path):
            os.mkdir(self.val_path)
            os.mkdir(self.val_path+'TD/')
            os.mkdir(self.val_path+'VM/')

        self.vm_path = path+'/Ventriculomegaly/recon_img/'
        self.vm_images = os.listdir(self.vm_path)
        self.td_path = path+'healthy_dataset/test/'
        self.td_images= os.listdir(self.td_path)

    def validation(self,):
        writer = open(self.val_path+'validation.txt', 'w')
        writer.write('Class, Case_ID, Slide_ID, MAE, MSE, SSIM, Anomaly'+'\n')
        for image in self.vm_images:
            loader = val_loader(self.vm_path+image, self.view, image[:-4], data='vm')
            for id, slice in enumerate(loader):
                img = slice['image'].to(self.device)
                if self.ga:
                    ga = slice['ga'].to(self.device)
                    recon_ref, rec_dic = self.model(img, ga)
                else:
                    recon_ref, rec_dic = self.model(img)

                MSE = l2_loss(img, recon_ref).item()
                MAE = l1_loss(img, recon_ref).item()
                SSIM = 1-ssim_loss(img, recon_ref)
                saliency, anomap = Anomaly.anomaly(img, recon_ref)
                anomap = anomap * saliency
                
                writer.write('vm, '+
                                  image[:-4]+', '+
                                  str(id+1)+', '+
                                  str(MAE)+', '+
                                  str(MSE)+', '+
                                  str(SSIM.item())+', '
                                  +str(torch.mean(anomap).item())+'\n')
                
                images = {"input": img[0][0], "recon": rec_dic["x_recon"][0], "saliency": rec_dic["saliency"][0],
                      "mask": -rec_dic["mask"][0], "ref_recon": rec_dic[0], "anomaly": anomap[0][0]} 
                fig = self.plot(images)
                fig.save(self.val_path+'VM/'+image[:-4]+'.png')
                
        for image in self.td_images:
            loader = val_loader(self.td_path+image, self.view, image[:-4])
            for id, slice in enumerate(loader):
                img = slice['image'].to(self.device)
                if self.ga:
                    ga = slice['ga'].to(self.device)
                    recon_ref, rec_dic = self.encoder(img, ga)
                else:
                    recon_ref, rec_dic = self.encoder(img)

                MSE = l2_loss(img, recon_ref).item()
                MAE = l1_loss(img, recon_ref).item()
                SSIM = 1-ssim_loss(img, recon_ref)
                saliency, anomap = Anomaly.anomaly(img, recon_ref)
                anomap = anomap * saliency
                anomap = torch.mean(anomap)

                images = {"input": img[0][0], "recon": rec_dic["x_recon"][0], "saliency": rec_dic["saliency"][0],
                      "mask": -rec_dic["mask"][0], "ref_recon": rec_dic[0], "anomaly": anomap[0][0]} 
                fig = self.plot(images)
                fig.save(self.val_path+'TD/'+image[:-4]+'.png')
                
                writer.write('td, '+
                                  image[:-4]+', '+
                                  str(id+1)+', '+
                                  str(MAE)+', '+
                                  str(MSE)+', '+
                                  str(SSIM.item())+', '
                                  +str(anomap.item())+'\n')
                
        writer.close()
                
    def plot(self, images):
        fig, axs = plt.subplots(2,3)
        names = [["input", "recon", "ref_recon"], ["saliency", "mask", "anomaly"]]
        cmap_i = ["gray", "hot"]
        for x in range(2):
            for y in range(3):
                if x == 1 and y == 1:
                    cmap_i[1] = "binary"
                axs[x, y].imshow(images[names[x][y]].detach().cpu().numpy().squeeze(), cmap = cmap_i[x])
                axs[x, y].title(names[x][y])
                axs[x, y].axis("off")
        return fig
    
    def tc(self,):
        df = pd.read_csv(self.hist_path)

    def stat_analysis(self,):
        df = pd.read_csv(self.val_path+'validation.txt')

        n_df = pd.melt(df, id_vars='Class', value_vars=['MSE', 'MAE', 'Anomaly'],
                      var_name='Metric', value_name='Value')
        td_df = df.where(df['Class']=='td').dropna()
        vm_df = df.where(df['Class']=='vm').dropna()

        if self.view == 'L':
            view = 'Sagittal'
        elif  self.view == 'A':
            view = 'Coronal'
        else:
            view = 'Axial'

        sns.set_style("whitegrid", {'axes.grid' : False})   
        ax = sns.violinplot(data=n_df, x="Metric", y="Value", hue="Class", split=True, inner="quart").set(
            title=view+' view.'
        )

        plt.savefig(self.val_path+view+'violin'+'.png')
        
        stat_mse, p_mse = stts.mannwhitneyu(td_df['MSE'], vm_df['MSE'])
        stat_mae, p_mae = stts.mannwhitneyu(td_df['MAE'], vm_df['MAE'])
        stat_ssim, p_ssim = stts.mannwhitneyu(td_df['SSIM'], vm_df['SSIM'])
        stat_anom, p_anom = stts.mannwhitneyu(td_df['Anomaly'], vm_df['Anomaly'])

        stats = [[str(stat_mse), str(stat_mae), str(stat_ssim), str(stat_anom)],
                 [str(p_mse), str(p_mae), str(p_ssim), str(p_anom)]]
        
        table = plt.table(rowLabels=['U1','P-Value'], colLabels=['MSE', 'MAE', 'SSIM', 'Anomaly'])
        plt.title(view+' Mann-Whitney U Test')

        plt.savefig(self.val_path+view+'_mann-whitman.png')