# Code written by @simonamador

from utils.config import load_model, val_loader
from utils.loss import ssim_loss, l1_loss, l2_loss
from models.anomaly import Anomaly
from models.framework import Framework

import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as stts
import seaborn as sns
import pandas as pd
import os
import numpy as np

class Validator:
    def __init__(self, path, model_path, base, model, view, method, z_dim, name, n, device):

        # Determine if model inputs GA
        if base == 'ga_VAE':
            self.ga = True
            model_name = name + '_' + view
        else:
            self.ga = False
            model_name = name + '_' + view

        self.view = view
        self.device = device

        # Generate and load model
        print(model_path)
        self.model = Framework(n, z_dim, method, device, model, self.ga)
        self.model.encoder, self.model.decoder, self.model.refineG = load_model(model_path, base, method, 
                                                            n, n, z_dim, model=model, pre = 'full')
       
        # Validation paths
        self.hist_path = path+'Results' + model_name + '/history.txt'
        self.val_path = path+'Results/Validations/'+model_name+'/'

        if not os.path.exists(self.val_path):
            os.mkdir(self.val_path)
            os.mkdir(self.val_path+'TD/')
            os.mkdir(self.val_path+'VM/')

        self.vm_path = path+'/VM_dataset/Raw/'
        self.vm_images = os.listdir(self.vm_path)
        self.td_path = path+'TD_dataset/test/'
        self.td_images= os.listdir(self.td_path)

    def validation(self,):

        # Create logger
        writer = open(self.val_path+'validation.txt', 'w')
        writer.write('Class, Case_ID, Slide_ID, MAE, MSE, SSIM, Anomaly'+'\n')

        # Load model to device
        self.model = self.model.to(self.device)

        print('----- BEGINNING VALIDATION -----')
        print('.')
        print('.')

        '''
        --------------------- VALIDATION FOR VENTRICULOMEGALY SUBJECTS ---------------------
        '''

        print('-----VM Dataset-----')

        # Validation loader
        loader = val_loader(self.vm_path, self.vm_images, self.view, data='vm')

        # Iterate for each image in loader
        for id, slice in enumerate(loader):
            # Forward through framework
            img = slice['image'].to(self.device)
            if self.ga:
                ga = slice['ga'].to(self.device)
                recon_ref, rec_dic = self.model(img, ga)
            else:
                recon_ref, rec_dic = self.model(img)

            # Extract metrics
            MSE = l2_loss(img, recon_ref).item()
            MAE = l1_loss(img, recon_ref).item()
            SSIM = 1-ssim_loss(img, recon_ref)
            anomap = abs(recon_ref-img)*self.model.anomap.saliency_map(recon_ref,img)

            # Log: Class | Image # | Mean Average Error | Mean Square Error | Structural-Similarity | Anomaly Metric
            writer.write('vm, '+
                                self.vm_images[int(id/30)][:-4]+', '+
                                str(id-30*int(id/30)+1)+', '+
                                str(MAE)+', '+
                                str(MSE)+', '+
                                str(SSIM.item())+', '
                                +str(torch.mean(anomap.flatten()).item())+'\n')
            
            # Visualize images
            images = {"input": img[0][0], "recon": rec_dic["x_recon"][0], "saliency": rec_dic["saliency"][0],
                    "mask": -rec_dic["mask"][0], "ref_recon": recon_ref[0], "anomaly": anomap[0]} 
            fig = self.plot(images, self.vm_images[int(id/30)][:-4], str(id-30*int(id/30)+1), 
                                     [MAE, MSE, SSIM.item(), torch.mean(anomap.flatten()).item()])
            fig.savefig(self.val_path+'VM/'+self.vm_images[int(id/30)][:-4]+'_'+str(id-30*int(id/30))+'.png')
            plt.close()

            # Print current subject
            if id%30 == 0:
                print('Subject {0} of {1}'.format(int(id/30)+1, len(self.vm_images)))
                
        '''
        --------------------- VALIDATION FOR VENTRICULOMEGALY SUBJECTS ---------------------
        '''

        print('-----TD Dataset-----')

        loader = val_loader(self.td_path, self.td_images, self.view)

        # Iterate for each image in loader
        for id, slice in enumerate(loader):
            # Forward through framework
            img = slice['image'].to(self.device)
            if self.ga:
                ga = slice['ga'].to(self.device)
                recon_ref, rec_dic = self.model(img, ga)
            else:
                recon_ref, rec_dic = self.model(img)

            # Extract metrics
            MSE = l2_loss(img, recon_ref).item()
            MAE = l1_loss(img, recon_ref).item()
            SSIM = 1-ssim_loss(img, recon_ref)
            anomap = abs(recon_ref-img)*self.model.anomap.saliency_map(recon_ref,img)

            # Log: Class | Image # | Mean Average Error | Mean Square Error | Structural-Similarity | Anomaly Metric
            writer.write('td, '+
                                self.td_images[int(id/30)][:-4]+', '+
                                str(id-30*int(id/30))+', '+
                                str(MAE)+', '+
                                str(MSE)+', '+
                                str(SSIM.item())+', '
                                +str(torch.mean(anomap.flatten()).item())+'\n')
            
            # Visualize images
            images = {"input": img[0][0], "recon": rec_dic["x_recon"][0], "saliency": rec_dic["saliency"][0],
                    "mask": -rec_dic["mask"][0], "ref_recon": recon_ref[0], "anomaly": anomap[0]} 
            fig = self.plot(images, self.td_images[int(id/30)][:-4], str(id-30*int(id/30)+1), 
                                     [MAE, MSE, SSIM.item(), torch.mean(anomap.flatten()).item()])
            fig.savefig(self.val_path+'TD/'+self.td_images[int(id/30)][:-4]+'_'+str(id-30*int(id/30))+'.png')
            plt.close()
            
            # Print current subject
            if id%30 == 0:
                print('Subject {0} of {1}'.format(int(id/30)+1, len(self.td_images)))

        print('.') 
        writer.close()
        print('.')
        print('Finished validation.')
                
    def plot(self, images, subject, sliceid, metrics):
        fig, axs = plt.subplots(2,3)
        names = [["input", "recon", "ref_recon"], ["saliency", "mask", "anomaly"]]
        cmap_i = ["gray", "hot"]
        for x in range(2):
            for y in range(3):
                if x == 1 and y == 1:
                    cmap_i[1] = "binary"
                if isinstance(images[names[x][y]],np.ndarray):
                    img = images[names[x][y]]
                else:
                    img = images[names[x][y]].detach().cpu().numpy().squeeze()

                axs[x, y].imshow(img, cmap = cmap_i[x])
                axs[x, y].set_title(names[x][y])
                axs[x, y].axis("off")
        axs[1, 2].imshow(images[names[1][2]].detach().cpu().numpy().squeeze(), cmap = 'hot', vmin = 0, vmax = 0.05)
        axs[1, 2].set_title(names[1][2])
        axs[1, 2].axis("off")
        fig.colorbar(cm.ScalarMappable(norm=None, cmap='hot'), ax = axs[1, 2])
        label = subject+', Slice '+sliceid+', MAE: '+str(metrics[0])+', MSE: '+str(metrics[1])+', SSIM: '+str(metrics[2])+', Anomaly: '+str(metrics[3])
        fig.text(0.025, .92, label, fontsize = 6)
        return fig
    
    def tc(self,):
        #Pending (tc = training curves)
        df = pd.read_csv(self.hist_path)

    def severity(self,):
        #Pending dev
        a = 0

    def age_differential(self,):
        # Create logger
        
        writer = open(self.val_path+'ga_validation.txt', 'w')
        writer.write('Case_ID, Slide_ID, GA_dif, MAE'+'\n')

        # Load model to device
        self.model = self.model.to(self.device)

        print('----- BEGINNING VALIDATION -----')
        print('.')
        print('.')

        '''
        --------------------- VALIDATION FOR TD SUBJECTS ---------------------
        '''

        print('-----GA Differential-----')

        loader = val_loader(self.td_path, self.td_images, self.view)

        # Iterate for each image in loader
        for id, slice in enumerate(loader):
            # Forward through framework
            img = slice['image'].to(self.device)
            if self.ga:
                ga = slice['ga'].to(self.device)
                ga_copy = ga.clone().detach().cpu()
                ga_variation = np.arange(ga_copy-5, ga_copy+5, 0.5)
                for ga_alt in ga_variation:
                    ga_alt = torch.tensor(np.expand_dims([ga_alt], axis = 0), dtype = torch.float).to(self.device)
                    z = self.model.encoder(img, ga_alt)
                    recon = self.model.decoder(z)
                    # Extract metrics
                    MAE = l1_loss(img, recon).item()

                    # Log: GA dif | Image # | Mean Square Error 
                    writer.write(self.td_images[int(id/30)][:-4]+', '+
                                str(id-30*int(id/30))+', '+
                                str(ga_alt.item()-ga.item())+', '+
                                str(MAE)+'\n')
            else:
                raise NameError('Cannot validate age difference on a non-ga model.')
            
            # Print current subject
            if id%30 == 0:
                print('Subject {0} of {1}'.format(int(id/30)+1, len(self.td_images)))

        print('.') 
        writer.close()
        print('.')
        print('Finished validation.')

        df = pd.read_csv(self.val_path+'ga_validation.txt', sep=', ')
        plt.scatter(df['GA_dif'], df['MAE'])

        mean = df.groupby('GA_dif').mean(numeric_only=True)['MAE']
        plt.plot([x/10.0 for x in range(-50,50,5)],mean)
        plt.xlabel('GA Difference')
        plt.ylabel('MAE')

        plt.savefig(self.val_path+'ga_error.png')

        print('Saved graphic.')

    def mannwhitneyu(self,):
        # Pending
        print('-----BEGINNING STATS-----')
        print('.')
        print('.')

        df = pd.read_csv(self.val_path+'validation.txt', sep=', ')
        n_df = pd.melt(df, id_vars='Class', value_vars=['MSE', 'MAE', 'Anomaly'],
                      var_name='Metric', value_name='Value')
        td_df = df.where(df['Class']=='td').dropna()
        td_df = df.where(td_df['Slide_ID']==15).dropna()
        vm_df = df.where(df['Class']=='vm').dropna()
        vm_df = df.where(vm_df['Slide_ID']==15).dropna()

        if self.view == 'L':
            view = 'Sagittal'
        elif  self.view == 'A':
            view = 'Coronal'
        else:
            view = 'Axial'

        fig, axs =plt.subplots(2,1)

        stat_mse, p_mse = stts.mannwhitneyu(td_df['MSE'], vm_df['MSE'])
        stat_mae, p_mae = stts.mannwhitneyu(td_df['MAE'], vm_df['MAE'])
        stat_ssim, p_ssim = stts.mannwhitneyu(td_df['SSIM'], vm_df['SSIM'])
        stat_anom, p_anom = stts.mannwhitneyu(td_df['Anomaly'], vm_df['Anomaly'])

        stats = [[str(stat_mse), str(stat_mae), str(stat_ssim), str(stat_anom)],
                 [str(p_mse), str(p_mae), str(p_ssim), str(p_anom)]]
        
        axs[0].axis('tight')
        axs[0].axis('off')
        axs[0].table(cellText=stats, rowLabels=['U1','P-Value'], 
                     colLabels=['MSE', 'MAE', 'SSIM', 'Anomaly'], loc = 'center')

        sns.set_style("whitegrid", {'axes.grid' : False})   
        axs[1] = sns.violinplot(data=n_df, x="Metric", y="Value", hue="Class", split=True, inner="quart")
        
        plt.title(view+' Mann-Whitney U Test')

        plt.savefig(self.val_path+view+'_mann-whitman.png')

        print('Finished stats.')