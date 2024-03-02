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
    def __init__(self, path, model_path, base, model, view, method, z_dim, name, n, device, training_folder, ga_n, raw):

        # Determine if model inputs GA
        if base == 'ga_VAE':
            self.ga = True
            model_name = name + '_' + view
        else:
            self.ga = False
            model_name = name + '_' + view

        self.view = view
        self.device = device
        self.raw = raw

        # Generate and load model
        print(model_path)
        self.model = Framework(n, z_dim, method, device, model, self.ga, ga_n=ga_n)
        self.model.encoder, self.model.decoder, self.model.refineG = load_model(model_path, base, method, 
                                                            n, n, z_dim, model=model, pre = 'full', ga_n=ga_n)
       
        # Validation paths
        self.hist_path = path+'Results' + model_name + '/history.txt'
        self.val_path = path+'Results/Validations/'+model_name+'/'

        if not os.path.exists(self.val_path):
            os.mkdir(self.val_path)
            os.mkdir(self.val_path+'TD/')
            os.mkdir(self.val_path+'VM/')

        self.vm_path = path+ ('/VM_dataset/Raw/' if not raw else '/VM_dataset_raw/Raw/') #path+'/VM_dataset/Raw/'
        self.vm_images = os.listdir(self.vm_path)
        self.td_path = path+training_folder+'test/'
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
        loader = val_loader(self.vm_path, self.vm_images, self.view, data='vm', raw = self.raw)

        # Iterate for each image in loader
        for id, slice in enumerate(loader):
            # Forward through framework
            img = slice['image'].to(self.device)
            if self.ga:
                ga = slice['ga'].to(self.device)
                #if ga != 29.29:
                #    continue
                
                recon_ref, rec_dic = self.model(img, ga)
            else:
                recon_ref, rec_dic = self.model(img)

            # Extract metrics
            MSE = l2_loss(img, recon_ref).item()
            MAE = l1_loss(img, recon_ref).item()
            SSIM = 1-ssim_loss(img, recon_ref)
            anomap0 = abs(rec_dic["x_recon"]-img)*self.model.anomap.saliency_map(rec_dic["x_recon"],img)
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
            images = {"input": img[0][0], "recon": rec_dic["x_recon"][0], "coarse_anomaly": anomap0,"saliency": rec_dic["saliency"][0],
                    "mask": -rec_dic["mask"][0], "ref_recon": recon_ref[0], "anomaly": anomap[0]} 
            fig = self.plot(images, self.vm_images[int(id/30)][:-4], str(id-30*int(id/30)+1), 
                                     [MAE, MSE, SSIM.item(), torch.mean(anomap.flatten()).item()])
            fig.savefig(self.val_path+'VM/'+self.vm_images[int(id/30)][:-4]+'_'+str(id-30*int(id/30))+'.png')
            plt.close()

            # Print current subject
            if id%30 == 0:
                print('Subject {0} of {1}'.format(int(id/30)+1, len(self.vm_images)))
                
        
        #exit(0)
        '''
        --------------------- VALIDATION FOR TD SUBJECTS ---------------------
        '''

        print('-----TD Dataset-----')

        loader = val_loader(self.td_path, self.td_images, self.view, raw = self.raw)

        # Iterate for each image in loader
        for id, slice in enumerate(loader):
            # Forward through framework
            img = slice['image'].to(self.device)
            if self.ga:
                ga = slice['ga'].to(self.device)
                #if ga != 30.57:
                #    continue
                recon_ref, rec_dic = self.model(img, ga)
            else:
                recon_ref, rec_dic = self.model(img)

            # Extract metrics
            MSE = l2_loss(img, recon_ref).item()
            MAE = l1_loss(img, recon_ref).item()
            SSIM = 1-ssim_loss(img, recon_ref)
            anomap0 = abs(rec_dic["x_recon"]-img)*self.model.anomap.saliency_map(rec_dic["x_recon"],img)
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
            images = {"input": img[0][0], "recon": rec_dic["x_recon"][0], "coarse_anomaly": anomap0,"saliency": rec_dic["saliency"][0],
                    "mask": -rec_dic["mask"][0], "ref_recon": recon_ref[0], "anomaly": anomap[0]} 
            fig = self.plot(images,
                             self.td_images[int(id/30)][:-4],
                             str(id-30*int(id/30)+1), 
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

    def GA_testing(self):
        pass
                
   

    def plot(self, images, subject, sliceid, metrics):
        def rotate(image):
            return np.rot90(image -1)
        
        fig, axs = plt.subplots(2,3, figsize=(20, 14))
        names = [["input", "recon", "ref_recon"], ["input", "input", "input"]]
        labels = [["Input MRI", "VAE Recon", "AOT-GAN Inpainting"], ["Saliency Map", "Mask", "Anomaly Map"]]
        overlays = ["", "", "", "saliency", "mask", "anomaly"]
        cmap_i = ["gray", "gray", "gray", "hot", "Blues", "hot"]
        alpha_values = [1, 1, 1, 0.8, 0.5, 0.9]

        for x in range(2):
            for y in range(3):
                base_img = images[names[x][y]].detach().cpu().numpy().squeeze()
                axs[x, y].imshow(rotate(base_img), cmap='gray')
                axs[x, y].set_title(labels[x][y], fontsize=34)
                axs[x, y].axis("off")
                
                if overlays[x*3 + y]:
                    overlay_img = rotate(images[overlays[x*3 + y]].detach().cpu().numpy().squeeze())
                    # Create a masked array where the mask is applied for zero values
                    masked_overlay = np.ma.masked_where(overlay_img == 0, overlay_img)
                    cmap = plt.cm.get_cmap(cmap_i[x*3 + y])
                    cmap.set_bad(alpha=0)  # Set the 'bad' values (masked) to be transparent
                    axs[x, y].imshow(masked_overlay, cmap=cmap, alpha=alpha_values[x*3 + y])

        #label = f'{subject}, Slice {sliceid}, MAE: {metrics[0]:.4f}, MSE: {metrics[1]:.4f}, SSIM: {metrics[2]:.4f}, Anomaly: {metrics[3]:.4f}'
        #fig.text(0.05, 1, label, fontsize=12)
        fig.subplots_adjust(hspace=0.3) 
        plt.tight_layout()
        #plt.subplots_adjust()
        return fig
    
    def tc(self,):
        #Pending (tc = training curves)
        df = pd.read_csv(self.hist_path)

    def severity(self,):
        #Pending dev
        a = 0

    def age_differential(self,delta_ga=10):
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

        loader = val_loader(self.td_path, self.td_images, self.view, raw = self.raw)

        # Iterate for each image in loader
        for id, slice in enumerate(loader):
            # Forward through framework
            img = slice['image'].to(self.device)
            if self.ga:
                ga = slice['ga'].to(self.device)
                ga_copy = ga.clone().detach().cpu()
                
                #if ga_copy < 30:
                #    continue
                
                ga_variation = np.arange(ga_copy-delta_ga, ga_copy+delta_ga+0.5, 0.5)
                
                
                for ga_alt in ga_variation:
                    ga_alt = torch.tensor(np.expand_dims([ga_alt], axis = 0), dtype = torch.float).to(self.device)
                    z = self.model.encoder(img, ga_alt)
                    recon = self.model.decoder(z)
                    # Extract metrics
                    MAE = l1_loss(img, recon).item()

                    # Log: GA dif | Image # | Mean Square Error 
                    writer.write(self.td_images[int(id/30)][:-4]+', '+
                                str(id-30*int(id/30))+', '+
                                f'{ga_alt.item()-ga.item():.1f}' +', '+
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

        mean = df.groupby('GA_dif').mean(numeric_only=True)['MAE']
        print(f'{mean=}')
        plt.plot([x/10.0 for x in range(-(delta_ga*10),delta_ga*10+5,5)],mean, color='orange')
        plt.xlabel('GA Difference')
        plt.ylabel('MAE')

        plt.savefig(self.val_path+f'ga_error_{delta_ga}.png')
        plt.close()

        
        plt.scatter(df['GA_dif'], df['MAE'])

        plt.plot([x/10.0 for x in range(-(delta_ga*10),delta_ga*10+5,5)],mean, color='orange')
        plt.xlabel('GA Difference')
        plt.ylabel('MAE')

        plt.savefig(self.val_path+f'ga_error_Dist_{delta_ga}.png')
        plt.close()

        print('Saved graphic.')

    def mannwhitneyu(self):
        # Existing code
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

        numeric_td_df = td_df.select_dtypes(include=[np.number])
        numeric_vm_df = vm_df.select_dtypes(include=[np.number])

        # Compute statistics
        td_means = numeric_td_df.mean()
        td_stds = numeric_td_df.std()
        vm_means = numeric_vm_df.mean()
        vm_stds = numeric_vm_df.std()

        # Mann-Whitney U Test for your metrics
        stat_mse, p_mse = stts.mannwhitneyu(td_df['MSE'], vm_df['MSE'])
        stat_mae, p_mae = stts.mannwhitneyu(td_df['MAE'], vm_df['MAE'])
        stat_ssim, p_ssim = stts.mannwhitneyu(td_df['SSIM'], vm_df['SSIM'])
        stat_anom, p_anom = stts.mannwhitneyu(td_df['Anomaly'], vm_df['Anomaly'])

        stats = [[str(stat_mse), str(stat_mae), str(stat_ssim), str(stat_anom)],
                [str(p_mse), str(p_mae), str(p_ssim), str(p_anom)],
                [str(td_means['MSE']), str(td_means['MAE']), str(td_means['SSIM']), str(td_means['Anomaly'])],
                [str(td_stds['MSE']), str(td_stds['MAE']), str(td_stds['SSIM']), str(td_stds['Anomaly'])],
                [str(vm_means['MSE']), str(vm_means['MAE']), str(vm_means['SSIM']), str(vm_means['Anomaly'])],
                [str(vm_stds['MSE']), str(vm_stds['MAE']), str(vm_stds['SSIM']), str(vm_stds['Anomaly'])]]

        # Plotting code
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))
        axs[0].axis('tight')
        axs[0].axis('off')
        axs[0].table(cellText=stats, rowLabels=['U1', 'P-Value', 'TD-Mean', 'TD-SD', 'VM-Mean', 'VM-SD'], 
                    colLabels=['MSE', 'MAE', 'SSIM', 'Anomaly'], loc='center')

        sns.set_style("whitegrid", {'axes.grid': False})
        axs[1] = sns.violinplot(data=n_df, x="Metric", y="Value", hue="Class", split=True, inner="quart")
        if self.view == 'L':
            view = 'Sagittal'
        elif  self.view == 'A':
            view = 'Coronal'
        else:
            view = 'Axial'
        plt.title(view + ' Mann-Whitney U Test')
        plt.savefig(self.val_path + view + '_mann-whitman.png')

        print('Finished stats.')

    # AUROC Computing
    def AUROC(self):
        from sklearn.metrics import roc_auc_score, roc_curve, auc #accuracy_score, confusion_matrix, roc_auc_score, roc_curve
        from scipy import interp
        from numpy import linspace
        print('-----BEGINNING AUROC-----')
        print()
        print()
        

        # Assuming 'self.val_path' contains the path to your 'validation.txt' file
        # And 'self.view' contains the view type ('L' for Sagittal, 'A' for Coronal, etc.)
        df = pd.read_csv(self.val_path + 'validation.txt', sep=', ')

        # Filter for Slide_ID 15
        td_df = df[(df['Class'] == 'td') & (df['Slide_ID'] == 15)]
        vm_df = df[(df['Class'] == 'vm') & (df['Slide_ID'] == 15)]

        # Define view based on self.view
        if self.view == 'L':
            view = 'Sagittal'
        elif self.view == 'A':
            view = 'Coronal'
        else:
            view = 'Axial'

        # Assuming 'Anomaly' scores are what you want to use for AUROC calculation
        # Prepare labels and scores
        y_true = np.concatenate((np.zeros(len(td_df)), np.ones(len(vm_df))))
        scores = np.concatenate((td_df['Anomaly'].values, vm_df['Anomaly'].values))

        # Compute AUROC
        auroc = roc_auc_score(y_true, scores)
        print(f"{view} AUROC: {auroc}")

        # Generate and plot ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        # This will create a high-resolution set of false positive rates
        base_fpr = linspace(0, 1, 101)

        # Interpolate the true positive rates at these points
        smooth_tpr = interp(base_fpr, fpr, tpr)
        smooth_tpr[0] = 0.0  # Ensuring the curve starts at 0

        # Plot the smoothed ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(base_fpr, smooth_tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1 - Specificity')
        plt.ylabel('Sensitivity')
        plt.title(F'Receiver Operating Characteristic for {view} View')
        plt.legend(loc="lower right")
        
        plt.savefig(self.val_path+view+'_AUROC.png')
        #plt.show()

    def multiview_AUROC_AS(self):
        from sklearn.metrics import roc_auc_score, roc_curve, auc
        from scipy import interp
        from numpy import linspace, mean
        
        base_path = self.val_path[:-2]  
        print(f'{base_path=}')

        # Load validation data for both views
        df_S = pd.read_csv(f'{base_path}S/validation.txt', sep=', ')
        df_A = pd.read_csv(f'{base_path}A/validation.txt', sep=', ')

        # Filter for Slide_ID 15 for both views
        td_df_S = df_S[(df_S['Class'] == 'td') & (df_S['Slide_ID'] == 15)]
        vm_df_S = df_S[(df_S['Class'] == 'vm') & (df_S['Slide_ID'] == 15)]
        td_df_A = df_A[(df_A['Class'] == 'td') & (df_A['Slide_ID'] == 15)]
        vm_df_A = df_A[(df_A['Class'] == 'vm') & (df_A['Slide_ID'] == 15)]

        # Ensure the data is aligned if necessary (e.g., by sorting by an identifier if the rows don't match up)

        # Combine anomaly scores from both views by averaging
        scores_L = np.concatenate((td_df_S['Anomaly'].values, vm_df_S['Anomaly'].values))
        scores_A = np.concatenate((td_df_A['Anomaly'].values, vm_df_A['Anomaly'].values))
        multiview_scores = (scores_L + scores_A) / 2

        # Prepare labels (same for both views)
        y_true = np.concatenate((np.zeros(len(td_df_S)), np.ones(len(vm_df_S))))  # Assuming both have same length

        # Compute AUROC for multiview scores
        multiview_auroc = roc_auc_score(y_true, multiview_scores)
        print(f"Multiview AUROC (Axial & Coronal): {multiview_auroc}")

        # Generate and plot ROC curve for multiview scores
        fpr, tpr, thresholds = roc_curve(y_true, multiview_scores)
        base_fpr = linspace(0, 1, 101)
        smooth_tpr = interp(base_fpr, fpr, tpr)
        smooth_tpr[0] = 0.0

        plt.figure(figsize=(8, 6))
        plt.plot(base_fpr, smooth_tpr, color='darkorange', lw=2, label=f'ROC curve (area = {multiview_auroc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1 - Specificity')
        plt.ylabel('Sensitivity')
        plt.title('Receiver Operating Characteristic for Coronal-Axial Multiview')
        plt.legend(loc="lower right")
        plt.savefig(f'{base_path}Axial_Coronal_Multiview_AUROC.png')


    def multiview_AUROC_ASL(self):

        from sklearn.metrics import roc_auc_score, roc_curve, auc
        from scipy import interp
        from numpy import linspace, mean
        import numpy as np  # Ensure numpy is imported
        import pandas as pd  # Ensure pandas is imported
        import matplotlib.pyplot as plt  # Ensure matplotlib is imported
        
        base_path = self.val_path[:-2]  # Assume the path ends with 'L/', 'A/', or 'S/'

        # Load validation data for all three views
        df_L = pd.read_csv(f'{base_path}L/validation.txt', sep=', ')
        df_A = pd.read_csv(f'{base_path}A/validation.txt', sep=', ')
        df_S = pd.read_csv(f'{base_path}S/validation.txt', sep=', ')

        # Filter for Slide_ID 15 for all views
        td_df_L = df_L[(df_L['Class'] == 'td') & (df_L['Slide_ID'] == 15)]
        vm_df_L = df_L[(df_L['Class'] == 'vm') & (df_L['Slide_ID'] == 15)]
        td_df_A = df_A[(df_A['Class'] == 'td') & (df_A['Slide_ID'] == 15)]
        vm_df_A = df_A[(df_A['Class'] == 'vm') & (df_A['Slide_ID'] == 15)]
        td_df_S = df_S[(df_S['Class'] == 'td') & (df_S['Slide_ID'] == 15)]
        vm_df_S = df_S[(df_S['Class'] == 'vm') & (df_S['Slide_ID'] == 15)]

        # Combine anomaly scores from all three views by averaging
        scores_L = np.concatenate((td_df_L['Anomaly'].values, vm_df_L['Anomaly'].values))
        scores_A = np.concatenate((td_df_A['Anomaly'].values, vm_df_A['Anomaly'].values))
        scores_S = np.concatenate((td_df_S['Anomaly'].values, vm_df_S['Anomaly'].values))
        multiview_scores = (scores_L + scores_A + scores_S) / 3

        # Prepare labels (assumed same for all views)
        y_true = np.concatenate((np.zeros(len(td_df_L)), np.ones(len(vm_df_L))))  # Assuming all have same length

        # Compute AUROC for multiview scores
        multiview_auroc = roc_auc_score(y_true, multiview_scores)
        print(f"Multiview AUROC (Sagittal, Coronal, & Axial): {multiview_auroc}")

        # Generate and plot ROC curve for multiview scores
        fpr, tpr, thresholds = roc_curve(y_true, multiview_scores)
        base_fpr = linspace(0, 1, 101)
        smooth_tpr = interp(base_fpr, fpr, tpr)
        smooth_tpr[0] = 0.0

        plt.figure(figsize=(8, 6))
        plt.plot(base_fpr, smooth_tpr, color='darkorange', lw=2, label=f'ROC curve (area = {multiview_auroc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1 - Specificity')
        plt.ylabel('Sensitivity')
        plt.title('Receiver Operating Characteristic for ASL Multiview')
        plt.legend(loc="lower right")
        plt.savefig(f'{base_path}Multiview_AUROC.png')

    # TODO Cross Validation

    def crossValidation():
        from sklearn.model_selection import KFold, cross_val_score 
        pass
        