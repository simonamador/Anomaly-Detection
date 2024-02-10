# Code written by @GuillermoTafoya

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
from utils.debugging_printers import *

class Visualizer:
    def __init__(self, path, model_path, base, model, view, method, z_dim, name, n, device):

            # Determine if model inputs GA
            self.ga =  base == 'ga_VAE'
            model_name = name + '_' + view
            print(f'{self.ga=}')
            print(f'{z_dim=}')
            

            self.view = view
            self.device = device
            
            prGreen(f'{device=}')

            # Generate and load model
            print(model_path)
            self.model = Framework(n, z_dim, method, device, model, self.ga)
            
            self.model.encoder, self.model.decoder, self.model.refineG = load_model(model_path, base, method, 
                                                                n, n, z_dim, model=model, pre = 'full')
        
            # Visualization paths
            self.hist_path = path+'Results' + model_name + '/history.txt'
            self.vis_path = path+'Results/Visualization/'+model_name+'/'

            if not os.path.exists(self.vis_path):
                os.mkdir(self.vis_path)
                os.mkdir(self.vis_path+'TD/')
                os.mkdir(self.vis_path+'VM/')

            self.vm_path = path+'/VM_dataset/Raw/'
            self.vm_images = os.listdir(self.vm_path)
            self.td_path = path+'TD_dataset/test/'
            self.td_images= os.listdir(self.td_path)

    


    def visualize_age_effect(self, delta_ga=5):
        print('----- BEGINNING VISUALIZATION -----')
        
            
        self.model = self.model.to(self.device)

        loader = val_loader(self.td_path, self.td_images, self.view)

        prev = ''
        reconstructed = 0

        for id, slice in enumerate(loader):
            if self.td_images[int(id/30)][:-4] == prev:
                continue
            prev = self.td_images[int(id/30)][:-4]
            prCyan(f'Working with id={id}')
            img = slice['image'].to(self.device)
            if self.ga:
                for validation in range(3):
                    ga = slice['ga'].to(self.device)
                    ga_copy = ga.clone().detach().cpu().numpy()
                    ga_variation = np.arange(ga_copy - delta_ga, ga_copy + delta_ga + 1, 1)

                    # Define the layout for subplot_mosaic
                    n_images = len(ga_variation)
                    layout = [['recon' + str(i) for i in range(n_images)],
                            ['refine' + str(i) for i in range(n_images)]]

                    # Create the figure with the specified layout
                    fig, axd = plt.subplot_mosaic(layout, figsize=(32, 16), dpi=80)

                    # Set the main title
                    fig.suptitle(f'GA Effect On Reconstruction for {self.td_images[int(id/30)][:-4]} GA({float(ga_copy):.2f})', fontsize=32)

                    
                    # Calculate the top of the subplots for placing the row titles
                    top_of_subplots = max(ax.get_position().ymax for ax in axd.values())

                    # Set the row titles
                    fig.text(0.5, top_of_subplots - 0.1, 'Reconstruction', ha='center', va='bottom', fontsize=26, transform=fig.transFigure)
                    fig.text(0.5, top_of_subplots / 2 - 0.02, 'Refinement', ha='center', va='bottom', fontsize=26, transform=fig.transFigure)

                    # Adjust the layout
                    plt.subplots_adjust(top=top_of_subplots - 0.01)  




                    

                    for idx, ga_val in enumerate(ga_variation):
                        ga_alt = torch.tensor([[ga_val]], dtype=torch.float).to(self.device)
                        recon_ref, rec_dic = self.model(img, ga_alt)

                        recon = np.rot90(rec_dic["x_recon"][0].detach().cpu().numpy().squeeze(), -1)
                        refinement = np.rot90(recon_ref[0].detach().cpu().numpy().squeeze(), -1)
                        

                        # Access the axes using the unique labels we created
                        ax_recon = axd['recon' + str(idx)]
                        ax_refine = axd['refine' + str(idx)]

                        ax_recon.imshow(recon, cmap='gray')
                        ax_refine.imshow(refinement, cmap='gray')

                        # Set GA values as titles for each subplot
                        title = f'{"+" if 0 < ga_val-float(ga_copy) else ""}{ga_val-float(ga_copy): .2f}'
                        ax_recon.set_title(title, fontsize=16)
                        ax_refine.set_title(title, fontsize=16)

                        # Turn off the axes
                        ax_recon.axis('off')
                        ax_refine.axis('off')

                    # Hide the tick labels for the recon images
                    for i in range(n_images):
                        axd['recon' + str(i)].set_xticklabels([])
                        axd['recon' + str(i)].set_yticklabels([])

                    plt.tight_layout(pad=4.0)

                    fig.savefig(self.vis_path+'TD/'+self.td_images[int(id/30)][:-4]+'_'+str(id-30*int(id/30))+'_'+str(validation)+'.png')

                    #plt.show()

                #if id == 0:  # Break after the first image for demonstration purposes
                #    break
                if reconstructed >= 5:  # Remove or modify this condition as needed
                    break
                reconstructed += 1

    def save_reconstruction_images(self, delta_ga=5):
        model = self.model.to(self.device)

        loader = val_loader(self.td_path, self.td_images, self.view)

        prev = ''
        reconstructed = 0

        for id, slice in enumerate(loader):
            if self.td_images[int(id/30)][:-4] == prev:
                continue
            prev = self.td_images[int(id/30)][:-4]
            img = slice['image'].to(self.device)
            ga = slice['ga'].to(self.device)
            ga_copy = ga.clone().detach().cpu().numpy()
            ga_variation = np.arange(ga_copy - delta_ga, ga_copy + delta_ga + 1, 1)

            for ga_val in ga_variation:
                ga_alt = torch.tensor([[ga_val]], dtype=torch.float).to(self.device)
                _, rec_dic = model(img, ga_alt)
                recon = np.rot90(rec_dic["x_recon"][0].detach().cpu().numpy().squeeze(), -1)

                # Save the reconstruction image
                fig, ax = plt.subplots()
                ax.imshow(recon, cmap='gray')
                ax.set_title(f'GA: {ga_val:.2f}')
                ax.axis('off')

                os.makedirs(self.vis_path+'TD/'+self.td_images[int(id/30)][:-4], exist_ok=True)

                fig.savefig(self.vis_path+'TD/'+self.td_images[int(id/30)][:-4]+'/'+str(id-30*int(id/30))+'_'+str(ga_val)+'.png')
                plt.close(fig)  # Close the figure to free memory

                # Optional: Print out a status message
                print(f'Saved Reconstruction image for GA value {ga_val:.2f}')

            if reconstructed >= 3:  # Remove or modify this condition as needed
                break
            reconstructed += 1
