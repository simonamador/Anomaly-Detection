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
    def __init__(self, path, model_path, base, model, view, method, z_dim, name, n, device, training_folder, ga_n, raw, th = 99, cGAN = False):

            # Determine if model inputs GA
            self.ga =  base == 'ga_VAE'
            model_name = name + '_' + view
            print(f'{self.ga=}')
            print(f'{z_dim=}')
            

            self.view = view
            self.device = device
            self.raw = raw
            
            prGreen(f'{device=}')

            # Generate and load model
            print(model_path)
            self.model = Framework(n, z_dim, method, device, model, self.ga, ga_n, th = th)
            
            self.model.encoder, self.model.decoder, self.model.refineG = load_model(model_path, base, method, 
                                                                n, n, z_dim, model=model, pre = 'full', ga_n=ga_n)
        
            # Visualization paths
            self.hist_path = path+'Results' + model_name + '/history.txt'
            self.vis_path = path+'Results/Visualization/'+model_name+'/'

            if not os.path.exists(self.vis_path):
                os.mkdir(self.vis_path)
                os.mkdir(self.vis_path+'TD/')
                os.mkdir(self.vis_path+'VM/')

            self.vm_path = path+ ('/VM_dataset/Raw/' if not raw else '/VM_dataset_raw/Raw/')
            self.vm_images = os.listdir(self.vm_path)
            self.td_path = path+training_folder+'/test/'
            self.td_images= os.listdir(self.td_path)

    


    def visualize_age_effect(self, delta_ga=5):
        print('----- BEGINNING VISUALIZATION -----')
        
            
        self.model = self.model.to(self.device)

        loader = val_loader(self.td_path, self.td_images, self.view, raw = self.raw)

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
                    ### GA Conditioning ###
                    # if ga > 25:
                    #     continue
                    ga_variation = np.arange(ga_copy - delta_ga, ga_copy + delta_ga + 1, 1)

                    # Define the layout for subplot_mosaic
                    n_images = len(ga_variation)
                    layout = [['recon' + str(i) for i in range(n_images)],
                            ['refine' + str(i) for i in range(n_images)]]

                    # Create the figure with the specified layout
                    fig, axd = plt.subplot_mosaic(layout, figsize=(64, 32), dpi=80)

                    # Set the main title
                    fig.suptitle(f'GA Effect On Reconstruction for {self.td_images[int(id/30)][:-4]} GA({float(ga_copy):.2f})', fontsize=64)

                    
                    # Calculate the top of the subplots for placing the row titles
                    top_of_subplots = max(ax.get_position().ymax for ax in axd.values())

                    # Set the row titles
                    fig.text(0.5, top_of_subplots - 0.1, 'Reconstruction', ha='center', va='bottom', fontsize=48, transform=fig.transFigure)
                    fig.text(0.5, top_of_subplots / 2 - 0.02, 'Refinement', ha='center', va='bottom', fontsize=48, transform=fig.transFigure)

                    # Adjust the layout
                    plt.subplots_adjust(top=top_of_subplots - 0.01)  


                    for idx, ga_val in enumerate(ga_variation):
                        ga_alt = torch.tensor([[ga_val]], dtype=torch.float).to(self.device)
                        recon_ref, rec_dic = self.model(img, ga_alt)

                        # recon = rec_dic["x_recon"][0].detach().cpu().numpy().squeeze() #np.rot90(rec_dic["x_recon"][0].detach().cpu().numpy().squeeze(), -1)
                        # refinement = recon_ref[0].detach().cpu().numpy().squeeze() #np.rot90(recon_ref[0].detach().cpu().numpy().squeeze(), -1)
                        recon = np.rot90(rec_dic["x_recon"][0].detach().cpu().numpy().squeeze(), 1)
                        refinement = np.rot90(recon_ref[0].detach().cpu().numpy().squeeze(), 1)
                        

                        # Access the axes using the unique labels we created
                        ax_recon = axd['recon' + str(idx)]
                        ax_refine = axd['refine' + str(idx)]

                        ax_recon.imshow(recon, cmap='gray')
                        ax_refine.imshow(refinement, cmap='gray')

                        # Set GA values as titles for each subplot
                        title = f'{"+" if 0 < ga_val-float(ga_copy) else ""}{ga_val-float(ga_copy): .2f}'
                        ax_recon.set_title(title, fontsize=32)
                        ax_refine.set_title(title, fontsize=32)

                        # Turn off the axes
                        ax_recon.axis('off')
                        ax_refine.axis('off')

                    # Hide the tick labels for the recon images
                    for i in range(n_images):
                        axd['recon' + str(i)].set_xticklabels([])
                        axd['recon' + str(i)].set_yticklabels([])

                    plt.tight_layout(pad=4.0)

                    fig.savefig(self.vis_path+'TD/'+self.td_images[int(id/30)][:-4]+'_'+str(id-30*int(id/30))+'_'+str(validation)+'.png')
                    plt.close(fig)  # Close the figure to free memory

                    #plt.show()

                #if id == 0:  # Break after the first image for demonstration purposes
                #    break
                if reconstructed >= 15:  # Remove or modify this condition as needed
                    break
                reconstructed += 1


    def find_nonzero_bounding_box(slice_2d, percentile=98, file_name='slice_with_bbox.png'):
        import matplotlib.patches as patches
        def rotate(image):
            return np.rot90(image, 1)
    
        slice_2d = rotate(slice_2d)

        # Determine the threshold based on the specified percentile of the non-zero values
        threshold = np.percentile(slice_2d[slice_2d > 0], percentile)
        
        # Apply the threshold to create a binary array: True for values above the threshold, False otherwise
        binary_slice = slice_2d > threshold

        # Find the indices of the True elements
        rows, cols = np.nonzero(binary_slice)
        if len(rows) == 0 or len(cols) == 0:  # If the slice is effectively empty after thresholding
            return 0, 0  # No bounding box

        # Calculate the bounding box dimensions
        min_col, max_col = cols.min(), cols.max()
        min_row, max_row = rows.min(), rows.max()
        width = max_col - min_col + 1
        height = max_row - min_row + 1

        # Plotting
        fig, ax = plt.subplots()
        ax.imshow(slice_2d, cmap='gray')
        # Add a rectangle patch for the bounding box
        rect = patches.Rectangle((min_col, min_row), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        voxel_size = 0.859375
        plt.title(f'Bounding Box: {width*voxel_size}x{height*voxel_size} mm (width x height)')  # Display the size of the bounding box
        plt.savefig(file_name)  # Save the figure
        plt.close(fig)  # Close the figure to free up memory

        return width, height
    
    # find_nonzero_bounding_box_3(normalized_slice, percentile=80, file_name=f'{case}_central_{view}_normalized_slice_with_bbox.png')

    def save_reconstruction_images(self, delta_ga=10):
        model = self.model.to(self.device)

        loader = val_loader(self.td_path, self.td_images, self.view, raw = self.raw)

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
                recon = np.rot90(rec_dic["x_recon"][0].detach().cpu().numpy().squeeze(), 1)
                #recon = rec_dic["x_recon"][0].detach().cpu().numpy().squeeze()

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

            if reconstructed >= 15:  # Remove or modify this condition as needed
                break
            reconstructed += 1

    def save_whole_range_plus_refined(self, TD = True):
        model = self.model.to(self.device)
        path = self.td_path if TD else self.vm_path
        images = self.td_images if TD else self.vm_images

        loader = val_loader(path, images, self.view, raw = self.raw, data='healthy' if TD else 'VM')

        prev = ''
        reconstructed = 0

        for id, slice in enumerate(loader):
            if images[int(id / 30)][:-4] == prev:
                continue
            prev = images[int(id / 30)][:-4]
            img = slice['image'].to(self.device)
            original_img = np.rot90(img.detach().cpu().numpy().squeeze(), 1)  # Assuming img is a single-channel image
            ga = slice['ga'].to(self.device)
            original_ga = ga.clone().detach().cpu().numpy().item()  # Get original GA as a float
            ga_variation = np.arange(20, 41, 1)  # Original range of gestational ages

            # Append the original GA to the range and ensure all values are unique
            ga_variation = np.unique(np.append(ga_variation, original_ga))

            for ga_val in ga_variation:
                ga_alt = torch.tensor([[ga_val]], dtype=torch.float).to(self.device)
                recon_ref, rec_dic = model(img, ga_alt)

                # Extracting the images from the model output
                refined_recon = np.rot90(recon_ref.detach().cpu().numpy().squeeze(), 1)
                vae_recon = np.rot90(rec_dic["x_recon"][0].detach().cpu().numpy().squeeze(), 1)
                saliency_map_raw = np.rot90(rec_dic["saliency"][0].detach().cpu().numpy().squeeze(), 1)
                refinement_mask = np.rot90(-rec_dic["mask"][0].detach().cpu().numpy().squeeze(), 1)
                anomap_vae = np.rot90(abs(rec_dic["x_recon"]-img).detach().cpu().numpy().squeeze()* self.model.anomap.saliency_map(rec_dic["x_recon"], img).detach().cpu().numpy().squeeze(), 1) 
                saliency_map_refined = self.model.anomap.saliency_map(recon_ref, img).detach().cpu().numpy().squeeze()
            
                anomap_refined = abs(recon_ref-img).detach().cpu().numpy().squeeze() * saliency_map_refined

                # Create a figure with subplots for both rows
                fig, axs = plt.subplots(3, 3, figsize=(15, 15))

                # First row: Original, VAE reconstructed, Refined reconstructed
                axs[0, 0].imshow(original_img, cmap='gray')
                axs[0, 0].set_title(f'Original GA: {original_ga:.2f}')

                axs[0, 1].imshow(vae_recon, cmap='gray')
                axs[0, 1].set_title(f'VAE GA: {ga_val:.2f}')

                axs[0, 2].imshow(refined_recon, cmap='gray')
                axs[0, 2].set_title(f'Refined GA: {ga_val:.2f}')

                # Second row
                axs[1, 0].imshow(original_img, cmap='gray')  # Base image for saliency map
                axs[1, 0].imshow(saliency_map_raw, cmap='hot', alpha=0.8)  # Saliency map overlaid
                axs[1, 0].set_title(f'Saliency map VAE')

                axs[1, 1].imshow(original_img, cmap='gray')  # Base image for anomaly map
                axs[1, 1].imshow(np.rot90(saliency_map_refined, 1), cmap='hot', alpha=0.8)  # Anomaly map overlaid
                axs[1, 1].set_title(f'Saliency map refined')

                axs[1, 2].imshow(original_img, cmap='gray')  # Base image for refinement mask
                axs[1, 2].imshow(refinement_mask, cmap='Blues', alpha=0.5)  # Refinement mask overlaid
                axs[1, 2].set_title(f'Refining mask')

                # Third row
                axs[2, 0].imshow(original_img, cmap='gray')  # Base image for saliency map
                axs[2, 0].imshow(anomap_vae, cmap='hot', alpha=0.9)  # Saliency map overlaid
                axs[2, 0].set_title(f'Anomaly map VAE')
                
                axs[2, 1].imshow(original_img, cmap='gray')  # Base image for refinement mask
                axs[2, 1].imshow(np.rot90(anomap_refined, 1), cmap='hot', alpha=0.9)  # Refinement mask overlaid
                axs[2, 1].set_title(f'Anomaly map refined')

                #axs[2, 2].imshow(original_img, cmap='gray')  # Base image for refinement mask
                #axs[2, 2].imshow(np.rot90(anomap_refined, 1), cmap='hot', alpha=0.9)  # Refinement mask overlaid
                #axs[2, 2].set_title(f'Anomaly map refined')

                MSE = l2_loss(img, recon_ref).item()
                MAE = l1_loss(img, recon_ref).item()
                SSIM = 1-ssim_loss(img, recon_ref)
                anomaly_metric = torch.mean(torch.tensor(anomap_refined)).item()
                
                x = 0.7
                y = 0.11
                fig.text(x, y, f'{MSE=:.4f}', fontsize=15)
                fig.text(x, y+0.02, f'{MAE=:.4f}', fontsize=15)
                fig.text(x, y+0.04, f'{SSIM=:.4f}', fontsize=15)
                fig.text(x, y+0.06, f'{anomaly_metric=:.4f}', fontsize=15)

                for i in range(3):
                    for j in range(3):
                        axs[i, j].axis('off')  # Remove axes for all plots

                # Adjust subplot parameters and save the figure
                fig.subplots_adjust(hspace=0.1, wspace=0.1)
                folder_path = self.vis_path + ('TD/' if TD else 'VM/') + images[int(id / 30)][:-4]
                os.makedirs(folder_path, exist_ok=True)
                comparison_filename = f'{str(id - 30 * int(id / 30))}_{str(ga_val)}.png'
                fig.savefig(os.path.join(folder_path, comparison_filename))
            
                plt.close(fig)  # Close the figure to free memory

                # Optional: Print out a status message
                print(f'Saved multi-image comparison for GA value {ga_val:.2f}')

            print(f'Processed {folder_path}')
            if reconstructed >= 14:  
                break
            reconstructed += 1