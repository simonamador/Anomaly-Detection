from math import floor
import os
import random

# Author: @simonamador

# The following code conducts the pre-processing of MRI images and saves them as 2d image datasets for training and testing.
# It generates 6 datasets (training and testing for 3 views), and saves them as .npy np arrays in the directory healthy dataset. 
# This dataset is used for the training of an anomaly detection model.

# Path of working directory
dir_path = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/healthy_dataset/'

# Obtains the number of MRI .nii images in dataset
images = os.listdir(dir_path+'Raw')
n = len(images)

print(f"Total number of MRI images: {n}")
random.shuffle(images)

for id, image in enumerate(images):
    source_path = dir_path + 'Raw/' + image

    if id < floor(n*.8):
        ds = 'train/'
    else:
        ds = 'test/'

    save_path = dir_path + ds

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    os.system('cp '+source_path+' '+save_path)
    print(f'Copied image {id} of {n}')