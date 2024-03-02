from math import floor
import os
import random

# Author: @simonamador & @GuillermoTafoya

# The following code conducts the pre-processing of MRI images and saves them as 2d image datasets for training and testing.
# It generates 6 datasets (training and testing for 3 views), and saves them as .npy np arrays in the directory healthy dataset. 
# This dataset is used for the training of an anomaly detection model.

# Path of working directory
path = '/neuro/labs/grantlab/research/MRI_processing/guillermo.tafoya/Anomaly_Detection/main/'
dir_path = path + 'TD_dataset_raw_complete/'

def train_test_splitting():
    # Obtains the number of MRI .nii images in dataset
    images = os.listdir(dir_path+'Raw/')
    n = len(images)

    print(f"Total number of MRI images: {n}")
    random.shuffle(images)

    for id, image in enumerate(images):
        source_path = dir_path + 'Raw/' + image
        
        if not source_path.endswith('native_nuc.nii'): continue
        

        if id < floor(n*.8):
            ds = 'train/'
        else:
            ds = 'test/'

        save_path = dir_path + ds

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        os.system('cp '+source_path+' '+save_path)
        print(f'Copied image {id} of {n}')


# TODO Stratified 

def stratified_train_test_splitting(k_fold=5):
    from sklearn.model_selection import KFold, cross_val_score
    import csv
    
    csv_path = path + 'TD_data.csv'
    images = os.listdir(dir_path+'Raw')
    print(f"Total number of MRI images: {len(images)}")

    for id, image in enumerate(images):
    
        id = 'Study ID'
        ga = None

        if 'recon_native' in image:
            image = image[:image.index('recon_native')-1]

        if 'X' in image:
            image = image[:image.index('X')]
            with open(csv_path, 'r') as csvfile:
                csvreader = csv.DictReader(csvfile)
                for row in csvreader:
                    if row[id] == image:

                        ga = float(row['GA'])
        else:
            with open(csv_path, 'r') as csvfile:
                csvreader = csv.DictReader(csvfile)
                for row in csvreader:
                    if row[id] == image:
                        ga = float(row['GA'])
        if not ga:
            print(f"GA not found for {image}")
            raise Exception(f"GA not found for {image}")  

if __name__ == '__main__':
    train_test_splitting()