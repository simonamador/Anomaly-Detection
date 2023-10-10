from math import floor
import os
import numpy as np
import nibabel as nib

# Author: @simonamador

# The following code conducts the pre-processing of MRI images and saves them as 2d image datasets for training and testing.
# It generates 6 datasets (training and testing for 3 views), and saves them as .npy np arrays in the directory healthy dataset. 
# This dataset is used for the training of an anomaly detection model.

# Normalization function: Conducts min-max normalization of the images
def norm(img):
    n_img = (img-np.min(img))/(np.max(img)-np.min(img))
    n_img *= 255
    return n_img

# Path of working directory
dir_path = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/healthy_dataset/'

# Generate directories for saving training and testing datasets in each view 
os.system('mkdir ' + dir_path + 'L_view_e')
os.system('mkdir ' + dir_path + 'A_view_e')
os.system('mkdir ' + dir_path + 'S_view_e')

# Obtains the number of MRI .nii images in dataset
n = os.listdir(dir_path+'recon_img')
print(f"Total number of MRI images: {len(n)}")

# Generates lists for the views (left, axial, sagittal)
L = []
A = []
S = []

# Loops for each MRI image, extract each slice and appends them in their corresponding view list.
# Assumes the first dimmension corresponds to the left view, second to the axial, third to the superior.
# Assumes the shape of the MRI images.
for subject in n:
    raw = nib.load(dir_path + 'recon_img/' + subject)
    mri = raw.get_fdata()
    print('-'*15)
    print(f"MRI {subject} loaded correctly.")
    for i in np.linspace(0,mri.shape[1]-1,mri.shape[1]):
        img = mri[:110,int(i),:]
        img = norm(img)
        if np.sum(img) > 5:
            A.append(img)
        if i < 125:
            img = mri[:110,:158,int(i)]
            img = norm(img)
            if np.sum(img) > 5:
                S.append(img)
            if i < 110:
                img = mri[int(i),:158,:]
                img = norm(img)
                if np.sum(img) > 5:
                    L.append(img)
    print('All slices extracted correctly')

    
# Conducts random permutation on the list to ensure randomized data, then partition in an 80-20 proportion for
# training and testing in each view, saves the datasets.

print('-'*15)
print('All MRIs extracted. Conducting train-test parition:')
print('.'*20)

id_L = np.random.permutation(len(L))
id_A = np.random.permutation(len(A))
id_S = np.random.permutation(len(S))

n_L = np.stack(L,axis=0)
n_A = np.stack(A,axis=0)
n_S = np.stack(S,axis=0)

L_train = n_L[id_L[:floor(.8*len(L))]]
np.save(dir_path+'L_view_e/train.npy',L_train)
L_test  = n_L[id_L[floor(.8*len(L)):]]
np.save(dir_path+'L_view_e/test.npy',L_test)
print('Left view saved.')

A_train = n_A[id_A[:floor(.8*len(A))]]
np.save(dir_path+'A_view_e/train.npy',A_train)
A_test  = n_A[id_A[floor(.8*len(A)):]]
np.save(dir_path+'A_view_e/test.npy',A_test)
print('Axial view saved.')

S_train = n_S[id_S[:floor(.8*len(S))]]
np.save(dir_path+'S_view_e/train.npy',S_train)
S_test  = n_S[id_S[floor(.8*len(S)):]]
np.save(dir_path+'S_view_e/test.npy',S_test)
print('Sagittal view saved.')