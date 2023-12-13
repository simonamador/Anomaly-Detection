import torch
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage import exposure
import imutils
import operator
import os
import csv
import nibabel as nib

def normalize(x, a):
    p = np.percentile(x, a)

    num = x-np.min(x)
    den = p-np.min(x)

    out = np.zeros((x.shape[0], x.shape[1]))
    x = np.divide(num, den, out=out, where=den!=0)

    return x.clip(0, 1)

def mask_builder(input, recon, lpips_vgg, device,mask=False):
  
    eq_input = exposure.equalize_adapthist(normalize(input,95))
    eq_recon = exposure.equalize_adapthist(normalize(recon,95))
    dif = abs(eq_input - eq_recon)

    norm95 = normalize(dif, 95)

    input = np.expand_dims(np.expand_dims(normalize(input,95), axis=0), axis=0)
    recon = np.expand_dims(np.expand_dims(normalize(recon,95), axis=0), axis=0)
    input = torch.from_numpy(input).type(torch.float).to(device)
    recon = torch.from_numpy(recon).type(torch.float).to(device)
    
    saliency = lpips_vgg(2*input-1, 2*recon-1, normalize=True).cpu().detach().numpy().squeeze()
    saliency = gaussian_filter(saliency, sigma=1.2)

    m = norm95*(saliency-0.25).clip(0,1)

    if mask == True:
        m95 = np.percentile(m, 95)
        m[m>m95] = 1
        m[m<1] = 0
        f_m = gaussian_filter(m, sigma=1.2)
        f_m[f_m>0.1] = 1
        f_m[f_m<1] = 0
        return f_m, saliency
    else:
        return mask

def resizing(img, target):
    if (img.shape > np.array(target)).any():
        target_shape2 = np.min([target, img.shape],axis=0)
        start = tuple(map(lambda a, da: a//2-da//2, img.shape, target_shape2))
        end = tuple(map(operator.add, start, target_shape2))
        slices = tuple(map(slice, start, end))
        img = img[tuple(slices)]
    offset = tuple(map(lambda a, da: a//2-da//2, target, img.shape))
    slices = [slice(offset[dim], offset[dim] + img.shape[dim]) for dim in range(img.ndim)]
    result = np.zeros(target)
    result[tuple(slices)] = img
    return result

# Dataset generator class. It inputs the dataset path and view, outputs the image given an index.
# performs image extraction according to the view, normalization and convertion to tensor.

class img_dataset(Dataset):
    def __init__(self, root_dir, view, key, data = 'healthy', size: int = 158, horizontal_flip: bool = False, 
                 vertical_flip: bool = False, rotation_angle: int = None):
        self.root_dir = root_dir
        self.view = view
        self.horizontal = horizontal_flip
        self.vertical = vertical_flip
        self.angle = rotation_angle
        self.size = size
        self.key = key
        self.data = data

    def __len__(self):
        if self.view == 'L':
            size = 110
        elif self.view == 'A':
            size = 158
        else:
            size = 126
        return size
    
    def extract_age(self):
        csv_path = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/extract_data.csv'
        id = 'Study ID'
        if self.data == 'vm':
            csv_path = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/ventriculomegaly-data.csv'
            id = 'subject'
        with open(csv_path, 'r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                if row[id] == self.key:
                    ga = float(row['GA'])
        ga = np.expand_dims(ga, axis = 0)
        ga = torch.tensor(ga).type(torch.float)
        return ga
    
    def rotation(self, x, alpha):
        y = x.astype(np.uint8)
        y_rot = imutils.rotate(y, angle = alpha)
        return y_rot.astype(np.float64)

    def normalize_95(self, x):
        p98 = np.percentile(x, 98)
        num = x-np.min(x)
        den = p98-np.min(x)
        out = np.zeros((x.shape[0], x.shape[1]))

        x = np.divide(num, den, out=out, where=den!=0)
        return x.clip(0, 1)

    def __getitem__(self, idx):
        raw = nib.load(self.root_dir).get_fdata()

        ga = self.extract_age()

        if self.view == 'L':
            n_img = resizing(raw[idx,:,:],(self.size,self.size))    
        elif self.view == 'A':
            n_img = resizing(raw[:,idx,:],(self.size,self.size))
        else:
            n_img = resizing(raw[:,:,idx],(self.size,self.size))
    
        n_img = self.normalize_95(n_img)

        if self.horizontal == True:
            n_img = np.flip(n_img,axis=0)

        if self.vertical == True:
            n_img = np.flip(n_img, axis=1)

        if self.angle is not None:
            n_img = self.rotation(n_img, self.angle)

        n_img = np.expand_dims(n_img,axis=0)
        img_torch = torch.from_numpy(n_img.copy()).type(torch.float)

        dict = {'image': img_torch, 'ga': ga}

        return dict

def center_slices(view):
    if view == 'L':
        ids = np.arange(start=40,stop=70)
    elif view == 'A':
        ids = np.arange(start=64,stop=94)
    else:
        ids = np.arange(start=48,stop=78)
    return ids

# Begin the initialization of the datasets. Creates dataset iterativey for each subject and
# concatenates them together for both training and testing datasets (implements img_dataset class).

def data_augmentation(base_set, path, view, key, h, ids):
    transformations = {1: (True, None),
                       2: (False, -10, 10), 3: (True, -10, 10),
                       4: (False, -10, 10), 5: (True, -10, 10),
                       6: (False, -10, 10), 7: (True, -10, 10),
                       8: (False, -10, 10), 9: (True, -10, 10)}
    
    for x, specs in transformations.items():
        aug = img_dataset(path, view, key, size = h, horizontal_flip = specs[0], rotation_angle = specs[1])
        aug = Subset(aug,ids)
        base_set = torch.utils.data.ConcatDataset([base_set, aug])
    return base_set

def loader(source_path, view, batch_size, h):
    train_id = os.listdir(source_path+'train/')
    test_id = os.listdir(source_path+'test/')

    ids = center_slices(view)

    train_set = img_dataset(source_path+'train/'+train_id[0], view, train_id[0][:-4], size = h)
    train_set = Subset(train_set,ids)
    train_set = data_augmentation(train_set, source_path+'train/'+train_id[0], view, 
                                  train_id[0][:-4], h, ids)

    test_set = img_dataset(source_path+'test/'+test_id[0],view, test_id[0][:-4], size = h)
    test_set = Subset(test_set,ids)

    for idx,image in enumerate(train_id):
        if idx != 0:
            train_path = source_path + 'train/' + image
            tr_set = img_dataset(train_path, view, image[:-4], size = h)
            tr_set = Subset(tr_set,ids)
            tr_set = data_augmentation(tr_set, train_path, view, image[:-4], h, ids)
            train_set = torch.utils.data.ConcatDataset([train_set, tr_set])

    for idx,image in enumerate(test_id):
        if idx != 0:
            test_path = source_path + 'test/' + image
            ts_set = img_dataset(test_path,view, image[:-4], size = h)
            ts_set = Subset(ts_set,ids)
            test_set = torch.utils.data.ConcatDataset([test_set, ts_set])

# Dataloaders generated from datasets 
    train_final = DataLoader(train_set, shuffle=True, batch_size=batch_size,num_workers=12)
    val_final = DataLoader(test_set, shuffle=True, batch_size=batch_size,num_workers=12)
    return train_final, val_final

def val_loader(val_path, view, key, data='healthy'):

    ids = center_slices(view)
    val_set = img_dataset(val_path,view, key, data=data)

    val_set = Subset(val_set,ids)

    loader = DataLoader(val_set, batch_size=1)

    return loader