import numpy as np
from model import Encoder, Decoder
import torch
from torch.utils.data import Dataset
from collections import OrderedDict
import re
import cv2
import operator
import nibabel as nib

def active_model(model_name, w, h, z_dim):
    encoder = Encoder(w,h,z_dim*2)
    decoder = Decoder(w,h,z_dim)

    cpe = torch.load(model_name+'encoder_best.pth', map_location=torch.device('cpu'))
    cpd = torch.load(model_name+'decoder_best.pth', map_location=torch.device('cpu'))

    cpe_new = OrderedDict()
    cpd_new = OrderedDict()

    for k, v in cpe['encoder'].items():
        name = k[7:]
        cpe_new[name] = v

    for k, v in cpd['decoder'].items():
        name = k[7:]
        cpd_new[name] = v

    encoder.load_state_dict(cpe_new)
    decoder.load_state_dict(cpd_new)

    return encoder, decoder

def model_define(name):
    if re.search('L_', name) is not None:
        w = 158
        h = 126
    elif re.search('A_', name) is not None:
        w = 110
        h = 126
    elif re.search('S_', name) is not None:
        w = 110
        h = 158
    else:
        print('Error, no view identified (filename does not identify view).')
        print(name)

    if re.search('NewZ', name) is not None:
        z_dim = 128
    elif re.search('HighZ', name) is not None:
        z_dim = 512
    elif re.search('20231113', name) is not None:
        z_dim = 400
    else:
        z_dim = 256
    
    return w, h, z_dim

def perceptual_loss(input, recon, model_name):
    w, h, z_dim = model_define(model_name)
    encoder, decoder = active_model(model_name=model_name, w=w, h=h, z_dim=z_dim)

    in_feats = {}
    out_feats = {}

    def get_activation(name, it):
        def hook(model, input, output):
            if it == 0:
                in_feats[name] = output.detach().cpu().numpy().squeeze()
            else:
                out_feats[name] = output.detach().cpu().numpy().squeeze()

        return hook
    
    a = encoder.step0.register_forward_hook(get_activation('step0',0))
    b = encoder.step1.register_forward_hook(get_activation('step1',0))
    c = encoder.step2.register_forward_hook(get_activation('step2',0))
    d = encoder.step3.register_forward_hook(get_activation('step3',0))

    output = encoder(input)

    a.remove()
    b.remove()
    c.remove()
    d.remove()

    a = encoder.step0.register_forward_hook(get_activation('step0',1))
    b = encoder.step1.register_forward_hook(get_activation('step1',1))
    c = encoder.step2.register_forward_hook(get_activation('step2',1))
    d = encoder.step3.register_forward_hook(get_activation('step3',1))

    output = encoder(recon)    

    V = np.absolute(in_feats['step1'] - out_feats['step1'])
    V = np.expand_dims(V,axis=0)
    V = torch.from_numpy(V).type(torch.float)
    per_dif = decoder.activation(decoder.step4(decoder.step3(V)))
    return per_dif

def threshold(img):
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    img_norm[np.where(img_norm>100)] = 0
    ret, th = cv2.threshold(img_norm, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return img_norm, th

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
    def __init__(self, root_dir, view, size: int = 158, transform: float = None):
        self.root_dir = root_dir
        self.view = view
        self.horizontal_flip = transform 
        self.size = size

    def __len__(self):
        if self.view == 'L':
            size = 110
        elif self.view == 'A':
            size = 158
        else:
            size = 126
        return size
    
    def __getitem__(self, idx):
        raw = nib.load(self.root_dir).get_fdata()
        if self.view == 'L':
            n_img = resizing(raw[idx,:,:],(self.size,self.size))    
        elif self.view == 'A':
            n_img = resizing(raw[:,idx,:],(self.size,self.size))
        else:
            n_img = resizing(raw[:,:,idx],(self.size,self.size))

        num = n_img-np.min(n_img)
        den = np.max(n_img)-np.min(n_img)
        out = np.zeros((n_img.shape[0], n_img.shape[1]))
    
        n_img = np.divide(num, den, out=out, where=den!=0)

        n_img = np.expand_dims(n_img,axis=0)
        n_img = torch.from_numpy(n_img).type(torch.float)

        if self.transform is not None and self.transform <= 1:
            if np.random.rand(1) < self.transform:
                n_img = np.flip(n_img,axis=0)

        return n_img
