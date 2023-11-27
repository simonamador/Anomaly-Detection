import numpy as np
from model import Encoder, Decoder
import torch
from collections import OrderedDict
import re
import cv2

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

    V = np.absolute(in_feats['step0'] - out_feats['step0'])
    V = np.expand_dims(V,axis=0)
    V = torch.from_numpy(V).type(torch.float)
    per_dif = decoder.activation(decoder.step4(V))
    return per_dif

def threshold(img):
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    img_norm[np.where(img_norm>100)] = 0
    ret, th = cv2.threshold(img_norm, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th