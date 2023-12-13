from collections import OrderedDict
import torch

def load_model(model_path, base, ga_method, w, h, z_dim):

    if base == 'ga_VAE':
        from models.ga_vae import Encoder, Decoder
        encoder = Encoder(w,h,z_dim*2, method = ga_method)
    else:
        from models.vae import Encoder, Decoder
        encoder = Encoder(w,h,z_dim*2)
    decoder = Decoder(w,h,z_dim)

    cpe = torch.load(model_path+'encoder_best.pth', map_location=torch.device('cpu'))
    cpd = torch.load(model_path+'decoder_best.pth', map_location=torch.device('cpu'))

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