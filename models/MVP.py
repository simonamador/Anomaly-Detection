# Multiview processing

# TTA implementation. Having trained models, we can use this to perform a more accurate prediction.
from framework import Framework as F

class TTA:
    def __init__(self, source_path, model_paths, tensor_path,
                 image_path, device, batch, z_dim, method, model, 
                 base, view, n, pretrained, pretrained_path, ga_n, raw, th = 99):
        
        # Determine if model inputs GA
        if base == 'ga_VAE':
            self.ga = True
            print('-'*50)
            print('')
            print('Training GA Model.')
            print('')
        else:
            self.ga = False
            print('-'*50)
            print('')
            print('Training default Model.')
            print('')

        self.device = device
        self.model_type = model
        self.model_paths = model_paths  
        self.tensor_path = tensor_path 
        self.image_path = image_path  
        self.th = th

        # Generate model
        self.model = F(n, z_dim, method, device, model, self.ga, ga_n, th=self.th)

class MVP:
    pass