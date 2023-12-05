import argparse

def settings_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_type',
        dest='type',
        choices=['default', 'residual', 'bVAE', 'self-attention','full'],
        required=True,
        help='''
        Type of model to train. Available options:
        "defalut" Default VAE using convolution blocks
        "residual: VAE which adds residual blocks between convolutions''')  
    parser.add_argument('--model_view',
        dest='view',
        choices=['L', 'A', 'S'],
        required=True,
        help='''
        The view of the image input for the model. Options:
        "L" Left view
        "A" Axial view
        "S" Sagittal view''') 
    parser.add_argument('--gpu',
        dest='gpu',
        choices=['0', '1', '2'],
        required=True,
        help='''
        The GPU that will be used for training. Terminals have the following options:
        Hanyang: 0, 1
        Busan: 0, 1, 2
        Sejong 0, 1, 2
        Songpa 0, 1
        Gangnam 0, 1
        ''')
    parser.add_argument('--epochs',
        dest='epochs',
        type=int,
        default=50,
        choices=range(1, 15000),
        required=False,
        help='''
        Number of epochs for training.
        ''')    
    parser.add_argument('--loss',
        dest='loss',
        default='SSIM',
        choices=['L2', 'SSIM', 'MS_SSIM', 'Mixed', 'Perceptual'],
        required=False,
        help='''
        Loss function:
        L2 = Mean square error.
        SSIM = Structural similarity index.
        ''')
    parser.add_argument('--batch',
        dest='batch',
        type=int,
        default=1,
        choices=range(1, 512),
        required=False,
        help='''
        Number of batch size.
        ''') 
    parser.add_argument('--beta',
        dest='beta',
        type=float,
        default=None,
        required=False,
        help='''
        Number of batch size.
        ''')
    parser.add_argument('--date',
    dest='date',
    default='20231013',
    required=False,
    help='''
    Date of model training.
    ''')
    parser.add_argument('--anomaly',
        dest='anomaly',
        default='healthy',
        choices = ['healthy', 'vm'],
        required=False,
        help='''
        Extra model name info.
        ''')
    parser.add_argument('--extra',
        dest='extra',
        default=False,
        required=False,
        help='''
        Extra model name info.
        ''')
    parser.add_argument('--z_dim',
        dest='z',
        type=int,
        default=512,
        required=False,
        help='''
        z dimension.
        ''')
    parser.add_argument('--path',
        dest = 'path',
        type = str,
        default = '/neuro/labs/grantlab/research/MRI_processing/carlos.amador/anomaly_detection/',
        required = False,
        help='''
        Path to the project directory
        ''')
    
    return parser