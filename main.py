# Code written by @simonamador

from train_framework import Trainer
from validation import Validator
from utils.config import *

import os, torch

# Obtain all configs from parser
parser = settings_parser()
args = parser.parse_args()

# Establish CUDA GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# Obtain all paths needed for training/validation from config
source_path, model_path, tensor_path, image_path, pre_path = path_generator(args)

if args.task == 'Train':
    trainer = Trainer(source_path, model_path, tensor_path,
                    image_path, device, args.batch, args.z, args.ga_method, args.type, 
                    args.model, args.view, args.n, args.pre, pre_path)
    trainer.train(args.epochs, args.loss)
elif args.task == 'Validate':
    validator = Validator(args.path, model_path, args.model, args.type, args.view, args.ga_method, 
                 args.z, args.name, args.n, device)
    # validator.validation()
    # if args.model == 'ga_VAE':
    #     validator.age_differential()
    validator.mannwhitneyu()