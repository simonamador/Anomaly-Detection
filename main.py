# Code written by @simonamador

from train_framework import Trainer
from validation import Validator
from visualizer import Visualizer
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

# Convert args to a dictionary and add device and paths dynamically
parameters = vars(args).copy()
parameters['device'] = device.type  
parameters['source_path'] = source_path
parameters['model_path'] = model_path
parameters['tensor_path'] = tensor_path
parameters['image_path'] = image_path
parameters['pre_path'] = pre_path

if __name__ == "__main__":
    if args.task == 'Train':
        trainer = Trainer(parameters)
        trainer.train(args.epochs, args.loss)
    elif args.task == 'Validate':
        # validator = Validator(args.path, model_path, args.model, args.type, args.view, args.ga_method, 
        #             args.z_dim, args.name, args.slice_size, device, args.training_folder, args.ga_n, args.raw, args.th, args.cGAN)
        validator = Validator(parameters)
        #validator.validation()
        #if args.model == 'ga_VAE':
        #    validator.age_differential(delta_ga=5)
        #    validator.age_differential(delta_ga=10)
        validator.mannwhitneyu()
        #validator.AUROC()
        #validator.multiview_AUROC_ASL()
        #validator.multiview_AUROC_AS()
    elif args.task == 'Visualize':
        # visualizer = Visualizer(args.path, model_path, args.VAE_model_type, args.type, args.view, args.ga_method, 
        #             args.z_dim, args.name, args.slice_size, device, args.training_folder, args.ga_n, args.raw, args.th, args.cGAN)
        visualizer = Visualizer(parameters)
        #visualizer.visualize_age_effect()
        #visualizer.save_reconstruction_images()
        visualizer.save_whole_range_plus_refined(TD=True)
        visualizer.save_whole_range_plus_refined(TD=False)
