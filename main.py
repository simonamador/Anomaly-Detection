from train_framework import Trainer
from validation import Validator
from utils.config import *


import os, torch

parser = settings_parser()
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

source_path, model_path, tensor_path, image_path, pre_path = path_generator(args)

if args.task == 'Train':
    trainer = Trainer(source_path, model_path, tensor_path,
                    image_path, device, args.batch, args.z, args.ga_method, args.type, 
                    args.model, args.view, args.n, args.pre, pre_path)
    trainer.train_inpainting(args.epochs)
elif args.task == 'Validate':
    validator = Validator(args.path, model_path, args.model, args.type, args.view, args.ga_method, 
                 args.loss, args.batch, args.z, args.date, args.n, device)
    validator.validation()
    validator.stat_analysis()