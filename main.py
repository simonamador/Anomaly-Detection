from train_framework import Trainer
from utils.config import *


import os, torch

parser = settings_parser()
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

source_path, model_path, tensor_path, image_path, pre_path = path_generator(args)

trainer = Trainer(source_path, model_path, tensor_path,
                 image_path, device, args.batch, args.z, args.ga_method, args.type, 
                 args.model, args.view, args.n, args.pre, pre_path)

trainer.train_inpainting(args.epochs)