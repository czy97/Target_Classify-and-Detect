import argparse
import torch
from Model import *
from ImageDataset import ImageDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from functions import train_model
import os

ap = argparse.ArgumentParser()
ap.add_argument("-num_epochs", "--num_epochs", type=int, default=100, help = "Epoch number to train")
ap.add_argument("-batchsize", "--batchsize", type=int, default=16, help = "batchsize")
ap.add_argument("-num_workers", "--num_workers", type=int, default=4, help = "num_workers")
ap.add_argument("-storeParamName", "--storeParamName", required = True, help = "storeParamName")
ap.add_argument("-gpu_id", "--gpu_id", required = True, help = "gpu id")
ap.add_argument("-loss_type", "--loss_type", required = True, help = "loss type")
ap.add_argument("-update_rule", "--update_rule", required = True, help = "update rule")
ap.add_argument("-model_type", "--model_type", required = True, help = "model type")
ap.add_argument("-loss_ratio", "--loss_ratio", type=float, default=1, help = "loss_ratio")
ap.add_argument("-lr_decay", "--lr_decay", type=float, default=1, help = "lr_decay")
ap.add_argument("-data_aug", "--data_aug", type=int, default=0, help = "data_aug")
args = vars(ap.parse_args())



cuda_name = "cuda:" + str(args['gpu_id'])
device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

#choose different model
if(args['model_type'] == 'Vgg'):
    model = VggBn_Net()
    initial_lr = 1e-3
else:
    model = Alex_Net()
    initial_lr = 1e-3

model = model.to(device)

#choose different update rule
params_to_update = model.parameters()
if(args['update_rule'] == 'Adam'):
    optimizer = torch.optim.Adam(params_to_update, lr=initial_lr)
else:
    optimizer = torch.optim.SGD(params_to_update, lr=initial_lr, momentum=0.9)


#get train and val dataloaders
#and choose whether data_augmentation
image_datasets = {x: ImageDataset(dataType = x,dataAug = (args['data_aug'] == 1) ) for x in ['train', 'val']}
dataloaders_dict = {x: DataLoader(image_datasets[x],
                        batch_size=args['batchsize'], shuffle=True, num_workers=args['num_workers']) for x in ['train', 'val']}

#choose different loss function
if(args['loss_type'] == 'L1'):
    criterions = [nn.CrossEntropyLoss(), nn.L1Loss()]
elif(args['loss_type'] == 'SmoothL1Loss'):
    criterions = [nn.CrossEntropyLoss(), nn.SmoothL1Loss()]
else:
    criterions = [nn.CrossEntropyLoss(), nn.MSELoss()]


model_ft, hist = train_model(model, dataloaders_dict, criterions, optimizer
                             , num_epochs=args['num_epochs'],device = device,loss_ratio = args['loss_ratio'],lr_decay = args['lr_decay'])


torch.save(model_ft.state_dict(), args['storeParamName'])



