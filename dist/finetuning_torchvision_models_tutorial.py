"""
Finetuning Torchvision Models
=============================
**Author:** `Nathan Inkawhich <https://github.com/inkawhich>`__
"""

from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dist_utils import dist_init, average_gradients, DistModule
from sampler_utils import DistributedSampler, TestDistributedSampler, IterationBasedBatchSampler, DistributedGivenIterationSampler
from torch.utils.data.sampler import RandomSampler, BatchSampler


print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

distributed = False #True

if distributed:
    rank, world_size = dist_init('23456')
else:
    rank = 0
    world_size = 1

# Top level data directory. Here we assume the format of the directory conforms 
#   to the ImageFolder structure
data_dir = "./data/hymenoptera_data"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "squeezenet"

# Number of classes in the dataset
num_classes = 2

# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Number of epochs to train for 
num_epochs = 15

# Flag for feature extracting. When False, we finetune the whole model, 
#   when True we only update the reshaped layer params
feature_extract = True


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
if rank == 0:
    print(model_ft) 

# Send the model to GPU
model_ft = model_ft.cuda() ##to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are 
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()

if distributed:
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

if distributed:
    model_ft = DistModule(model_ft)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

if rank == 0:
    print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# Create training and validation dataloaders
train_sampler = RandomSampler(image_datasets['train'])
val_sampler = None #RandomSampler(image_datasets['val'])

if distributed:
    total_iter = (num_epochs * len(image_datasets['train'])-1) // world_size // batch_size + 1
    #train_sampler = DistributedGivenIterationSampler(image_datasets['train'], total_iter, batch_size, world_size=None, rank=rank, last_iter=-1) #DistributedSampler(image_datasets['train'])
    train_sampler = DistributedSampler(image_datasets['train'])
    val_sampler = DistributedSampler(image_datasets['val']) #TestDistributedSampler(image_datasets['val'])

sampler_dict = {'train': train_sampler, 'val': val_sampler}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], sampler=sampler_dict[x], shuffle=False, batch_size=batch_size, num_workers=4) for x in ['train', 'val']}

#train_batch_sampler = BatchSampler(sampler=train_sampler, batch_size=batch_size, drop_last=False)
#train_batch_sampler = IterationBasedBatchSampler(batch_sampler=train_batch_sampler)
#val_batch_sampler = BatchSampler(sampler=val_sampler, batch_size=batch_size, drop_last=False)

#batch_sampler = {'train': train_batch_sampler, 'val': val_batch_sampler}

#dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_sampler=batch_sampler[x], shuffle=False, num_workers=4) for x in ['train', 'val']}

#dataloaders_dict['train'].batch_sampler.set_start_iter(0)
#dataloaders_dict['train'].batch_sampler.set_num_iterations(num_epochs)
#dataloaders_dict['train'].size = len(dataloaders_dict['train'].batch_sampler.batch_sampler)

#data_size_dict = {'train': len(dataloaders_dict['train'].batch_sampler.batch_sampler), 'val': len(dataloaders_dict['val'].batch_sampler)}

#dataloaders_dict['train'] = iter(dataloaders_dict['train'])
#dataloaders_dict['val'] = iter(dataloaders_dict['val'])

if rank == 0:
    print('dataset length, train: {}, val: {}'.format(len(dataloaders_dict['train'].dataset), len(dataloaders_dict['val'].dataset)))
    #print('train batchsampler: {}, val batchsampler: {}'.format(len(dataloaders_dict['train'].batch_sampler.batch_sampler), len(dataloaders_dict['val'].batch_sampler)))
    print('dataloader length, train: {}, val: {}'.format(len(dataloaders_dict['train']), len(dataloaders_dict['val'])))
# Detect if we have a GPU available
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    if distributed:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0

    for epoch in range(num_epochs):
        if rank == 0:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            
            if distributed:
                dataloaders[phase].sampler.set_epoch = epoch 

            # Iterate over data.
            for iter, (inputs, labels) in enumerate(dataloaders[phase]):
                #if rank == 0: ##debug
                #    print('length of dataloader: {}'.format(len(dataloaders[phase])))
                inputs = inputs.cuda() ##to(device)
                labels = labels.cuda() ##to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    loss = loss / world_size
                    reduced_loss = loss.data.clone()
                    if distributed:
                        dist.all_reduce(reduced_loss)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        if distributed:
                            average_gradients(model)
                        optimizer.step()

                # statistics
                running_loss += reduced_loss.item() * inputs.size(0) * world_size
                reduced_corrects = torch.sum(preds == labels.data).float()
                # print('rank: {}, inputs.size: {}, reduced_corrects: {}'.format(rank, inputs.size(0), reduced_corrects))
                reduced_corrects = reduced_corrects / world_size
                if distributed:
                    dist.all_reduce(reduced_corrects)
                running_corrects += reduced_corrects * world_size

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if rank == 0:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

        if rank == 0:
            print()

    if rank == 0:
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))



