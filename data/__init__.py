from importlib import import_module
from torchvision import transforms
from utils.random_erasing import RandomErasing
from data.sampler import RandomSampler
from torch.utils.data import dataloader
import pandas as pd
import numpy as np

class Data:
    def __init__(self, args):
        print('[INFO] Making data...')
        #train_list = [
        #    transforms.Resize((args.height, args.width), interpolation=3),
        #    transforms.RandomHorizontalFlip(),
        #    transforms.ToTensor(),
        #    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #]

        #train_transform = transforms.Compose(train_list)

        #test_transform = transforms.Compose([
        #    transforms.Resize((args.height, args.width), interpolation=3),
        #    transforms.ToTensor(),
        #    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #])
        module_dataset = import_module('data.' + args.dataset.lower())
        feature_data, feature_dim = None, None #getattr(module_dataset, 'import_feature')() #None, None

        if not args.test_only:            
            self.trainset = getattr(module_dataset, args.dataset)(subset='training', feature_data=feature_data, feature_dim=feature_dim)
            self.train_loader = dataloader.DataLoader(self.trainset,
                            #sampler=RandomSampler(self.trainset,args.batchid,batch_image=args.batchimage),
                            shuffle=True,
                            batch_size=args.batch_size,
                            num_workers=args.workers)
        else:
            self.train_loader = None
        
        if args.dataset in ['ActivityNet']:
            module = import_module('data.' + args.dataset.lower())
            self.testset = getattr(module, args.dataset)(subset='validation', feature_data=feature_data, feature_dim=feature_dim)
            self.evaluateset = getattr(module, args.dataset)(subset='validation', feature_data=feature_data, feature_dim=feature_dim, output_meta=True)
        else:
            raise Exception()

        self.test_loader = dataloader.DataLoader(self.testset, batch_size=args.batch_size, num_workers=args.workers)
        
        # For evaluation, sometimes we just need pseudo labels
        self.evaluate_loader = dataloader.DataLoader(self.evaluateset, batch_size=args.batch_size, num_workers=args.workers)
        
        print(len(self.trainset), len(self.testset), len(self.evaluateset))
