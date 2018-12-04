import os
from importlib import import_module

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, args, ckpt):
        super(Model, self).__init__()
        print('[INFO] Making model...')

        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.nGPU = args.nGPU
        self.cpu = args.cpu
        self.save_models = args.save_models

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)

        if not args.cpu and args.nGPU > 1:
            self.model = nn.DataParallel(self.model, range(args.nGPU))

        self.load(
            ckpt.dir,
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )

    def forward(self, x, rois):
        return self.model(x, rois)

    def get_model(self):
        if self.nGPU == 1 or self.cpu:
            return self.model
        else:
            return self.model.module

    def save(self, apath, epoch, is_best=False):
        '''save state_dict'''        
        target = self.get_model()
        torch.save(
            target.state_dict(), 
            os.path.join(apath, 'model', 'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_best.pt')
            )
        
        if self.save_models:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_{}.pt'.format(epoch))
            )

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        '''load state_dict'''
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if resume == -1:
            load_path = os.path.join(apath, 'model', 'model_latest.pt')
            print('Loading model from {}'.format(load_path))        
            self.get_model().load_state_dict(
                torch.load(
                    load_path,
                    **kwargs
                ),
                strict=True
            )
        elif resume == 0:
            if pre_train != '':
                print('Loading model from {}'.format(pre_train))
                self.get_model().load_state_dict(
                    torch.load(pre_train, **kwargs),
                    strict=False
                )
        else:
            load_path = os.path.join(apath, 'model', 'model_{}.pt'.format(resume))
            print('Loading model from {}'.format(load_path))
            self.get_model().load_state_dict(
                torch.load(
                    load_path,
                    **kwargs
                ),
                strict=True
            )
