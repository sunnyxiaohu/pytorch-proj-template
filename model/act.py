import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import scipy
import scipy.interpolate
import numpy as np
#from roi_temporal_pooling import RoITemporalPoolFunction, _RoITemporalPooling

def min_max(x, xmin, xmax):
    assert(xmin<=xmax)
    x = max(x, xmin)
    x = min(x, xmax)
    return x
    
class TemporalPooling(nn.Module):
    '''pool the input feature to a fixed length'''
    def __init__(self, input_dim, pooled_length, pool_type='AVE'):
        super(TemporalPooling, self).__init__()
        self.input_length = input_dim[1] 
        self.pooled_length = pooled_length
        self.upsampled_length = self.pooled_length * self.input_length
        self.pool_type = pool_type
        
    def forward(self, input, segment):
        '''
        @param: input, input feature (batch_size, 400, 100)
        @param: segment, start and end index
        @return: pooled feature (batch_size, 400, 16)
        '''
        batch_size, ch = input.shape[0], input.shape[1]
        output = torch.zeros(batch_size, ch, self.pooled_length)
        #segment = segment.int()
        segment = segment + 0.5
        for b in range(batch_size):
            # A lazy implementation
            # 1. interp
            x_t = [(x+0.5)*self.pooled_length for x in range(self.input_length)]
            upsample_f = scipy.interpolate.interp1d(x_t, input[b], axis=1)
            # 2. get features to be pooled
            #print(segment)
            pool_x_t = [min_max(x, x_t[0], x_t[-1]) for x in range(self.pooled_length*segment[b, 0], self.pooled_length*(segment[b, 1]))]
            pool_feature = upsample_f(pool_x_t)
            #print(pool_feature.shape)
            # 3. pool features according pool type
            bin_size = len(pool_x_t) / self.pooled_length                        
            for l in range(self.pooled_length):
                start, end = int(l*bin_size) , int((l+1)*bin_size)
                if start>=end:
                    print(segment[b])
                    continue
                if self.pool_type == 'AVE':
                    x = torch.from_numpy(pool_feature[:, start:end])
                    #print(x.shape)
                    pooled_feature = torch.mean(x, dim=1)
                    output[b, :, l] = pooled_feature

        return output.type_as(input)  

    #def backward(self, grad_output):
    #    grad_output = None
    #    return grad_output
        
class Act(nn.Module):
    def __init__(self, input_dim, class_num):
        super(Act, self).__init__()
        self.input_dim = input_dim
        self.class_num = class_num
        self.roi_pooling = TemporalPooling(input_dim=input_dim, pooled_length=16)
        self.features = nn.Sequential(            
            nn.Conv1d(self.input_dim[0], 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Conv1d(256, self.class_num, kernel_size=3, padding=1)
        )
        
    def forward(self, input, rois):
        loss = None
        rois_feature = self.roi_pooling(input, rois)
        #print("roi_feature: ", rois_feature.shape)
        x = self.features(rois_feature)
        # Average over the temporal axis
        x = self.classifier(x).mean(-1)
                  
        return x
        
        
#TODO: parameter optionally
def make_model(args):
    return Act(input_dim=(400, 100), class_num=args.class_num)          
        
if __name__ == '__main__':
    x=torch.randn(2,400,100)
    m = Act((400,100), 201)
    seg=torch.from_numpy(np.array([[3,16],[15,100]]))
    y = m(x, seg)
