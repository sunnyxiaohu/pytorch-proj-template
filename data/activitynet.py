import torch
import torch.utils.data as data
import json
import numpy as np
import pandas as pd
import scipy
import os
from data.utils import get_blocked_videos
from data.utils import interpolated_prec_rec
from data.utils import segment_iou
from data.utils import wrapper_segment_iou
from utils.functions import temporal_nms

this_dir = os.path.dirname(os.path.abspath(__file__))

DEBUG = False

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def min_max(x, xmin, xmax):
    assert(xmin<=xmax)
    x = max(x, xmin)
    x = min(x, xmax)
    return x
        
def import_feature():
    feature_root = os.path.join(this_dir, '../../data/activitynet_feature_cuhk/csv_mean_100')
    csv_files = [f for f in os.listdir(feature_root) if f.endswith('.csv')]
    features = []
    #'''
    count = 0
    for cf in csv_files:
        tmp_df = pd.read_csv(os.path.join(feature_root, cf))
        count = count + 1
        if count==1:
            feature_dim = tmp_df.values.shape
        assert feature_dim == tmp_df.values.shape                
        features.append(tmp_df.values.reshape(-1))
        if count %1000==0:
            print('read {} feature files'.format(count))
    #'''            
    video_ids = [cf[:-4] for cf in csv_files]   
    #feature_dim = (100, 400)        
    #features = [np.random.rand(*feature_dim)] * len(video_ids)
    feature_data = pd.DataFrame({'video-id': video_ids, 'feature': features})   
    return feature_data , feature_dim           

class ActivityNet(data.Dataset):
    def __init__(self, subset, feature_data=None, feature_dim=None, transform=None, output_meta=False):
        self.subset = subset
        self.transform = transform
        self.proposal_file = os.path.join(this_dir, '../../output/result_activitynet_proposal_trainval.json')
        self.anno_file = os.path.join(this_dir, '../../Evaluation/data/activity_net.v1-3.min.json')
        self.output_meta = output_meta
        self.evaluate_mode = output_meta
        self.train_nms = 0.8
        self.val_nms = 0.7
        
        
        self.help_anno_file = os.path.join(this_dir, '../../data/activitynet_annotations/anet_anno_action.json')
        with open(self.help_anno_file, 'r') as fobj:
            self.help_anno = json.load(fobj)
        
        # Import ground truth and proposals.
        self.ground_truth, self.activity_index = self._import_ground_truth(
            self.anno_file)
        self.proposal = self._import_proposal(self.proposal_file)
        print("import ", self.proposal.shape[0], " proposals")
        # Do nms
        self.proposal = nms_proposals(self.proposal, self.train_nms if self.subset == 'training' else self.val_nms)       
        print("after nms, we get ", self.proposal.shape[0], " proposals")
        
        # Import feature
        self.feature_data, self.feature_dim = feature_data, feature_dim
        if self.feature_data is None:
            self.feature_root = '../../data/activitynet_feature_cuhk/csv_mean_100'

        # Compute targets for proposals
        if (self.subset in ['training', 'validation']): # and (not self.evaluate_mode):
            self.proposal_targets = compute_targets(self.ground_truth, self.proposal, pos_thresh=0.7, neg_thresh=0.3)
        else:
            self.proposal_targets = self.proposal.copy()
            self.proposal_targets['label'] = np.full(self.proposal_targets.shape[0], -1)                

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)

        # Read ground truth data. label_idx is starting from 1, 0 for background
        activity_index, cidx = {}, 1
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        for videoid, v in data['database'].items():
            if self.subset != v['subset']:
                continue

            for ann in v['annotations']:
                if ann['label'] not in activity_index:
                    activity_index[ann['label']] = cidx
                    cidx += 1
                video_lst.append(videoid)
                t_start_lst.append(ann['segment'][0])
                t_end_lst.append(ann['segment'][1])
                label_lst.append(activity_index[ann['label']])

        ground_truth = pd.DataFrame({'video-id': video_lst,
                                     't-start': t_start_lst,
                                     't-end': t_end_lst,
                                     'label': label_lst})
        return ground_truth, activity_index

    def _import_proposal(self, proposal_filename):
        """Reads proposal file, checks if it is well formatted, and returns
           the proposal instances.

        Parameters
        ----------
        proposal_filename : str
            Full path to the proposal json file.

        Outputs
        -------
        proposal : df
            Data frame containing the proposal instances.
        """
        with open(proposal_filename, 'r') as fobj:
            data = json.load(fobj)

        video_set = self.ground_truth['video-id'].unique()
        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        score_lst = []
        for videoid, v in data['results'].items():
            # Just load videos in the subset
            if videoid not in video_set:
                continue
            for result in v:
                video_lst.append(videoid)
                t_start_lst.append(result['segment'][0])
                t_end_lst.append(result['segment'][1])
                score_lst.append(result['score'])
        proposal = pd.DataFrame({'video-id': video_lst,
                                   't-start': t_start_lst,
                                   't-end': t_end_lst,
                                   'score': score_lst})
        return proposal
                    
    def __getitem__(self, index):
        videoid = 'v_' + self.proposal_targets.loc[index, 'video-id']
        segment_label = self.proposal_targets.loc[index, ['t-start', 't-end', 'label']].values
        if self.feature_data is None:
            feature_root = os.path.join(this_dir, '../../data/activitynet_feature_cuhk/csv_mean_100')
            tmp_df = pd.read_csv(os.path.join(feature_root, videoid+'.csv'))
            features = tmp_df.values
            self.feature_dim = features.shape
        else:
            features_gbvn = self.feature_data.groupby('video-id')
            this_video_features = features_gbvn.get_group(videoid)
            features = this_video_features['feature'].values[0].reshape(self.feature_dim)
        # Interpolate, Normalize data            
        # Use helper annotation file for correcting duration_sec ??
        corrected_sec = float(self.help_anno[videoid]['feature_frame']) / self.help_anno[videoid]['duration_frame'] * self.help_anno[videoid]['duration_second']
        # Using cv_mean_100 features (100, 400)
        start_idx = min_max(round(self.feature_dim[0] * segment_label[0] / corrected_sec), 0, self.feature_dim[0]-1)
        end_idx = min_max(round(self.feature_dim[0] * segment_label[1] / corrected_sec), start_idx+1, self.feature_dim[0])
        if end_idx-start_idx<1:
            print("sec: ", segment_label[0], " ", segment_label[1], ", idx: ", start_idx, " ", end_idx)
        input_data = torch.from_numpy(features).float()
        input_data = input_data.permute(1, 0) #(400, len)
        target = torch.from_numpy(np.array([start_idx, end_idx, segment_label[2]])).long() #(batch_index, start, end, label)
        #print("input_data: ", input_data.shape, " target: ", target.shape)
        if self.transform is not None:
            self.transform(input_data)

        if self.output_meta:
            return input_data, target, videoid[2:],
        else:            
            return input_data, target
        
    def __len__(self):
        return self.proposal_targets.shape[0]

def nms_proposals(proposals, nms_thresh=1.0):
    '''
    @proposals: proposals, df
    @return: proposals_nms, df
    '''      
    keep_all = []
    video_set = proposals['video-id'].unique()
    proposals_gbvn = proposals.groupby('video-id')
    for vid in video_set:
        this_video_proposals = proposals_gbvn.get_group(vid)
        this_video_proposals_nms = this_video_proposals.loc[:, ['t-start', 't-end', 'score']].values
        this_video_proposals_idx = this_video_proposals.index
        keep = temporal_nms(this_video_proposals_nms, nms_thresh)
        keep_all.extend(this_video_proposals_idx[keep])
    proposals_nms = proposals.loc[keep_all].reset_index(drop=True)
    return proposals_nms

def compute_targets(ground_truth, proposals, pos_thresh=0.7, neg_thresh=0.3):
    # Get list of videos.
    video_set = set(ground_truth['video-id'].unique()).intersection(proposals['video-id'].unique())
    
    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')
    proposals_gbvn = proposals.groupby('video-id')
    
    proposal_targets = proposals.copy()
    # label -1 for ignore
    proposal_labels = np.full(proposal_targets.shape[0], -1)
    proposal_tiou = np.full(proposal_targets.shape[0], -1.0)

    # For each video, compute tiou scores among the proposals and ground_truth
    for videoid in video_set:
        ground_truth_videoid = ground_truth_gbvn.get_group(videoid)
        this_video_ground_truth_idx = ground_truth_videoid.reset_index()
        this_video_ground_truth = this_video_ground_truth_idx.loc[:,['t-start', 't-end']].values
        
        proposals_videoid = proposals_gbvn.get_group(videoid)
        this_video_proposals = proposals_videoid.loc[:, ['t-start', 't-end']].values
        this_video_proposals_idx = proposals_videoid.loc[:, ['t-start', 't-end']].index

        for idx, this_proposal in enumerate(this_video_proposals):
            tiou = segment_iou(this_proposal, this_video_ground_truth)
            argmax = tiou.argmax()
            if tiou[argmax] > pos_thresh:
                proposal_labels[this_video_proposals_idx[idx]] = this_video_ground_truth_idx.label[argmax] # foreground
            elif tiou[argmax] < neg_thresh:
                proposal_labels[this_video_proposals_idx[idx]] = 0 # background        
            proposal_tiou[this_video_proposals_idx[idx]] = tiou[argmax]
    
    # Select samples according a criterion
    pos_idxs = np.where(proposal_labels>0)[0]
    num_pos = pos_idxs.shape[0]    
    neg_idxs = np.where(proposal_labels==0)[0]
    class_num = max(proposal_labels)
    num_neg = min(int(num_pos/class_num), neg_idxs.shape[0])
    neg_idxs = np.random.permutation(neg_idxs)
    proposal_labels[neg_idxs[num_neg:]] = -1
    
    proposal_targets['label'] = proposal_labels
    proposal_targets['tiou'] = proposal_tiou
    proposal_targets = proposal_targets[ proposal_targets.label != -1 ].reset_index()
    
    if DEBUG:
        for l in range(201):
            num = sum(proposal_targets['label'].values == l)
            print (num, " samples for class ", l)
    return proposal_targets            

