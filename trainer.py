import torch
import torch.nn as nn
import numpy as np
import os
from collections import defaultdict
import pandas as pd
import utils.utility as utility
from utils.functions import AverageMeter, accuracy, temporal_nms

class Trainer:
    def __init__(self, args, model, loss, loader, ckpt):
        self.args = args
        self.train_loader = loader.train_loader
        self.test_loader = loader.test_loader
        self.trainset = loader.trainset
        self.testset = loader.testset
        self.evaluateset = loader.evaluateset
        self.evaluate_loader = loader.evaluate_loader

        self.class_index = self.evaluateset.activity_index
        self.index_class = dict(zip(self.class_index.values(), self.class_index.keys()))

        self.ckpt = ckpt
        self.model = model
        self.loss = loss
        self.lr = 0.
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.device = torch.device('cpu' if args.cpu else 'cuda')

        if args.load != '':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckpt.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckpt.log)*args.test_every): self.scheduler.step()
    
    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]
        if lr != self.lr:
            self.ckpt.write_log('[INFO] Epoch: {}\tLearning rate: {:.2e}'.format(epoch, lr))
            self.lr = lr
        self.loss.start_log()
        self.model.train()

        for batch, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            rois = targets[:, :-1]
            labels = targets[:, -1]
            #print("inputs: ", inputs.shape, ", rois: ", rois.shape, ", labels: ", labels.shape)
            self.optimizer.zero_grad()
            outputs = self.model(inputs, rois)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()

            self.ckpt.write_log('\r[INFO] [{}/{}]\t{}/{}\t{}'.format(
                epoch, self.args.epochs,
                batch + 1, len(self.train_loader),
                self.loss.display_loss(batch)), 
            end='' if batch+1 != len(self.train_loader) else '\n')

        self.loss.end_log(len(self.train_loader))
                       
    def test(self):
        top1 = AverageMeter()
        top5 = AverageMeter()
        loss = AverageMeter()
        self.ckpt.add_log(torch.zeros(1, 3))
        
        epoch = self.scheduler.last_epoch + 1
        self.ckpt.write_log('\n[INFO] Test:')
        self.model.eval()

        for batch, (inputs, targets) in enumerate(self.test_loader):
            inputs = inputs.to(self.device)
            
            targets = targets.to(self.device)
            rois = targets[:, :-1]
            labels = targets[:, -1]
            
            # compute outputs
            outputs = self.model(inputs, rois)
            loss_tmp = self.loss(outputs, labels)
            outputs = nn.functional.softmax(outputs, dim=1)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, labels, topk=(1,5))

            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            loss.update(loss_tmp.item(), inputs.size(0))
            
        self.ckpt.log[-1, 0] = top1.avg
        self.ckpt.log[-1, 1] = top5.avg
        self.ckpt.log[-1, 2] = loss.avg
        bests = self.ckpt.log.max(0)
        self.ckpt.write_log(
            '[INFO] top1: {:.4f} top5: {:.4f} loss: {:.4f} (Best: {:.4f} @epoch {})'.format(
            top1.avg,
            top5.avg,
            loss.avg,
            bests[0][0],
            (bests[1][0] + 1)*self.args.test_every
            )
        )            
        if not self.args.test_only:
            self.ckpt.save(self, epoch, is_best=((bests[1][0] + 1)*self.args.test_every == epoch))
            
    def evaluate(self):
        self.model.eval()
        # 1. foward, get the score for each class
        video_ids = []
        segment_t = np.zeros((2, len(self.evaluateset)))
        scores = np.zeros((self.args.class_num, len(self.evaluateset)))
        
        for batch, (inputs, targets, videoids) in enumerate(self.evaluate_loader):
            if batch%self.args.print_freq==0:
                print("evaluate batch {}/{}".format(batch+1, len(self.evaluate_loader)))
            inputs = inputs.to(self.device)                                    
            targets = targets.to(self.device)
            rois = targets[:, :-1]
            labels = targets[:, -1]
            
            batch_size = inputs.shape[0]            
            # compute outputs
            outputs = self.model(inputs, rois)
            outputs = nn.functional.softmax(outputs, dim=1)
            outputs = outputs.detach().cpu().numpy()

            for i in range(batch_size):
                video_ids.append(videoids[i])
                segment_t[:, batch*batch_size+i] = rois.cpu().numpy()[i, 1:]
                scores[:, batch*batch_size+i] = outputs[i]
            #print(videoids[0], starts[0], ends[0], outputs.detach().cpu().numpy()[0])
            
        # postprocess for scores
        score = np.max(scores, axis=0)
        pred_labels = np.argmax(scores, axis=0)
        print(pred_labels)
        detections = pd.DataFrame({"video-id": video_ids, "start_t": segment_t[0],
            "end_t": segment_t[1], "label": pred_labels, "score": score})
        if 1:            
            detections.to_csv("dets.csv", index=False)            
        # 2. nms for each video for each class        
        detections_gbvn = detections.groupby("video-id")
        video_set = set(video_ids)
        results = defaultdict(list)
        for vid in video_set:
            detections_this_vid = detections_gbvn.get_group(vid)
            detections_this_vid_gbl = detections_this_vid.groupby("label")
            for l in range(1, self.args.class_num):
                try: 
                    detections_this_vid_label = detections_this_vid_gbl.get_group(l)
                    segments = detections_this_vid_label.loc[:, ["start_t", "end_t", "score"]].values
                    keep = temporal_nms(segments, thresh=0.6)
                    segments = segments[keep]
                    for seg in segments:
                        seg = list(seg)
                        results[vid].append({"label": self.index_class[l], "score": seg[2], "segment": seg[:2]})                       
                except: # there may be no label in the detections
                    #print("No segments for class: ", self.index_class[l], " with video: ", vid)                       
                    pass

                results_json = {"version":"VERSION 1.3","results":results,"external_data":{}}                

        self.ckpt.save_results(self.args.evaluate_results, results_json)
                                                  
        
    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        elif self.args.evaluate_only:
            self.evaluate()
            return True            
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs                   
