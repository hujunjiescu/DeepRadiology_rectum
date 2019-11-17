import os, logging, traceback, pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CrossEntropy2D(object):
    def __init__(self, weight=None, size_average=True, batch_average=True):
        self.size_average = size_average
        self.batch_average = batch_average
        if weight is None:
            self.criterion = nn.CrossEntropyLoss(weight=weight, size_average=False)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().cuda(), size_average=False)
    
    def __call__(self, logit, target):
        n, c, h, w = logit.size()
        target = target.squeeze(1)
        loss = self.criterion(logit, target.long())
        if self.size_average:
            loss /= (h * w)

        if self.batch_average:
            loss /= n
        return loss