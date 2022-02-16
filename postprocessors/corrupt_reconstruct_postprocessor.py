import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor


class CorruptReconstructPostprocessor(BasePostprocessor):
    
    def __init__(self):
        super(CorruptReconstructPostprocessor, self).__init__()
        
    @torch.no_grad()
    def __call__(self, net, cor_data, data):
        # resnet_ae
        logit, rec_data = net(cor_data)
        
        rec_err = torch.sum(F.mse_loss(rec_data, data, reduction='none'), dim=[1, 2, 3])
        
        soft_max = F.softmax(logit, dim=1)
        _, pred = torch.max(soft_max, dim=1)
        
        id_score = -rec_err
        return pred, id_score
        