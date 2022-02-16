import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor


class ReconstructPostprocessor(BasePostprocessor):
    
    def __init__(self):
        super(ReconstructPostprocessor, self).__init__()
        
    @torch.no_grad()
    def __call__(self, net, data):
        # resnet_ae
        logit, rec_data = net(data)
        
        # reconstruction error
        rec_error = torch.sum(F.mse_loss(rec_data, data, reduction='none'), dim=[1, 2, 3])
        # maximum softmax score
        soft_max = F.softmax(logit, dim=1)
        _, pred = torch.max(soft_max, dim=1)
        
        # KL-divergence between U & prediction distribution
        # uniform_dist = torch.ones_like(soft_max) * (1 / soft_max.shape[1])
        # kl_score = torch.sum(F.kl_div(soft_max.log(), uniform_dist, reduction='none'), dim=1)
        
        # define in-distribution score
        # id_score = kl_score - reconstruction_error
        id_score = -rec_error
        return pred, id_score