from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor


class KldPostprocessor(BasePostprocessor):
    
    def __init__(self):
        super(KldPostprocessor, self).__init__()
        
    @torch.no_grad()
    def __call__(
        self,
        net: nn.Module,
        data: Any
    ):
        logit, rec_data = net(data)
        
        soft_max = torch.softmax(logit, dim=1)
        _, pred = torch.max(soft_max, dim=1)
        
        # KL-divergence between U & prediction distribution
        uniform_dist = torch.ones_like(soft_max) * (1 / soft_max.shape[1])
        kl_score = torch.sum(F.kl_div(soft_max.log(), uniform_dist, reduction='none'), dim=1)
        
        id_score = kl_score
        return pred, id_score