from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasePostprocessor():
    @torch.no_grad()
    def __call__(
        self,
        net: nn.Module,
        data: Any
    ):
        output = net(data)
        if type(output) == tuple:
            # resnet_vae
            logit, _ = output
        else:
            # resnet
            logit = output
        
        soft_max = torch.softmax(logit, dim=1)
        conf, pred = torch.max(soft_max, dim=1)
        
        # conf = maximum_softmax
        return pred, conf