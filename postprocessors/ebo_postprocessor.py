import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor


class EBOPostprocessor(BasePostprocessor):

    def __init__(self, temperature=100.0):
        super().__init__()
        self.temperature = temperature

    @torch.no_grad()
    def __call__(self, net, data):
        logit, _ = net(data)

        score = torch.softmax(logit, dim=1)
        conf, pred = torch.max(score, dim=1)

        conf = self.temperature * torch.logsumexp(logit / self.temperature, dim=1)

        return pred, conf

