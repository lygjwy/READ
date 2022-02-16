import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor


class ODINPostprocessor(BasePostprocessor):

    def __init__(self, temperature=1000, magnitude=0.0014):
        super().__init__()

        self.temperature = temperature
        self.magnitude = magnitude


    def __call__(self, net, data):
        data.requires_grad = True
        logit, _ = net(data)

        criterion = nn.CrossEntropyLoss()

        pred = logit.detach().argmax(axis=1)

        logit = logit / self.temperature

        loss = criterion(logit, pred)
        loss.backward()

        # normalizing the gradient to binary in {-1, 1}
        gradient = torch.ge(data.grad.detach(), 0)
        gradient = (gradient.float() - 0.5) * 2

        # scaling the gradient by train set std
        gradient[:, 0] = gradient[:, 0] / (63.0 / 255.0)
        gradient[:, 1] = gradient[:, 1] / (62.1 / 255.0)
        gradient[:, 2] = gradient[:, 2] / (66.7/ 255.0)

        # add small pertubations to images
        tempInputs = torch.add(data.detach(), gradient, alpha=-self.magnitude)
        logit, _ = net(tempInputs)
        logit = logit / self.temperature
        # calculating the confidence after add the pertubation
        nnOutput = logit.detach()
        nnOutput = nnOutput - nnOutput.max(dim=1, keepdims=True).values
        nnOutput = nnOutput.exp() / nnOutput.exp().sum(dim=1, keepdims=True)

        conf, pred = nnOutput.max(dim=1)

        return pred, conf

