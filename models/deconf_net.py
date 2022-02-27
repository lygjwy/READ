import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(x):
    norm = torch.norm(x, p=2, dim=1)
    x = x / (norm.expand(1, -1).t() + .0001)
    return x


class CosineDeconf(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CosineDeconf, self).__init__()
        
        self.h = nn.Linear(in_features, num_classes, bias=False)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity='relu')
    
    def forward(self, x):
        x = norm(x)
        w = norm(self.h.weight)
        
        ret = torch.matmul(x, w.T)  # why nor w * x?
        return ret


class EuclideanDeconf(nn.Module):
    def __init__(self, in_features, num_classes):
        super(EuclideanDeconf, self).__init__()
        
        self.h = nn.Linear(in_features, num_classes, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity='relu')
        
    def forward(self, x):
        x = x.unsqueeze(2) # batch * latent * 1
        w = self.h.weight.T.unsqueeze(0) # 1 * latent * num_classes
        ret = -((x - w).pow(2)).mean(1)
        return ret
    
class InnerDeconf(nn.Module):
    def __init__(self, in_features, num_classes):
        super(InnerDeconf, self).__init__()
        
        self.h = nn.Linear(in_features, num_classes)
        self._init_weights()
        
    def _init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity='relu')
        self.h.bias.data = torch.zeros(size=self.h.bias.size())
    
    def forward(self, x):
        return self.h(x)


def get_h(h_name, in_features, num_classes):
    if h_name == 'inner':
        h = InnerDeconf(in_features, num_classes)
        h.name = 'inner'
    elif h_name == 'euclidean':
        h = EuclideanDeconf(in_features, num_classes)
        h.name = 'euclidean'
    elif h_name == 'cosine':
        h = CosineDeconf(in_features, num_classes)
        h.name = 'cosine'
    else:
        raise RuntimeError('<--- invalid h name: {}'.format(h_name))
    
    return h
    
class DeconfNet(nn.Module):
    def __init__(self, feature_extractor, h):
        super(DeconfNet, self).__init__()

        self.feature_extractor = feature_extractor
        in_features = self.feature_extractor.output_size
        
        self.h = h
        self.g = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
        
        # self.softmax = nn.Softmax()
        
    def forward(self, x):
        output = self.feature_extractor(x)
        
        numerators = self.h(output)
        denominators = self.g(output)
        
        quotients = numerators / denominators
        return quotients, numerators, denominators


    def penultimate_feature(self, x):
        return self.feature_extractor(x)

    