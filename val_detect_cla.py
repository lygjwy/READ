''' Tune the ood detector's hyper-parameters
'''
import numpy as np
from pathlib import Path
from functools import partial
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from torchvision import transforms

from datasets import get_transforms, get_dataset, get_dataset_info
from datasets import get_dataloader, get_uniform_noise_dataloader
from datasets import AvgOfPair, GeoMeanOfPair
from datasets import get_shift_transform
from models import get_classifier
from evaluation import compute_all_metrics


def get_odin_scores(classifier, data_loader, temperature, magnitude):
    classifier.eval()
    
    odin_scores = []
    
    for sample in data_loader:
        if data_loader.dataset.labeled:
            data, _ = sample
        else:
            data = sample
        data = data.cuda()
        
        data.requires_grad = True
        logit = classifier(data)
        pred = logit.detach().argmax(axis=1)
        logit = logit / temperature
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logit, pred)
        loss.backward()
        
        # normalizing the gradient to binary in {-1, 1}
        gradient = torch.ge(data.grad.detach(), 0)
        gradient = (gradient.float() - 0.5) * 2
        
        gradient[:, 0] = gradient[:, 0] / (63.0 / 255)
        gradient[:, 1] = gradient[:, 1] / (62.1 / 255)
        gradient[:, 2] = gradient[:, 2] / (66.7 / 255)
        
        tmpInputs = torch.add(data.detach(), -magnitude, gradient)
        logit = classifier(tmpInputs)
        logit = logit / temperature
        # calculating the confidence after add the perturbation
        nnOutput = logit.detach()
        nnOutput = nnOutput - nnOutput.max(dim=1, keepdims=True).values
        nnOutput = nnOutput.exp() / nnOutput.exp().sum(dim=1, keepdims=True)
        
        odin_scores.extend(nnOutput.max(dim=1)[0].tolist())
    
    return odin_scores


def get_ood_val_loader(name, mean, std, get_dataloader_default):
    if name == 'pixelate':
        transform = transforms.Compose([
            get_shift_transform('pixelate'),
            transforms.ToTensor(), 
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            get_shift_transform(name),
            transforms.Normalize(mean, std)
        ])
    
    ood_val_loader = get_dataloader_default(transform=transform)
    
    return ood_val_loader


def main(args):
    #  print hyper-parameters
    test_transform = get_transforms(args.id, stage='test')
    
    get_dataloader_default = partial(
        get_dataloader,
        root=args.data_dir,
        name=args.id,
        split='test',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.prefetch
    )
    id_loader = get_dataloader_default(transform=test_transform)
    mean, std = get_dataset_info(args.id, 'mean_and_std')
    
    ood_loaders = []
    
    uniform_noise_loader = get_uniform_noise_dataloader(10000, args.batch_size, False, args.prefetch)
    ood_loaders.append(uniform_noise_loader)
    
    id_dataset = get_dataset(root=args.data_dir, name=args.id, split='test', transform=test_transform)
    avg_pair_loader = DataLoader(
        AvgOfPair(id_dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.prefetch,
        pin_memory=True
    )
    ood_loaders.append(avg_pair_loader)
    
    id_dataset = get_dataset(root=args.data_dir, name=args.id, split='test', transform=transforms.ToTensor())
    geo_mean_loader = DataLoader(
        GeoMeanOfPair(id_dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.prefetch,
        pin_memory=True
    )
    ood_loaders.append(geo_mean_loader)
    
    jigsaw_loader = get_ood_val_loader('jigsaw', mean, std, get_dataloader_default)
    ood_loaders.append(jigsaw_loader)
    
    speckle_loader = get_ood_val_loader('speckle', mean, std, get_dataloader_default)
    ood_loaders.append(speckle_loader)
    
    pixelate_loader = get_ood_val_loader('pixelate', mean, std, get_dataloader_default)
    ood_loaders.append(pixelate_loader)
    
    rgb_shift_loader = get_ood_val_loader('rgb_shift', mean, std, get_dataloader_default)
    ood_loaders.append(rgb_shift_loader)
    
    invert_loader = get_ood_val_loader('invert', mean, std, get_dataloader_default)
    ood_loaders.append(invert_loader)
    
    # load classifier
    num_classes = len(get_dataset_info(args.id, 'classes'))
    classifier = get_classifier(args.classifier, num_classes)
    classifier_path = Path(args.classifier_path)
    
    if classifier_path.exists():
        cla_params = torch.load(str(classifier_path))
        cla_acc = cla_params['cla_acc']
        classifier.load_state_dict(cla_params['state_dict'])
        # print('>>> load classifier from {} (classify acc {:.4f}%)'.format(str(classifier_path), cla_acc))
    else:
        raise RuntimeError('<--- invalid classifier path: {}'.format(str(classifier_path)))
    
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        classifier.cuda()
    cudnn.benchmark = True
    
    fpr_at_tprs, aurocs, aupr_ins, aupr_outs = [], [], [], []
    id_scores = get_odin_scores(classifier, id_loader, args.temperature, args.magnitude)
    id_label = np.zeros(len(id_scores))
    
    for ood_loader in ood_loaders:
        ood_scores = get_odin_scores(classifier, ood_loader, args.temperature, args.magnitude)
        ood_label = np.ones(len(ood_scores))
        
        scores = np.concatenate([id_scores, ood_scores])
        labels = np.concatenate([id_label, ood_label])
        
        fpr_at_tpr, auroc, aupr_in, aupr_out = compute_all_metrics(scores, labels, verbose=False)
        
        fpr_at_tprs.append(fpr_at_tpr)
        aurocs.append(auroc)
        aupr_ins.append(aupr_in)
        aupr_outs.append(aupr_out)
        
    # print ood datasets average
    # print('>>> Temperature: {:.4f} | Magnitude: {:.4f}'.format(args.temperature, args.magnitude))
    print('>>> [Temperature: {:.4f}, Magnitude: {:.4f}] [avg auroc: {:.4f} | avg fpr_at_tpr: {:.4f} | avg aupr_in: {:.4f} | avg aupr_out: {:.4f}]'.format(
            args.temperature,
            args.magnitude,
            np.mean(aurocs),
            np.mean(fpr_at_tprs),
            np.mean(aupr_ins),
            np.mean(aupr_outs)
        )
    )
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ID & OOD-val to tune hyper-parameter')
    parser.add_argument('--data_dir', type=str, default='./datasets')
    parser.add_argument('--id', type=str, default='cifar10')
    parser.add_argument('--temperature', type=int, default=1)
    parser.add_argument('--magnitude', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument('--classifier', type=str, default='wide_resnet')
    parser.add_argument('--classifier_path', type=str, default='./snapshots/e-p.pth')
    parser.add_argument('--gpu_idx', type=int, default=0)
    
    args = parser.parse_args()

    main(args)
