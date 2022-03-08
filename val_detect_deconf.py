''' Tune the ood detector's hyper-parameters
'''
import numpy as np
from pathlib import Path
from functools import partial
import argparse

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from torchvision import transforms

from datasets import get_transforms, get_dataset, get_dataset_info
from datasets import get_dataloader, get_uniform_noise_dataloader
from datasets import AvgOfPair, GeoMeanOfPair
from datasets import get_shift_transform
from models import get_deconf_net
from evaluation import compute_all_metrics


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
    
    ood_val_loader = get_dataloader_default(name=args.id, transform=transform)
    
    return ood_val_loader


def get_godin_scores(deconf_net, data_loader, magnitude=0.0010, score_func='h', std=(0.2470, 0.2435, 0.2616)):
    deconf_net.eval()
    
    godin_scores = []
    
    for sample in data_loader:
        if isinstance(sample, dict):
                data = sample['data']
        else:
            if data_loader.dataset.labeled:
                data, _ = sample
            else:
                data = sample
        data = data.cuda()
        
        data.requires_grad = True
        logits, h, g = deconf_net(data)
        
        if score_func == 'h':
            scores = h
        elif score_func == 'g':
            scores = g
        else:
            scores = logits
        
        max_scores, _ = torch.max(scores, dim=1)
        max_scores.backward(torch.ones(len(max_scores)).cuda())

        gradient = torch.ge(data.grad.detach(), 0)
        gradient = (gradient.float() - 0.5) * 2
        
        gradient[:, 0] = gradient[:, 0] / std[0]
        gradient[:, 1] = gradient[:, 1] / std[1]
        gradient[:, 2] = gradient[:, 2] / std[2]
        
        tmpInputs = torch.add(data.detach(), magnitude, gradient)
        
        logits, h, g = deconf_net(tmpInputs)
        if score_func == 'h':
            scores = h
        elif score_func == 'g':
            scores = g
        else:
            scores = logits
        
        godin_scores.extend(torch.max(scores, dim=1)[0].tolist())
    
    return godin_scores


scores_dic = {
    'godin': get_godin_scores
}


def main(args):
    #  print hyper-parameters
    test_transform = get_transforms(args.id, stage='test')
    
    get_dataloader_default = partial(
        get_dataloader,
        root=args.data_dir,
        split='test',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.prefetch
    )
    
    id_loader = get_dataloader_default(name=args.id, transform=test_transform)
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
    
    #  load deconf net
    num_classes = len(get_dataset_info(args.id.split('-')[0], 'classes'))
    # print('>>> Deconf: {} - {}'.format(args.feature_extractor, args.h))
    deconf_net = get_deconf_net(args.feature_extractor, args.h, num_classes)
    deconf_path = Path(args.deconf_path)
    
    if deconf_path.exists():
        deconf_params = torch.load(str(deconf_path))
        # cla_acc = deconf_params['cla_acc']
        deconf_net.load_state_dict(deconf_params['state_dict'])
        # print('>>> load deconf net from {} (classifiy acc {:.4f}%)'.format(str(deconf_path), cla_acc))
    else:
        raise RuntimeError('<--- invalid deconf path: {}'.format(str(deconf_path)))
    
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        deconf_net.cuda()
    cudnn.benchmark = True
    
    # ------------------------------------ detect ood ------------------------------------
    get_scores = scores_dic[args.scores]
    fpr_at_tprs, aurocs, aupr_ins, aupr_outs = [], [], [], []
    
    if args.scores == 'godin':
        id_scores = get_scores(deconf_net, id_loader, args.magnitude, args.score_func, std)
    else:
        id_scores = get_scores(deconf_net, id_loader)
    
    # another validation metrics
    avg_score = np.mean(id_scores)
    id_label = np.zeros(len(id_scores))
    
    for ood_loader in ood_loaders:
         
        if args.scores == 'godin':
            ood_scores = get_scores(deconf_net, ood_loader, args.magnitude, args.score_func, std)
        else:
            ood_scores = get_scores(deconf_net, ood_loader)
        ood_label = np.ones(len(ood_scores))
        
        scores = np.concatenate([id_scores, ood_scores])
        labels = np.concatenate([id_label, ood_label])
        
        fpr_at_tpr, auroc, aupr_in, aupr_out = compute_all_metrics(scores, labels, verbose=False)
        
        fpr_at_tprs.append(fpr_at_tpr)
        aurocs.append(auroc)
        aupr_ins.append(aupr_in)
        aupr_outs.append(aupr_out)
    
    if args.scores == 'godin':
        print('---> [Magnitude: {:.4f}] [avg_score: {:.4f} | avg auroc: {:.4f} | avg fpr_at_tpr: {:.4f} | avg aupr_in: {:.4f} | avg aupr_out: {:.4f}]'.format(
                args.magnitude,
                avg_score,
                np.mean(aurocs),
                np.mean(fpr_at_tprs),
                np.mean(aupr_ins),
                np.mean(aupr_outs)
            )
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ID & OOD-val to tune hyper-parameter')
    parser.add_argument('--data_dir', type=str, default='/home/iip/datasets')
    parser.add_argument('--id', type=str, default='cifar10')
    parser.add_argument('--scores', type=str, default='godin')
    parser.add_argument('--score_func', type=str, default='h')
    parser.add_argument('--magnitude', type=float, default=0.0014)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument('--feature_extractor', type=str, default='wide_resnet')
    parser.add_argument('--h', type=str, default='inner')
    parser.add_argument('--deconf_path', type=str, default='./snapshots/cifar10/wrn_i.pth')
    parser.add_argument('--gpu_idx', type=int, default=0)
    
    args = parser.parse_args()

    main(args)
