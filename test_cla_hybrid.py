from pathlib import Path
import argparse
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from datasets import get_ae_transform, get_ae_normal_transform, get_dataset_info, get_dataloader, get_corrupt_dataloader
from models import get_ae, get_classifier


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            # t.mul_(s).add_(m)
            t.sub_(m).div_(s)
        return tensor


def eval_or_classification(ae, classifier, data_loader, normalize):
    ae.eval()
    classifier.eval()
    
    total, ori_correct, rec_correct = 0, 0, 0
    
    for sample in data_loader:
        assert len(sample) == 2
        data, target = sample
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            rec_data = ae(data)
        
        data = torch.stack([normalize(img) for img in data], dim=0)
        rec_data = torch.stack([normalize(img) for img in rec_data], dim=0)
        
        with torch.no_grad():
            ori_logit = classifier(data)
            rec_logit = classifier(rec_data)
            total += target.size(0)
            _, ori_pred = ori_logit.max(dim=1)
            _, rec_pred = rec_logit.max(dim=1)
            ori_correct += ori_pred.eq(target).sum().item()
            rec_correct += rec_pred.eq(target).sum().item()
        
    metrics = {
        'ori_cla_acc': 100. * ori_correct / total,
        'rec_cla_acc': 100. * rec_correct / total
    }
    
    return metrics


def eval_cor_classification(ae, classifier, data_loader, normalize):
    total, cor_correct, ori_correct, rec_correct = 0, 0, 0, 0
    
    for sample in data_loader:
        assert len(sample) == 3
        cor_data, data, target = sample
        cor_data, data, target = cor_data.cuda(), data.cuda(), target.cuda()
        with torch.no_grad():
            rec_data = ae(cor_data)
        
        cor_data = torch.stack([normalize(img) for img in cor_data], dim=0)
        data = torch.stack([normalize(img) for img in data], dim=0)
        rec_data = torch.stack([normalize(img) for img in rec_data], dim=0)
        
        with torch.no_grad():
            cor_logit = classifier(cor_data)
            ori_logit = classifier(data)
            rec_logit = classifier(rec_data)
            total += target.size(0)
            _, cor_pred = cor_logit.max(dim=1)
            _, ori_pred = ori_logit.max(dim=1)
            _, rec_pred = rec_logit.max(dim=1)
            cor_correct += cor_pred.eq(target).sum().item()
            ori_correct += ori_pred.eq(target).sum().item()
            rec_correct += rec_pred.eq(target).sum().item()
    
    metrics = {
        'cor_cla_acc': 100. * cor_correct / total,
        'ori_cla_acc': 100. * ori_correct / total,
        'rec_cla_acc': 100. * rec_correct / total
    }
    
    return metrics


def main(args):
    
    # -------------------- data loader -------------------- #
    ae_transform = get_ae_transform('test')
    ae_normal_transform = get_ae_normal_transform()
    
    means, stds = get_dataset_info(args.dataset, 'mean_and_std')
    normalize = Normalize(means, stds)
    print('>>> Reconstruction Dataset: {} with {} data mode'.format(args.dataset, args.data_mode))
    if args.data_mode == 'original':
        get_dataloader_default = partial(
            get_dataloader,
            root=args.data_dir,
            name=args.dataset,
            transform=ae_transform,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.prefetch
        )
    elif args.data_mode == 'corrupt':
        get_dataloader_default = partial(
            get_corrupt_dataloader,
            root=args.data_dir,
            name=args.dataset,
            corrupt=args.corrupt,
            severity=args.severity,
            random=False,
            transform=ae_normal_transform,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.prefetch
        )
    else:
        raise RuntimeError('<--- invlaid data mode: '.format(args.data_mode))
    
    test_loader_train = get_dataloader_default(split='train')
    test_loader_test = get_dataloader_default(split='test')
    
    # -------------------- ae & classifier -------------------- #
    ae = get_ae(args.ae)
    num_classes = len(get_dataset_info(args.dataset, 'classes'))
    classifier = get_classifier(args.classifier, num_classes)
    ae_path = Path(args.ae_path)
    classifier_path = Path(args.classifier_path)
    
    if ae_path.exists():
        ae_params = torch.load(str(ae_path))
        rec_err = ae_params['rec_err']
        ae.load_state_dict(ae_params['state_dict'])
        print('>>> load ae from {} (rec err {})'.format(str(ae_path), rec_err))
    else:
        raise RuntimeError('<--- invalid ae path: {}'.format(str(ae_path)))
    
    if classifier_path.exists():
        cla_params = torch.load(str(classifier_path))
        cla_acc = cla_params['cla_acc']
        classifier.load_state_dict(cla_params['state_dict'])
        print('>>> load classifier from {} (classification acc {:.4f})'.format(classifier_path, cla_acc))
    else:
        raise RuntimeError('<--- invalid classifier path: {}'.format(str(classifier_path)))
    
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        ae.cuda()
        classifier.cuda()
    cudnn.benchmark = True
    
    ae.eval()
    classifier.eval()
    
    # -------------------- inference -------------------- #
    if args.data_mode == 'original':
        test_train_rec_cla_acc = eval_or_classification(ae, classifier, test_loader_train, normalize)['rec_cla_acc']
        test_test_rec_cla_acc = eval_or_classification(ae, classifier, test_loader_test, normalize)['rec_cla_acc']
    elif args.data_mode == 'corrupt':
        test_train_rec_cla_acc = eval_cor_classification(ae, classifier, test_loader_train, normalize)['rec_cla_acc']
        test_test_rec_cla_acc = eval_cor_classification(ae, classifier, test_loader_test, normalize)['rec_cla_acc']
    else:
        raise RuntimeError('<--- invalid data mode: {}'.format(args.data_mode))
    
    print('[train set rec cla acc: {:.4f}% | test set rec cla acc: {:.4f}%]'.format(test_train_rec_cla_acc, test_test_rec_cla_acc))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reconstruction image classify')
    parser.add_argument('--data_dir', type=str, default='./datasets')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--data_mode', type=str, default='original')
    parser.add_argument('--corrupt', type=str, default='gaussian_noise')
    parser.add_argument('--severity', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument('--ae', type=str, default='res_ae')
    parser.add_argument('--ae_path', type=str, default='./outputs/res_ae/rec_best.pth')
    parser.add_argument('--classifier', type=str, default='resnet18')
    parser.add_argument('--classifier_path', type=str, default='./outputs/resnet18/cla_best.pth')
    parser.add_argument('--gpu_idx', type=int, default=0)
    
    args = parser.parse_args()
    
    main(args)
    