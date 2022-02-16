'''Compare uni_ori_kl, ori_rec_kl ood detection within different classifiers
'''
import numpy as np
from pathlib import Path
from functools import partial
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from datasets import get_ae_transform, get_dataset_info, get_dataloader
from evaluation.metrics import compute_all_metrics
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
    

def get_scores(ae, classifier, data_loader, normalize):
    ae.eval()
    classifier.eval()
    
    uni_ori_kls, ori_rec_kls, scores = [], [], []
    
    for sample in data_loader:
        if type(sample) == list:
            data, _ = sample
        else:
            data = sample
        
        data = data.cuda()
        with torch.no_grad():
            rec_data = ae(data)
            
        data = torch.stack([normalize(img) for img in data], dim=0)
        rec_data = torch.stack([normalize(img) for img in rec_data], dim=0)
        
        with torch.no_grad():
            ori_logit = classifier(data)
            rec_logit = classifier(rec_data)
        ori_softmax = torch.softmax(ori_logit, dim=1)
        rec_softmax = torch.softmax(rec_logit, dim=1)
        
        uniform_dist = torch.ones_like(ori_softmax) * (1 / ori_softmax.shape[1])
        uni_ori_kls.extend(torch.sum(F.kl_div(ori_softmax.log(), uniform_dist, reduction='none'), dim=1).tolist())
        ori_rec_kls.extend(torch.sum(F.kl_div(rec_softmax.log(), ori_softmax, reduction='none'), dim=1).tolist())
        
    for uni_ori_kl, ori_rec_kl in zip(uni_ori_kls, ori_rec_kls):
        scores.append(uni_ori_kl - ori_rec_kl)
    return uni_ori_kls, [-1.0 * ori_rec_kl for ori_rec_kl in ori_rec_kls], scores


def draw_hist(ax, data, colors, labels, title):
    ax.hist(data, density=True, histtype='bar', color=colors, label=labels)
    ax.set_xlabel('score')
    ax.set_ylabel('density')
    ax.legend(prop={'size': 15})
    ax.set_title(title)
    

def detect_ood(id_scores, ood_scores, ax, colors, labels, title):
    id_labels = np.zeros(len(id_scores))
    ood_labels = np.ones(len(ood_scores))
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([id_labels, ood_labels])
    # fpr_at_tpr, auroc, aupr_in, aupr_out = compute_all_metrics(scores, labels)
    compute_all_metrics(scores, labels)
    # print('AUROC: {:.6f}'.format(auroc))
    # draw plot
    scores = [id_scores, ood_scores]
    draw_hist(ax, scores, colors, labels, title)
    
    
def main(args):
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ae_transform = get_ae_transform('test')
    
    means, stds = get_dataset_info(args.id, 'mean_and_std')
    normalize = Normalize(means, stds)
    
    
    # -------------------- dataloader -------------------- #
    get_dataloader_default = partial(
        get_dataloader,
        root=args.data_dir,
        split='test',
        transform=ae_transform,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.prefetch
    )
    
    id_loader = get_dataloader_default(name=args.id)
    ood_loader = get_dataloader_default(name=args.ood)
    
    # -------------------- ae & classifiers -------------------- #
    ae = get_ae(args.ae)
    num_classes = len(get_dataset_info(args.id, 'classes'))
    pure_cla = get_classifier(args.classifier, num_classes)
    or_cla = get_classifier(args.classifier, num_classes)
    orh_cla = get_classifier(args.classifier, num_classes)
    
    ae_path = Path(args.ae_path)
    pure_cla_path = Path(args.pure_cla_path)
    or_cla_path = Path(args.or_cla_path)
    orh_cla_path = Path(args.orh_cla_path)
    
    if ae_path.exists():
        ae_params = torch.load(str(ae_path))
        rec_err = ae_params['rec_err']
        ae.load_state_dict(ae_params['state_dict'])
        print('>>> load ae from {} (rec err {})'.format(str(ae_path), rec_err))
    else:
        raise RuntimeError('---> invalid ae path: {}'.format(str(ae_path)))
    
    if pure_cla_path.exists():
        cla_params = torch.load(str(pure_cla_path))
        cla_acc = cla_params['cla_acc']
        pure_cla.load_state_dict(cla_params['state_dict'])
        print('>>> load classifier from {} (classification acc {:.4f})'.format(pure_cla_path, cla_acc))
    else:
        raise RuntimeError('---> invalid classifier path: {}'.format(str(pure_cla_path)))
    
    if or_cla_path.exists():
        cla_params = torch.load(str(or_cla_path))
        cla_acc = cla_params['cla_acc']
        or_cla.load_state_dict(cla_params['state_dict'])
        print('>>> load classifier from {} (classification acc {:.4f})'.format(or_cla_path, cla_acc))
    else:
        raise RuntimeError('---> invalid classifier path: {}'.format(str(or_cla_path)))
    
    if orh_cla_path.exists():
        cla_params = torch.load(str(orh_cla_path))
        cla_acc = cla_params['cla_acc']
        orh_cla.load_state_dict(cla_params['state_dict'])
        print('>>> load classifier from {} (classification acc {:.4f})'.format(orh_cla_path, cla_acc))
    else:
        raise RuntimeError('---> invalid classifier path: {}'.format(str(orh_cla_path)))
    
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        ae.cuda()
        pure_cla.cuda()
        or_cla.cuda()
        orh_cla.cuda()
    cudnn.benchmark = True
    
    # -------------------- inference -------------------- #
    id_pure_uni_ori_kls, id_pure_ori_rec_kls, id_pure_scores = get_scores(ae, pure_cla, id_loader, normalize)
    ood_pure_uni_ori_kls, ood_pure_ori_rec_kls, ood_pure_scores = get_scores(ae, pure_cla, ood_loader, normalize)
    
    id_or_uni_ori_kls, id_or_ori_rec_kls, id_or_scores = get_scores(ae, or_cla, id_loader, normalize)
    ood_or_uni_ori_kls, ood_or_ori_rec_kls, ood_or_scores = get_scores(ae, or_cla, ood_loader, normalize)
    
    id_orh_uni_ori_kls, id_orh_ori_rec_kls, id_orh_scores = get_scores(ae, orh_cla, id_loader, normalize)
    ood_orh_uni_ori_kls, ood_orh_ori_rec_kls, ood_orh_scores = get_scores(ae, orh_cla, ood_loader, normalize)
    
    fig, axs = plt.subplots(3, 3, figsize=(20, 15))
    
    # plot hists
    colors = ['lime', 'red']
    labels = ['id', 'ood']
    #  one fig compares three classifiers with 3 metrics
    detect_ood(id_pure_uni_ori_kls, ood_pure_uni_ori_kls, axs[0, 0], colors, labels, args.ood + '-pure-uni_ori_kl')
    detect_ood(id_pure_ori_rec_kls, ood_pure_ori_rec_kls, axs[0, 1], colors, labels, args.ood + '-pure-ori_rec_kl')
    detect_ood(id_pure_scores, ood_pure_scores, axs[0, 2], colors, labels, args.ood + '-pure-scores')
    
    detect_ood(id_or_uni_ori_kls, ood_or_uni_ori_kls, axs[1, 0], colors, labels, args.ood + '-or-uni_ori_kl')
    detect_ood(id_or_ori_rec_kls, ood_or_ori_rec_kls, axs[1, 1], colors, labels, args.ood + '-or-ori_rec_kl')
    detect_ood(id_or_scores, ood_or_scores, axs[1, 2], colors, labels, args.ood + '-or-scores')
    
    detect_ood(id_orh_uni_ori_kls, ood_orh_uni_ori_kls, axs[2, 0], colors, labels, args.ood + '-orh-uni_ori_kl')
    detect_ood(id_orh_ori_rec_kls, ood_orh_ori_rec_kls, axs[2, 1], colors, labels, args.ood + '-orh-ori_rec_kl')
    detect_ood(id_orh_scores, ood_orh_scores, axs[2, 2], colors, labels, args.ood + '-orh-scores')
    
    plt.savefig(str(output_path / ('com_clas-detect-' + args.ood + '.png')))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compare diff clas')
    parser.add_argument('--data_dir', type=str, default='./datasets')
    parser.add_argument('--output_dir', help='dir to store experiment artifacts', default='outputs')
    parser.add_argument('--id', type=str, default='cifar10')
    parser.add_argument('--ood', type=str, default='cifar100')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument('--ae', type=str, default='res_ae')
    parser.add_argument('--ae_path', type=str, default='./outputs/res_ae/rec_best.pth')
    parser.add_argument('--classifier', type=str, default='resnet18')
    parser.add_argument('--pure_cla_path', type=str, default='./outputs/resnet18/cla_best.pth')
    parser.add_argument('--or_cla_path', type=str, default='./outputs/resnet18_or/cla_best.pth')
    parser.add_argument('--orh_cla_path', type=str, default='./outputs/resnet18_or_hybrid/cla_best.pth')
    parser.add_argument('--gpu_idx', type=int, default=0)
    
    args = parser.parse_args()

    main(args)