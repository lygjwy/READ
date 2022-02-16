import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from datasets import get_ae_transform, get_dataset_info, get_dataloader, get_corrupt_dataloader
from models import get_ae, get_classifier
from evaluation import compute_all_metrics


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
            data, _ = sample  # labeled
        else:
            data = sample  # unlabeled
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


def get_cor_cla_msp_kl(ae, classifier, data_loader, normalize):
    ae.eval()
    classifier.eval()
    
    cor_cla_msp, ori_cla_msp, rec_cla_msp = [], [], []
    cor_cla_kl, ori_cla_kl, rec_cla_kl = [], [], []
    
    for sample in data_loader:
        
        if len(sample) ==  2:
            cor_data, data = sample # unlabeled
        elif len(sample) == 3:
            cor_data, data, _ = sample # labeled
        else:
            raise RuntimeError('---> invlaid sample length: {}'.format(len(sample)))
        
        cor_data, data = cor_data.cuda(), data.cuda()
        with torch.no_grad():
            rec_data = ae(cor_data)
        
        cor_data = torch.stack([normalize(img) for img in cor_data], dim=0)
        data =  torch.stack([normalize(img) for img in data], dim=0)
        rec_data = torch.stack([normalize(img) for img in rec_data], dim=0)
        with torch.no_grad():
            cor_logit = classifier(cor_data)
            ori_logit = classifier(data)
            rec_logit = classifier(rec_data)
        
        cor_softmax = torch.softmax(cor_logit, dim=1)
        ori_softmax = torch.softmax(ori_logit, dim=1)
        rec_softmax = torch.softmax(rec_logit, dim=1)
        cor_cla_msp.extend(torch.max(cor_softmax, dim=1)[0].tolist())
        ori_cla_msp.extend(torch.max(ori_softmax, dim=1)[0].tolist())
        rec_cla_msp.extend(torch.max(rec_softmax, dim=1)[0].tolist())
        
        uniform_dist = torch.ones_like(ori_softmax) * (1 / ori_softmax.shape[1])
        cor_cla_kl.extend(torch.sum(F.kl_div(cor_softmax.log(), uniform_dist, reduction='none'), dim=1).tolist())
        ori_cla_kl.extend(torch.sum(F.kl_div(ori_softmax.log(), uniform_dist, reduction='none'), dim=1).tolist())
        rec_cla_kl.extend(torch.sum(F.kl_div(rec_softmax.log(), uniform_dist, reduction='none'), dim=1).tolist())
    
    return cor_cla_msp, ori_cla_msp, rec_cla_msp, cor_cla_kl, ori_cla_kl, rec_cla_kl


def draw_hist(data, colors, labels, title, fig_path):
    plt.clf()
    plt.hist(data, density=True, histtype='bar', color=colors, label=labels)
    plt.xlabel('score')
    plt.ylabel('density')
    plt.legend(prop={'size': 10})
    plt.title(title)
    plt.savefig(fig_path)


def main(args):
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ae_transform = get_ae_transform('test')
    
    means, stds = get_dataset_info(args.id, 'mean_and_std')
    normalize = Normalize(means, stds)
    
    if args.data_mode == 'original':
        get_dataloader_default = partial(
            get_dataloader,
            root=args.data_dir,
            split='test',
            transform=ae_transform,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.prefetch
        )
    elif args.data_mode == 'corrupt':
        get_dataloader_default = partial(
            get_corrupt_dataloader,
            root=args.data_dir,
            corrupt=args.corrupt,
            severity=args.severity,
            split='test',
            transform=ae_transform,
            batch_size=args.batch_size,
            shuffle=False,
            num_wrokers=args.prefetch
        )
    else:
        raise RuntimeError('---> invlaid data mode: '.format(args.data_mode))
    
    id_loader = get_dataloader_default(name=args.id)
    ood_loaders = []
    for ood in args.oods:
        ood_loaders.append(get_dataloader_default(name=ood))
    
    # -------------------- ae & classifier -------------------- #
    ae = get_ae(args.ae)
    num_classes = len(get_dataset_info(args.id, 'classes'))
    classifier = get_classifier(args.classifier, num_classes)
    ae_path = Path(args.ae_path)
    classifier_path = Path(args.classifier_path)
    
    if ae_path.exists():
        ae_params = torch.load(str(ae_path))
        rec_err = ae_params['rec_err']
        ae.load_state_dict(ae_params['state_dict'])
        print('>>> load ae from {} (rec err {})'.format(str(ae_path), rec_err))
    else:
        raise RuntimeError('---> invalid ae path: {}'.format(str(ae_path)))
    
    if classifier_path.exists():
        cla_params = torch.load(str(classifier_path))
        cla_acc = cla_params['cla_acc']
        classifier.load_state_dict(cla_params['state_dict'])
        print('>>> load classifier from {} (classification acc {:.4f})'.format(classifier_path, cla_acc))
    else:
        raise RuntimeError('---> invalid classifier path: {}'.format(str(classifier_path)))
    
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        ae.cuda()
        classifier.cuda()
    cudnn.benchmark = True
    
    # -------------------- inference -------------------- #
    uni_ori_kls_result_dic_list, ori_rec_kls_result_dic_list, scores_result_dic_list = [], [], []
    
    if args.data_mode == 'original':
        #  detect ood utilizing kl-divergence between ori-img & rec-img prediction distribution
        id_uni_ori_kls, id_ori_rec_kls, id_scores = get_scores(ae, classifier, id_loader, normalize)
        id_label = np.zeros(len(id_scores))
        
        for ood_loader in ood_loaders:
            uni_ori_kls_result_dic, ori_rec_kls_result_dic, scores_result_dic = {}, {}, {}
            ood_uni_ori_kls, ood_ori_rec_kls, ood_scores = get_scores(ae, classifier, ood_loader, normalize)
            ood_label = np.ones(len(ood_scores))
        
            uni_ori_kls = np.concatenate([id_uni_ori_kls, ood_uni_ori_kls])
            ori_rec_kls = np.concatenate([id_ori_rec_kls, ood_ori_rec_kls])
            scores = np.concatenate([id_scores, ood_scores])
            labels = np.concatenate([id_label, ood_label])
            
            uni_ori_kls_result_dic['fpr_at_tpr'], uni_ori_kls_result_dic['auroc'], uni_ori_kls_result_dic['aupr_in'], uni_ori_kls_result_dic['aupr_out'] = compute_all_metrics(uni_ori_kls, labels)
            uni_ori_kls_result_dic_list.append(uni_ori_kls_result_dic)
            
            ori_rec_kls_result_dic['fpr_at_tpr'], ori_rec_kls_result_dic['auroc'], ori_rec_kls_result_dic['aupr_in'], ori_rec_kls_result_dic['aupr_out'] = compute_all_metrics(ori_rec_kls, labels)
            ori_rec_kls_result_dic_list.append(ori_rec_kls_result_dic)
            
            scores_result_dic['fpr_at_tpr'], scores_result_dic['auroc'], scores_result_dic['aupr_in'], scores_result_dic['aupr_out'] = compute_all_metrics(scores, labels)
            scores_result_dic_list.append(scores_result_dic)
            
            # plot hist
            uni_ori_kls = [id_uni_ori_kls, ood_uni_ori_kls]
            ori_rec_kls = [id_ori_rec_kls, ood_ori_rec_kls]
            scores = [id_scores, ood_scores]
            colors = ['lime', 'cyan']
            labels = ['id', 'ood']
            uni_ori_kls_title = '-'.join([ood_loader.dataset.name, args.id, 'uni_ori_kls'])
            ori_rec_kls_title = '-'.join([ood_loader.dataset.name, args.id, 'ori_rec_kls'])
            scores_title = '-'.join([ood_loader.dataset.name, args.id, 'scores'])
            uni_ori_kls_fig_path = output_path / (uni_ori_kls_title + '.png')
            ori_rec_kls_fig_path = output_path / (ori_rec_kls_title + '.png')
            scores_fig_path = output_path / (scores_title + '.png')
            draw_hist(uni_ori_kls, colors, labels, uni_ori_kls_title, uni_ori_kls_fig_path)
            draw_hist(ori_rec_kls, colors, labels, ori_rec_kls_title, ori_rec_kls_fig_path)
            draw_hist(scores, colors, labels, scores_title, scores_fig_path)
        
        #  save result
        uni_ori_kls_result = pd.DataFrame(uni_ori_kls_result_dic_list)
        ori_rec_kls_result = pd.DataFrame(ori_rec_kls_result_dic_list)
        scores_result = pd.DataFrame(scores_result_dic_list)
        uni_ori_kls_log_path = output_path / 'uni_ori_kls.csv'
        ori_rec_kls_log_path = output_path / 'ori_rec_kls.csv'
        scores_log_path = output_path / 'scores.csv'
        uni_ori_kls_result.to_csv(str(uni_ori_kls_log_path), index=False, header=True)
        ori_rec_kls_result.to_csv(str(ori_rec_kls_log_path), index=False, header=True)
        scores_result.to_csv(str(scores_log_path), index=False, header=True)
        
    elif args.data_mode == 'corrupt':
        cor_id_msp, ori_id_msp, rec_id_msp, cor_id_kl, ori_id_kl, rec_id_kl = get_cor_cla_msp_kl(ae, classifier, id_loader, normalize)
        
        for ood_loader in ood_loaders:
            cor_ood_msp, ori_ood_msp, rec_ood_msp, cor_ood_kl, ori_ood_kl, rec_ood_kl = get_cor_cla_msp_kl(ae, classifier, ood_loader, normalize)
            
            # plot hist
            msps = [cor_id_msp, ori_id_msp, rec_id_msp, cor_ood_msp, ori_ood_msp, rec_ood_msp]
            kls = [cor_id_kl, ori_id_kl, rec_id_kl, cor_ood_kl, ori_ood_kl, rec_ood_kl]
            colors = ['blue', 'lime', 'cyan', '', 'yellow', 'orangered', 'magenta']
            labels = ['id_cor', 'id_ori', 'id_rec', 'ood_cor', 'ood_ori', 'ood_rec']
            msp_title = '-'.join([ood_loader.dataset.name, args.id, 'msp-cor'])
            msp_fig_path = output_path / (msp_title + '.png')
            kl_title = '-'.join([ood_loader.dataset.name, args.id, 'kl-cor'])
            kl_fig_path = output_path / (kl_title + '.png')
            draw_hist(msps, colors, labels, msp_title, msp_fig_path)
            draw_hist(kls, colors, labels, kl_title, kl_fig_path)
    
    else:
        raise RuntimeError('---> invalid data mode: {}'.format(args.data_mode))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reconstruciton Detect')
    parser.add_argument('--data_dir', type=str, default='./datasets')
    parser.add_argument('--output_dir', help='dir to store experiment artifacts', default='outputs')
    parser.add_argument('--id', type=str, default='cifar10')
    parser.add_argument('--oods', nargs='+', default=['svhn', 'cifar100', 'tinc', 'tinr', 'lsunc', 'lsunr', 'dtd', 'places365', 'isun'])
    parser.add_argument('--data_mode', type=str, default='original')
    parser.add_argument('--corrupt', type=str, default='gaussian_noise')
    parser.add_argument('--severity', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument('--ae', type=str, default='res_ae')
    parser.add_argument('--ae_path', type=str, default='./outputs/res_ae/rec_best.pth')
    parser.add_argument('--classifier', type=str, default='resnet18')
    parser.add_argument('--classifier_path', type=str, default='./outputs/resnet18/cla_best.pth')
    parser.add_argument('--gpu_idx', type=int, default=0)
    
    args = parser.parse_args()

    main(args)