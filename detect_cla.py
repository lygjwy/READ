import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from datasets import get_transforms, get_ood_transforms, get_dataset_info, get_dataloader
from models import get_classifier
from evaluation import compute_all_metrics


def get_cla_msp_kl(classifier, data_loader):
    classifier.eval()
    
    cla_msp, cla_kl = [], []
    
    for sample in data_loader:
        
        if type(sample) == list:
                # labeled
                data, _ = sample
        else:
            # unlabeled
            data = sample
        data = data.cuda()
        
        with torch.no_grad():
            logit = classifier(data)
        softmax = torch.softmax(logit, dim=1)
        cla_msp.extend(torch.max(softmax, dim=1)[0].tolist())
        
        uniform_dist = torch.ones_like(softmax) * (1 / softmax.shape[1])
        cla_kl.extend(torch.sum(F.kl_div(softmax.log(), uniform_dist, reduction='none'), dim=1).tolist())
    
    return cla_msp, cla_kl
    

def draw_hist(data, colors, labels, title, fig_path):
    plt.clf()
    plt.hist(data, density=True, bins=50, histtype='bar', color=colors, label=labels)
    plt.xlabel('score')
    plt.ylabel('density')
    plt.legend(prop={'size': 10})
    plt.title(title)
    plt.savefig(fig_path)
    plt.close()


def main(args):
    output_path = Path(args.output_dir) / args.output_sub_dir
    print('>>> Log dir: {}'.format(str(output_path)))
    output_path.mkdir(parents=True, exist_ok=True)
    
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
    ood_loaders = []
    for ood in args.oods:
        ood_test_transform = get_ood_transforms(args.id, ood, 'test')
        ood_loaders.append(get_dataloader_default(name=ood, transform=ood_test_transform))
    
    #  load classifier
    num_classes = len(get_dataset_info(args.id, 'classes'))
    classifier = get_classifier(args.classifier, num_classes)
    classifier_path = Path(args.classifier_path)
    
    if classifier_path.exists():
        cla_params = torch.load(str(classifier_path))
        cla_acc = cla_params['cla_acc']
        classifier.load_state_dict(cla_params['state_dict'])
        print('>>> load classifier from {} (classifiy acc {:.4f}%)'.format(str(classifier_path), cla_acc))
    else:
        raise RuntimeError('<--- invalid classifier path: {}'.format(str(classifier_path)))
    
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        classifier.cuda()
    cudnn.benchmark = True
    
    msp_result_dic_list, kl_result_dic_list = [], []
    
    id_msp, id_kl = get_cla_msp_kl(classifier, id_loader)
    id_label = np.zeros(len(id_msp))
    
    for ood_loader in ood_loaders:
        msp_result_dic, kl_result_dic = {'name': ood_loader.dataset.name}, {'name': ood_loader.dataset.name}
        
        ood_msp, ood_kl = get_cla_msp_kl(classifier, ood_loader)
        ood_label = np.ones(len(ood_msp))
        # detect ood
        msps, kls = np.concatenate([id_msp, ood_msp]), np.concatenate([id_kl, ood_kl])
        labels = np.concatenate([id_label, ood_label])
        
        msp_result_dic['fpr_at_tpr'], msp_result_dic['auroc'], msp_result_dic['aupr_in'], msp_result_dic['aupr_out'] = compute_all_metrics(msps, labels)
        kl_result_dic['fpr_at_tpr'], kl_result_dic['auroc'], kl_result_dic['aupr_in'], kl_result_dic['aupr_out'] = compute_all_metrics(kls, labels)
    
        msp_result_dic_list.append(msp_result_dic)
        kl_result_dic_list.append(kl_result_dic)
        
        # plot hist
        hist_msps = [id_msp, ood_msp]
        hist_kls = [id_kl, ood_kl]
        colors = ['lime', 'cyan']
        labels = ['id',  'ood']
        msp_title = '-'.join([ood_loader.dataset.name, args.id, 'msp'])
        msp_fig_path = output_path / (msp_title + '.png')
        kl_title = '-'.join([ood_loader.dataset.name, args.id, 'kl'])
        kl_fig_path = output_path / (kl_title + '.png')
        draw_hist(hist_msps, colors, labels, msp_title, msp_fig_path)
        draw_hist(hist_kls, colors, labels, kl_title, kl_fig_path)
        
    # save result
    msp_result = pd.DataFrame(msp_result_dic_list)
    msp_log_path = output_path / 'msp.csv'
    msp_result.to_csv(str(msp_log_path), index=False, header=True)
    
    kl_result = pd.DataFrame(kl_result_dic_list)
    kl_log_path = output_path / 'kl.csv'
    kl_result.to_csv(str(kl_log_path), index=False, header=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reconstruciton Detect')
    parser.add_argument('--data_dir', type=str, default='./datasets')
    parser.add_argument('--output_dir', help='dir to store log', default='logs')
    parser.add_argument('--output_sub_dir', help='sub dir to store log', default='tmp')
    parser.add_argument('--id', type=str, default='cifar10')
    parser.add_argument('--oods', nargs='+', default=['svhn', 'cifar100', 'tinc', 'tinr', 'lsunc', 'lsunr', 'dtd', 'places365_10k', 'isun'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument('--classifier', type=str, default='wide_resnet')
    parser.add_argument('--classifier_path', type=str, default='./snapshots/p.pth')
    parser.add_argument('--gpu_idx', type=int, default=0)
    
    args = parser.parse_args()

    main(args)