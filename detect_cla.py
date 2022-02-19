import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from datasets import get_transforms, get_ood_transforms, get_dataset_info, get_dataloader
from models import get_classifier
from evaluation import compute_all_metrics


def get_msp_scores(classifier, data_loader):
    classifier.eval()
    
    msp_scores = []
    for sample in data_loader:
        if data_loader.dataset.labeled:
            data, _ = sample
        else:
            data = sample
        data = data.cuda()
        
        with torch.no_grad():
            logit = classifier(data)
        
            softmax = torch.softmax(logit, dim=1)
            msp_scores.extend(torch.max(softmax, dim=1)[0].tolist())
            
    return msp_scores


def get_kl_scores(classifier, data_loader):
    classifier.eval()
    
    kl_scores = []
    
    for sample in data_loader:
        if data_loader.dataset.labeled:
            data, _ = sample
        else:
            data = sample
        data = data.cuda()
        
        with torch.no_grad():
            logit = classifier(data)
            softmax = torch.softmax(logit, dim=1)
        
            uniform_dist = torch.ones_like(softmax) * (1 / softmax.shape[1])
            kl_scores.extend(torch.sum(F.kl_div(softmax.log(), uniform_dist, reduction='none'), dim=1).tolist())
    
    return kl_scores


def get_odin_scores(classifier, data_loader, temperature=1000.0, magnitude=0.0014):
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


def get_energy_scores(classifier, data_loader, temperature=1.0):
    classifier.eval()
    
    energy_scores = []
         
    for sample in data_loader:
        if data_loader.dataset.labeled:
            data, _ = sample
        else:
            data = sample
        data = data.cuda()
        
        with torch.no_grad():
            logit = classifier(data)
            energy_scores.extend((temperature * torch.logsumexp(logit / temperature, dim=1)).tolist())
    
    return energy_scores


scores_dic = {
    'msp': get_msp_scores,
    'kl': get_kl_scores,
    'odin': get_odin_scores,
    'energy': get_energy_scores
}


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
    
    get_scores = scores_dic[args.scores]
    result_dic_list = []
    
    id_scores = get_scores(classifier, id_loader)
    id_label = np.zeros(len(id_scores))
    
    for ood_loader in ood_loaders:
        print(ood_loader.dataset.name)
        result_dic = {'name': ood_loader.dataset.name}
        
        ood_scores = get_scores(classifier, ood_loader)
        ood_label = np.ones(len(ood_scores))
        # detect ood
        scores = np.concatenate([id_scores, ood_scores])
        labels = np.concatenate([id_label, ood_label])
        
        result_dic['fpr_at_tpr'], result_dic['auroc'], result_dic['aupr_in'], result_dic['aupr_out'] = compute_all_metrics(scores, labels)
        result_dic_list.append(result_dic)
        
        # plot hist
        hist_scores = [id_scores, ood_scores]
        colors = ['lime', 'red']
        labels = ['id',  'ood']
        title = '-'.join([ood_loader.dataset.name, args.id, args.scores])
        fig_path = output_path / (title + '.png')
        draw_hist(hist_scores, colors, labels, title, fig_path)
    
    # save result
    result = pd.DataFrame(result_dic_list)
    log_path = output_path / (args.scores + '.csv')
    result.to_csv(str(log_path), index=False, header=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reconstruciton Detect')
    parser.add_argument('--data_dir', type=str, default='./datasets')
    parser.add_argument('--output_dir', help='dir to store log', default='logs')
    parser.add_argument('--output_sub_dir', help='sub dir to store log', default='tmp')
    parser.add_argument('--id', type=str, default='cifar10')
    parser.add_argument('--oods', nargs='+', default=['svhn', 'cifar100', 'tinc', 'tinr', 'lsunc', 'lsunr', 'dtd', 'places365_10k', 'isun'])
    parser.add_argument('--scores', type=str, default='msp')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument('--classifier', type=str, default='wide_resnet')
    parser.add_argument('--classifier_path', type=str, default='./snapshots/p.pth')
    parser.add_argument('--gpu_idx', type=int, default=0)
    
    args = parser.parse_args()

    main(args)