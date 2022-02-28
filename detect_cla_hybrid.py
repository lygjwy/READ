import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from datasets import get_ae_transforms, get_ae_ood_transforms, get_dataset_info, get_dataloader
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


def get_hybrid_inner_scores(ae, classifier, data_loader, normalize, combination):
    # feature level
    ae.eval()
    classifier.eval()
    
    ori_scores, rec_scores, similarity_scores, hybrid_scores = [], [], [], []
    
    for sample in data_loader:
        data = sample['data'].cuda()
        complexity = sample['complexity'].cuda()

        with torch.no_grad():
            rec_data = ae(data)
            
            data = torch.stack([normalize(img) for img in data], dim=0)
            rec_data = torch.stack([normalize(img) for img in rec_data], dim=0)
            
            penultimate_feature = classifier.penultimate_feature(data)
            rec_penultimate_feature = classifier.penultimate_feature(rec_data)
            
            ori_score, cla_idx = torch.max(classifier.fc(penultimate_feature), dim=1)
            ori_scores.extend(ori_score.tolist())
            
            cla_idx = torch.unsqueeze(cla_idx, 0).t()  # column vector
            rec_score = torch.gather(classifier.fc(rec_penultimate_feature), dim=1, index=cla_idx)
            rec_scores.extend(torch.squeeze(rec_score).tolist())
    
            # calculate the ori & rec similarity
            similarity = torch.bmm(penultimate_feature.view(args.batch_size, 1, -1), rec_penultimate_feature.view(args.batch_size, -1, 1))
            similarity = torch.squeeze(similarity)

            # process the similarity scores
            similarity_scores.extend(similarity.tolist())
    
    # combine
    if combination == 'ori':
        return ori_scores
    elif combination == 'diff':
        return similarity_scores
    elif combination == 'hybrid':
        for ori_score, similarity_score in zip(ori_scores, similarity_scores):
            hybrid_scores.append(ori_score + similarity_score)
        return hybrid_scores
    else:
        raise RuntimeError('<--- invalid combination: {}'.format(combination))


def get_hybrid_kl_scores(ae, classifier, data_loader, normalize, combination):
    # pred distribution level
    ae.eval()
    classifier.eval()
    
    complexities = []
    ori_scores, rec_scores, diff_scores, hybrid_scores = [], [], [], []
    
    for sample in data_loader:
        data = sample['data'].cuda()
        complexity = sample['complexity']
        complexities.extend(complexity.tolist())
        
        with torch.no_grad():
            rec_data = ae(data)
            
            data = torch.stack([normalize(img) for img in data], dim=0)
            rec_data = torch.stack([normalize(img) for img in rec_data], dim=0)
            
            ori_logit = classifier(data)
            rec_logit = classifier(rec_data)
            ori_softmax = torch.softmax(ori_logit, dim=1)
            rec_softmax = torch.softmax(rec_logit, dim=1)
            
            uniform_dist = torch.ones_like(ori_softmax) * (1 / ori_softmax.shape[1])
            ori_scores.extend(torch.sum(F.kl_div(ori_softmax.log(), uniform_dist, reduction='none'), dim=1).tolist())
            rec_scores.extend(torch.sum(F.kl_div(rec_softmax.log(), uniform_dist, reduction='none'), dim=1).tolist())
        
            diff_scores.extend(torch.sum(F.kl_div(rec_softmax.log(), ori_softmax, reduction='none'), dim=1).tolist())
            # print(diff_scores)

    # change complexities
    
    diff_coefficients = []
    for complexity in complexities:
        if complexity <= 0.65 or complexity >= 0.85:
            diff_coefficients.append(100.0)
        else:
            diff_coefficients.append(1.0)
    
    similarity_scores = [-1.0 * diff_score * diff_coefficient for diff_score, diff_coefficient in zip(diff_scores, diff_coefficients)]
    # combine
    if combination == 'ori':
        return ori_scores
    elif combination == 'diff':
        return similarity_scores
    elif combination == 'hybrid':
        for ori_score, similarity_score in zip(ori_scores, similarity_scores):
            hybrid_scores.append(ori_score + similarity_score)
        return hybrid_scores
    else:
        raise RuntimeError('<--- invalid combination: {}'.format(combination))


# def get_hybrid_maha_kl_scores(ae, classifier, data_loader, num_classes, sample_mean, precision, normalize, combination):
#     ae.eval()
#     classifier.eval()


scores_dic = {
    'hybrid_inner': get_hybrid_inner_scores,
    # 'hybrid_maha': get_hybrid_maha_scores,
    'hybrid_kl': get_hybrid_kl_scores
}


def draw_hist(data, colors, labels, title, fig_path):
    plt.clf()
    plt.hist(data, density=True, histtype='bar', color=colors, label=labels)
    plt.xlabel('score')
    plt.ylabel('density')
    plt.legend(prop={'size': 10})
    plt.title(title)
    plt.savefig(fig_path)


def main(args):
    output_path = Path(args.output_dir) / args.output_sub_dir
    print('>>> Log dir: {}'.format(str(output_path)))
    output_path.mkdir(parents=True, exist_ok=True)
    
    ae_transform = get_ae_transforms('test')
    
    means, stds = get_dataset_info(args.id, 'mean_and_std')
    normalize = Normalize(means, stds)
    
    get_dataloader_default = partial(
        get_dataloader,
        root=args.data_dir,
        split='test',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.prefetch
    )
    
    id_loader = get_dataloader_default(name=args.id, transform=ae_transform)
    ood_loaders = []
    for ood in args.oods:
        ood_transform = get_ae_ood_transforms(ood, 'test')
        ood_loaders.append(get_dataloader_default(name=ood, transform=ood_transform))
    
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
        print('>>> load classifier from {} (classification acc {:.4f}%)'.format(classifier_path, cla_acc))
    else:
        raise RuntimeError('---> invalid classifier path: {}'.format(str(classifier_path)))
    
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        ae.cuda()
        classifier.cuda()
    cudnn.benchmark = True
    
    # -------------------- inference -------------------- #
    get_scores = scores_dic[args.scores]
    result_dic_list = []

    id_scores = get_scores(ae, classifier, id_loader, normalize, args.combination)
    id_label = np.zeros(len(id_scores))
    
    for ood_loader in ood_loaders:
        result_dic = {'name': ood_loader.dataset.name}
        
        ood_scores = get_scores(ae, classifier, ood_loader, normalize, args.combination)
        ood_label = np.ones(len(ood_scores))
        scores = np.concatenate([id_scores, ood_scores])
        labels = np.concatenate([id_label, ood_label])
        
        result_dic['fpr_at_tpr'], result_dic['auroc'], result_dic['aupr_in'], result_dic['aupr_out'] = compute_all_metrics(scores, labels, verbose=False)
        result_dic_list.append(result_dic)

        print('---> [ID: {:7s} - OOD: {:9s}] [auroc: {:3.3f}%, aupr_in: {:3.3f}%, aupr_out: {:3.3f}%, fpr@95tpr: {:3.3f}%]'.format(
            id_loader.dataset.name, ood_loader.dataset.name, 100. * result_dic['auroc'], 100. * result_dic['aupr_in'], 100. * result_dic['aupr_out'], 100. * result_dic['fpr_at_tpr']))
        
        # plot hist
        hist_scores = [id_scores, ood_scores]
        colors = ['lime', 'red']
        labels = ['id', 'ood']
        title = '-'.join([ood_loader.dataset.name, args.id, args.scores])
        fig_path = output_path / (title + '.png')
        draw_hist(hist_scores, colors, labels, title, fig_path)
        
    #  save result
    result = pd.DataFrame(result_dic_list)
    log_path = output_path / (args.scores + '.csv')
    result.to_csv(str(log_path), index=False, header=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reconstruciton Detect')
    parser.add_argument('--data_dir', type=str, default='/home/iip/datasets')
    parser.add_argument('--output_dir', help='dir to store log', default='logs')
    parser.add_argument('--output_sub_dir', help='sub dir to store log', default='tmp')
    parser.add_argument('--id', type=str, default='cifar10')
    parser.add_argument('--oods', nargs='+', default=['svhn', 'cifar100', 'tinc', 'tinr', 'lsunc', 'lsunr', 'dtd', 'places365_10k', 'isun'])
    parser.add_argument('--ae', type=str, default='res_ae')
    parser.add_argument('--ae_path', type=str, default='./snapshots/r.pth')
    parser.add_argument('--classifier', type=str, default='wide_resnet')
    parser.add_argument('--classifier_path', type=str, default='./snapshots/p.pth')
    parser.add_argument('--scores', type=str, default='hybrid_inner')
    parser.add_argument('--combination', type=str, default='hybrid')  # ori, diff, both 
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument('--gpu_idx', type=int, default=0)
    
    args = parser.parse_args()

    main(args)