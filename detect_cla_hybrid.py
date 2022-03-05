import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
import argparse
import matplotlib.pyplot as plt
import sklearn.covariance

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
    
    complexities = []
    scores, similarities, hybrid_scores = [], [], []
    
    for sample in data_loader:
        data = sample['data'].cuda()
        complexity = sample['complexity']
        complexities.extend(complexity.tolist())
        
        with torch.no_grad():
            rec_data = ae(data)
            
            data = torch.stack([normalize(img) for img in data], dim=0)
            rec_data = torch.stack([normalize(img) for img in rec_data], dim=0)
            
            penultimate_feature = classifier.penultimate_feature(data)
            rec_penultimate_feature = classifier.penultimate_feature(rec_data)
            
            score, _ = torch.max(classifier.fc(penultimate_feature), dim=1)
            scores.extend(score.tolist())
            
            # calculate the ori & rec similarity
            similarity = torch.bmm(penultimate_feature.view(args.batch_size, 1, -1), rec_penultimate_feature.view(args.batch_size, -1, 1))
            similarity = torch.squeeze(similarity)

            # process the similarity scores
            similarities.extend(similarity.tolist())
    
    # change complexities
    simi_coefficients = []
    for complexity in complexities:
        if complexity <= 0.55:
            simi_coefficients.append(0.01)
        elif complexity < 0.85:
            simi_coefficients.append(0.1)
        else:
            simi_coefficients.append(0.01)
    
    simi_scores = [similarity * simi_coefficient for similarity, simi_coefficient in zip(similarities, simi_coefficients)]
    # combine
    if combination == 'ori':
        return scores
    elif combination == 'diff':
        return simi_scores
    elif combination == 'hybrid':
        for score, simi_score in zip(scores, simi_scores):
            hybrid_scores.append(score + simi_score)
        return hybrid_scores
    else:
        raise RuntimeError('<--- invalid combination: {}'.format(combination))


def get_hybrid_kl_scores(ae, classifier, data_loader, normalize, combination):
    # pred distribution level
    ae.eval()
    classifier.eval()
    
    complexities = []
    scores, differents, hybrid_scores = [], [], []
    
    for sample in data_loader:
        data = sample['data'].cuda()
        complexity = sample['complexity']
        complexities.extend(complexity.tolist())
        
        with torch.no_grad():
            rec_data = ae(data)
            
            data = torch.stack([normalize(img) for img in data], dim=0)
            rec_data = torch.stack([normalize(img) for img in rec_data], dim=0)
            
            logit = classifier(data)
            rec_logit = classifier(rec_data)
            softmax = torch.softmax(logit, dim=1)
            rec_softmax = torch.softmax(rec_logit, dim=1)
            
            uniform_dist = torch.ones_like(softmax) * (1 / softmax.shape[1])
            scores.extend(torch.sum(F.kl_div(softmax.log(), uniform_dist, reduction='none'), dim=1).tolist())
            differents.extend(torch.sum(F.kl_div(rec_softmax.log(), softmax, reduction='none'), dim=1).tolist())

    # change complexities
    diff_coefficients = []
    for complexity in complexities:
        if complexity <= 0.55:
            diff_coefficients.append(10.0)
        elif complexity < 0.85:
            diff_coefficients.append(0.1)
        else:
            diff_coefficients.append(1.0)
    
    simi_scores = [-1.0 * different * diff_coefficient for different, diff_coefficient in zip(differents, diff_coefficients)]
    # combine
    if combination == 'ori':
        return scores
    elif combination == 'diff':
        return simi_scores
    elif combination == 'hybrid':
        for ori_score, simi_score in zip(scores, simi_scores):
            hybrid_scores.append(ori_score + simi_score)
        return hybrid_scores
    else:
        raise RuntimeError('<--- invalid combination: {}'.format(combination))


def sample_estimator(classifier, data_loader, normalize, num_classes, feature_dim_list):
    classifier.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    
    num_layers = len(feature_dim_list)  # num of layers
    num_sample_per_class = np.zeros(num_classes)
    list_features = [[0] * num_classes] * num_layers
    
    for sample in data_loader:
        data = sample['data'].cuda()
        target = sample['label'].cuda()
        
        data = torch.stack([normalize(img) for img in data], dim=0)
        hidden_features = classifier.feature_list(data)
        
        # get hidden features
        for i in range(num_layers):
            hidden_features[i] = hidden_features[i].view(hidden_features[i].size(0), hidden_features[i].size(1), -1)
            hidden_features[i] = torch.mean(hidden_features[i].data, 2) # shape [batch_size, nChannels]
        
        # construct the sample matrix
        for i in range(target.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                layer_count = 0
                for hidden_feature in hidden_features:
                    list_features[layer_count][label] = hidden_feature[i].view(1, -1)
                    layer_count += 1
            else:
                layer_count = 0
                for hidden_feature in hidden_features:
                    list_features[layer_count][label] = torch.cat((list_features[layer_count][label], hidden_feature[i].view(1, -1)), 0)
                    layer_count += 1
            num_sample_per_class[label] += 1

    category_sample_mean = []
    layer_count = 0
    for feature_dim in feature_dim_list:
        tmp_list = torch.Tensor(num_classes, int(feature_dim)).cuda()
        for j in range(num_classes):
            tmp_list[j] = torch.mean(list_features[layer_count][j], 0)
        category_sample_mean.append(tmp_list)
        layer_count += 1
    
    precision = []
    for k in range(num_layers):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - category_sample_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - category_sample_mean[k][i]), 0)
        
        # find inverse
        group_lasso.fit(X.cpu().numpy())
        tmp_precision = group_lasso.precision_
        tmp_precision = torch.from_numpy(tmp_precision).float().cuda()
        precision.append(tmp_precision)
    
    return category_sample_mean, precision


def get_hybrid_maha_kl_scores(ae, classifier, data_loader, num_classes, sample_mean, precision, layer_index, normalize, combination):
    # prediction distribution level
    ae.eval()
    classifier.eval()

    complexities = []
    scores, differents, hybrid_scores = [], [], []
    
    for sample in data_loader:
        data = sample['data'].cuda()
        complexity = sample['complexity']
        complexities.extend(complexity.tolist())
        
        with torch.no_grad():
            rec_data = ae(data)
            
            data = torch.stack([normalize(img) for img in data], dim=0)
            rec_data = torch.stack([normalize(img) for img in rec_data], dim=0)
            
            hidden_feature = classifier.hidden_feature(data, layer_index)
            hidden_feature = hidden_feature.view(hidden_feature.size(0), hidden_feature.size(1), -1)
            hidden_feature = torch.mean(hidden_feature, 2)
            
            rec_hidden_feature = classifier.hidden_feature(rec_data, layer_index)
            rec_hidden_feature = rec_hidden_feature.view(rec_hidden_feature.size(0), rec_hidden_feature.size(1), -1)
            rec_hidden_feature = torch.mean(rec_hidden_feature, 2)
            
            gaussian_score = 0
            rec_gaussian_score = 0
            
            for i in range(num_classes):
                category_sample_mean = sample_mean[layer_index][i]
                zero_f = hidden_feature.data - category_sample_mean
                rec_zero_f = rec_hidden_feature.data - category_sample_mean
                term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
                rec_term_gau = -0.5 * torch.mm(torch.mm(rec_zero_f, precision[layer_index]), rec_zero_f.t()).diag()
                
                if i == 0:
                    gaussian_score = term_gau.view(-1, 1)
                    rec_gaussian_score = rec_term_gau.view(-1, 1)
                else:
                    gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)
                    rec_gaussian_score = torch.cat((rec_gaussian_score, rec_term_gau.view(-1, 1)), 1)

            # ori_maha_softmax = torch.softmax(gaussian_score, dim=1)
            gaussian_score = gaussian_score - gaussian_score.max(dim=1, keepdims=True).values
            ori_maha_softmax = gaussian_score.exp() / gaussian_score.exp().sum(dim=1, keepdims=True) + 1e-40  # prevent inf kl-div
            
            # rec_maha_softmax = torch.softmax(rec_gaussian_score, dim=1)
            rec_gaussian_score = rec_gaussian_score - rec_gaussian_score.max(dim=1, keepdims=True).values
            rec_maha_softmax = rec_gaussian_score.exp() / rec_gaussian_score.exp().sum(dim=1, keepdims=True) + 1e-40
            
            uniform_dist = torch.ones_like(ori_maha_softmax) * (1. / num_classes)
            # get gaussian score [batch_size, num_classes]
            scores.extend(torch.sum(F.kl_div(ori_maha_softmax.log(), uniform_dist, reduction='none'), dim=1).tolist())
            differents.extend(torch.sum(F.kl_div(rec_maha_softmax.log(), ori_maha_softmax, reduction='none'), dim=1).tolist())
    
    # change complexities
    diff_coefficients = []
    for complexity in complexities:
        if complexity <= 0.55:
            diff_coefficients.append(10.0)
        elif complexity < 0.85:
            diff_coefficients.append(0.1)
        else:
            diff_coefficients.append(1.0)
    
    simi_scores = [-1.0 * different * diff_coefficient for different, diff_coefficient in zip(differents, diff_coefficients)]
     # combine
    if combination == 'ori':
        return scores
    elif combination == 'diff':
        return simi_scores
    elif combination == 'hybrid':
        for ori_score, simi_score in zip(scores, simi_scores):
            hybrid_scores.append(ori_score + simi_score)
        return hybrid_scores
    else:
        raise RuntimeError('<--- invalid combination: {}'.format(combination))


scores_dic = {
    'hybrid_inner': get_hybrid_inner_scores,
    'hybrid_kl': get_hybrid_kl_scores,
    'hybrid_maha_kl': get_hybrid_maha_kl_scores
}


def draw_hist(data, colors, labels, title, fig_path):
    plt.clf()
    plt.hist(data, bins=100, density=True, histtype='bar', color=colors, label=labels)
    plt.xlabel('score')
    plt.ylabel('density')
    plt.legend(prop={'size': 10})
    plt.title(title)
    plt.savefig(fig_path)


def main(args):
    output_path = Path(args.output_dir) / args.id / args.output_sub_dir
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

    if args.scores == 'hybrid_maha_kl':
        num_layers = 1
        feature_dim_list = np.empty(num_layers)
        feature_dim_list[0] = 128  # for wide_resnet
        
        sample_mean, precision = sample_estimator(classifier, id_loader, normalize, num_classes, feature_dim_list)
        id_scores = get_hybrid_maha_kl_scores(ae, classifier, id_loader, num_classes, sample_mean, precision, num_layers-1, normalize, args.combination)
    else:
        id_scores = get_scores(ae, classifier, id_loader, normalize, args.combination)
    id_label = np.zeros(len(id_scores))
    
    for ood_loader in ood_loaders:
        result_dic = {'name': ood_loader.dataset.name}
        
        if args.scores == 'hybrid_maha_kl':
            ood_scores = get_hybrid_maha_kl_scores(ae, classifier, ood_loader, num_classes, sample_mean, precision, num_layers-1, normalize, args.combination)
        else:
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
    parser = argparse.ArgumentParser(description='Detect OOD using ori_score - coefficient * diff')
    parser.add_argument('--data_dir', type=str, default='/home/iip/datasets')
    parser.add_argument('--output_dir', help='dir to store log', default='logs')
    parser.add_argument('--output_sub_dir', help='sub dir to store log', default='hybrid_cla')
    parser.add_argument('--id', type=str, default='cifar10')
    parser.add_argument('--oods', nargs='+', default=['svhn', 'lsunc', 'dtd', 'places365_10k', 'cifar100', 'tinc', 'lsunr', 'tinr', 'isun'])
    parser.add_argument('--ae', type=str, default='res_ae')
    parser.add_argument('--ae_path', type=str, default='./snapshots/cifar10/rec.pth')
    parser.add_argument('--classifier', type=str, default='wide_resnet')
    parser.add_argument('--classifier_path', type=str, default='./snapshots/cifar10/wrn.pth')
    parser.add_argument('--scores', type=str, default='hybrid_maha_kl')
    parser.add_argument('--combination', type=str, default='hybrid')  # ori, diff, hybrid 
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument('--gpu_idx', type=int, default=0)
    
    args = parser.parse_args()

    main(args)