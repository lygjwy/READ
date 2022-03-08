import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
import argparse
import matplotlib.pyplot as plt
import sklearn.covariance

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
        data = sample['data'].cuda()
        
        with torch.no_grad():
            logit = classifier(data)
        
            softmax = torch.softmax(logit, dim=1)
            msp_scores.extend(torch.max(softmax, dim=1)[0].tolist())
    
    return msp_scores


def get_kl_scores(classifier, data_loader):
    classifier.eval()
    
    kl_scores = []
    
    for sample in data_loader:
        data = sample['data'].cuda()
        
        with torch.no_grad():
            logit = classifier(data)
            softmax = torch.softmax(logit, dim=1)
        
            uniform_dist = torch.ones_like(softmax) * (1 / softmax.shape[1])
            kl_scores.extend(torch.sum(F.kl_div(softmax.log(), uniform_dist, reduction='none'), dim=1).tolist())
    
    return kl_scores


def get_odin_scores(classifier, data_loader, temperature=1000.0, magnitude=0.0014, std=(0.2470, 0.2435, 0.2616)):
    classifier.eval()
    
    odin_scores = []
    
    for sample in data_loader:
        data = sample['data'].cuda()
        
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
        
        gradient[:, 0] = gradient[:, 0] / std[0]
        gradient[:, 1] = gradient[:, 1] / std[1]
        gradient[:, 2] = gradient[:, 2] / std[2]
        
        tmpInputs = torch.add(data.detach(), -magnitude, gradient)
        logit = classifier(tmpInputs)
        logit = logit / temperature
        # calculating the confidence after add the perturbation
        nnOutput = logit.detach()
        nnOutput = nnOutput - nnOutput.max(dim=1, keepdims=True).values
        nnOutput = nnOutput.exp() / nnOutput.exp().sum(dim=1, keepdims=True)
        
        odin_scores.extend(nnOutput.max(dim=1)[0].tolist())
    
    return odin_scores


def sample_estimator(classifier, data_loader, num_classes, feature_dim_list):
    classifier.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    
    num_layers = len(feature_dim_list)
    num_sample_per_class = np.zeros(num_classes)
    list_features = [[0] * num_classes] * num_layers

    for sample in data_loader:
        data = sample['data'].cuda()
        target = sample['label'].cuda()
        
        hidden_features = classifier.feature_list(data)
        
        # get hidden features
        for i in range(num_layers):
            hidden_features[i] = hidden_features[i].view(hidden_features[i].size(0), hidden_features[i].size(1), -1)
            hidden_features[i] = torch.mean(hidden_features[i].data, 2)
        
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


def get_maha_scores(classifier, data_loader, num_classes, sample_mean, precision, layer_index, magnitude, std):
    classifier.eval()
    
    maha_scores = []
    for sample in data_loader:
        data = sample['data'].cuda()
        
        data.requires_grad = True 
        
        hidden_feature = classifier.hidden_feature(data, layer_index)
        hidden_feature = hidden_feature.view(hidden_feature.size(0), hidden_feature.size(1), -1)
        hidden_feature = torch.mean(hidden_feature, 2)
        
        # compute maha score
        gaussian_score = 0
        for i in range(num_classes):
            category_sample_mean = sample_mean[layer_index][i]
            zero_f = hidden_feature.data - category_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)
        
        # Input precessing
        sample_pred = gaussian_score.max(1)[1]
        category_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        
        zero_f = hidden_feature - category_sample_mean
        pure_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()
        
        gradient = torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        
        gradient[:, 0] = gradient[:, 0] / std[0]
        gradient[:, 1] = gradient[:, 1] / std[1]
        gradient[:, 2] = gradient[:, 2] / std[2]
        
        tmpInputs = torch.add(data.data, -magnitude, gradient)
        with torch.no_grad():
            noise_out_features = classifier.hidden_feature(tmpInputs, layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            category_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - category_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1, 1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)
        
        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        maha_scores.extend(noise_gaussian_score.tolist())

    return maha_scores


def get_energy_scores(classifier, data_loader, temperature=1.0):
    classifier.eval()
    
    energy_scores = []

    for sample in data_loader:
        data = sample['data'].cuda()
        
        with torch.no_grad():
            logit = classifier(data)
            energy_scores.extend((temperature * torch.logsumexp(logit / temperature, dim=1)).tolist())
    
    return energy_scores


scores_dic = {
    'msp': get_msp_scores,
    'kl': get_kl_scores,
    'odin': get_odin_scores,
    'maha': get_maha_scores,
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
    output_path = Path(args.output_dir) / args.id / args.output_sub_dir
    print('>>> Log dir: {}'.format(str(output_path)))
    output_path.mkdir(parents=True, exist_ok=True)
    
    _, std = get_dataset_info(args.id, 'mean_and_std')
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
    
    if args.scores == 'odin':
        id_scores = get_scores(classifier, id_loader, args.temperature, args.magnitude, std)
    elif args.scores == 'maha':
        num_layers = 1
        feature_dim_list = np.empty(num_layers)
        feature_dim_list[0] = 128  # 64 * widen_factor
    
        sample_mean, precision = sample_estimator(classifier, id_loader, num_classes, feature_dim_list)
        id_scores = get_maha_scores(classifier, id_loader, num_classes, sample_mean, precision, num_layers-1, args.magnitude, std)
    else:
        id_scores = get_scores(classifier, id_loader)
    id_label = np.zeros(len(id_scores))
    
    for ood_loader in ood_loaders:
        result_dic = {'name': ood_loader.dataset.name}
        
        if args.scores == 'odin':
            ood_scores = get_scores(classifier, ood_loader, args.temperature, args.magnitude, std)
        elif args.scores == 'maha':
            ood_scores = get_maha_scores(classifier, ood_loader, num_classes, sample_mean, precision, num_layers-1, args.magnitude, std)
        else:
            ood_scores = get_scores(classifier, ood_loader)
        ood_label = np.ones(len(ood_scores))
        # detect ood
        scores = np.concatenate([id_scores, ood_scores])
        labels = np.concatenate([id_label, ood_label])
        
        result_dic['fpr_at_tpr'], result_dic['auroc'], result_dic['aupr_in'], result_dic['aupr_out'] = compute_all_metrics(scores, labels, verbose=False)
        result_dic_list.append(result_dic)
        
        print('---> [ID: {:7s} - OOD: {:9s}] [auroc: {:3.3f}%, aupr_in: {:3.3f}%, aupr_out: {:3.3f}%, fpr@95tpr: {:3.3f}%]'.format(
            id_loader.dataset.name, ood_loader.dataset.name, 100. * result_dic['auroc'], 100. * result_dic['aupr_in'], 100. * result_dic['aupr_out'], 100. * result_dic['fpr_at_tpr']))
        
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
    parser = argparse.ArgumentParser(description='detect ood')
    parser.add_argument('--data_dir', type=str, default='/home/iip/datasets')
    parser.add_argument('--output_dir', help='dir to store log', default='logs')
    parser.add_argument('--output_sub_dir', help='sub dir to store log', default='tmp')
    parser.add_argument('--id', type=str, default='cifar10')
    parser.add_argument('--oods', nargs='+', default=['svhn', 'lsunc', 'dtd', 'places365_10k', 'cifar100', 'tinc', 'lsunr', 'tinr', 'isun'])
    parser.add_argument('--scores', type=str, default='msp')
    parser.add_argument('--temperature', type=int, default=1000)
    parser.add_argument('--magnitude', type=float, default=0.0014)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument('--classifier', type=str, default='wide_resnet')
    parser.add_argument('--classifier_path', type=str, default='./snapshots/cifar10/wrn.pth')
    parser.add_argument('--gpu_idx', type=int, default=0)

    args = parser.parse_args()

    main(args)