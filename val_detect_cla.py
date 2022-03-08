''' Tune the ood detector's hyper-parameters
'''
import numpy as np
from pathlib import Path
from functools import partial
import argparse
import sklearn.covariance

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from torchvision import transforms

from datasets import get_transforms, get_dataset, get_dataset_info
from datasets import get_dataloader, get_uniform_noise_dataloader
from datasets import AvgOfPair, GeoMeanOfPair
from datasets import get_shift_transform
from models import get_classifier
from evaluation import compute_all_metrics


def get_odin_scores(classifier, data_loader, temperature, magnitude, std):
    classifier.eval()
    
    odin_scores = []
    
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
        if isinstance(sample, dict):
                data = sample['data']
        else:
            if data_loader.dataset.labeled:
                data, _ = sample
            else:
                data = sample
        data = data.cuda()
        
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


scores_dic = {
    'odin': get_odin_scores,
    'maha': get_maha_scores
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
    
    # load classifier
    num_classes = len(get_dataset_info(args.id, 'classes'))
    classifier = get_classifier(args.classifier, num_classes)
    classifier_path = Path(args.classifier_path)
    
    if classifier_path.exists():
        cla_params = torch.load(str(classifier_path))
        # cla_acc = cla_params['cla_acc']
        classifier.load_state_dict(cla_params['state_dict'])
        # print('>>> load classifier from {} (classify acc {:.4f}%)'.format(str(classifier_path), cla_acc))
    else:
        raise RuntimeError('<--- invalid classifier path: {}'.format(str(classifier_path)))
    
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        classifier.cuda()
    cudnn.benchmark = True
    
    # ------------------------------------ detect ood ------------------------------------
    get_scores = scores_dic[args.scores]
    fpr_at_tprs, aurocs, aupr_ins, aupr_outs = [], [], [], []
    
    if args.scores == 'odin':
        id_scores = get_scores(classifier, id_loader, args.temperature, args.magnitude, std)
    elif args.scores == 'maha':
        num_layers = 1
        feature_dim_list = np.empty(num_layers)
        feature_dim_list[0] = 128  # 64 * widen_factor
    
        sample_mean, precision = sample_estimator(classifier, id_loader, num_classes, feature_dim_list)
        
        id_scores = get_maha_scores(classifier, id_loader, num_classes, sample_mean, precision, num_layers - 1, args.magnitude, std)
    else:
        raise RuntimeError('<--- invalid scores: '.format(args.scores))
    
    # another validation metrics
    avg_score = np.mean(id_scores)
    id_label = np.zeros(len(id_scores))
    
    for ood_loader in ood_loaders:
        if args.scores == 'odin':
            ood_scores = get_scores(classifier, ood_loader, args.temperature, args.magnitude, std)
        elif args.scores == 'maha':
            ood_scores = get_maha_scores(classifier, ood_loader, num_classes, sample_mean, precision, num_layers-1, args.magnitude, std)
        else:
            raise RuntimeError('<--- invalid scores: '.format(args.scores))
        ood_label = np.ones(len(ood_scores))
        
        scores = np.concatenate([id_scores, ood_scores])
        labels = np.concatenate([id_label, ood_label])
        
        fpr_at_tpr, auroc, aupr_in, aupr_out = compute_all_metrics(scores, labels, verbose=False)
        
        fpr_at_tprs.append(fpr_at_tpr)
        aurocs.append(auroc)
        aupr_ins.append(aupr_in)
        aupr_outs.append(aupr_out)
    
    if args.scores == 'odin':
        print('---> [Temperature: {:.4f}, Magnitude: {:.4f}] [avg_score: {:.4f} | avg auroc: {:.4f} | avg fpr_at_tpr: {:.4f} | avg aupr_in: {:.4f} | avg aupr_out: {:.4f}]'.format(
                args.temperature,
                args.magnitude,
                avg_score,
                np.mean(aurocs),
                np.mean(fpr_at_tprs),
                np.mean(aupr_ins),
                np.mean(aupr_outs)
            )
        )
    
    if args.scores == 'maha':
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
    parser.add_argument('--scores', type=str, default='odin')
    parser.add_argument('--temperature', type=int, default=1000)
    parser.add_argument('--magnitude', type=float, default=0.0014)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument('--classifier', type=str, default='wide_resnet')
    parser.add_argument('--classifier_path', type=str, default='./snapshots/cifar10/wrn.pth')
    parser.add_argument('--gpu_idx', type=int, default=0)
    
    args = parser.parse_args()

    main(args)
