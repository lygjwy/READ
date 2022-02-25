import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
import argparse
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn

from datasets import get_transforms, get_ood_transforms, get_dataset_info, get_dataloader
from models import get_deconf_net
from evaluation import compute_all_metrics


def get_godin_scores(deconf_net, data_loader, magnitude=0.0010, score_func='h'):
    deconf_net.eval()
    
    godin_scores = []
    
    for sample in data_loader:
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
        
        gradient[:, 0] = gradient[:, 0] / (63.0 / 255)
        gradient[:, 1] = gradient[:, 1] / (62.1 / 255)
        gradient[:, 2] = gradient[:, 2] / (66.7 / 255)
        
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
    
    #  load deconf net
    num_classes = len(get_dataset_info(args.id.split('-')[0], 'classes'))
    print('>>> Deconf: {} - {}'.format(args.feature_extractor, args.h))
    deconf_net = get_deconf_net(args.feature_extractor, args.h, num_classes)
    deconf_path = Path(args.deconf_path)
    
    if deconf_path.exists():
        deconf_params = torch.load(str(deconf_path))
        cla_acc = deconf_params['cla_acc']
        deconf_net.load_state_dict(deconf_params['state_dict'])
        print('>>> load deconf net from {} (classifiy acc {:.4f}%)'.format(str(deconf_path), cla_acc))
    else:
        raise RuntimeError('<--- invalid deconf path: {}'.format(str(deconf_path)))
    
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        deconf_net.cuda()
    cudnn.benchmark = True
    
    get_scores = scores_dic[args.scores]
    result_dic_list = []
    
    if args.scores == 'godin':
        id_scores = get_scores(deconf_net, id_loader, args.magnitude, args.score_func)
    else:
        id_scores = get_scores(deconf_net, id_loader)
    
    id_label = np.zeros(len(id_scores))
    
    for ood_loader in ood_loaders:
        result_dic = {'name': ood_loader.dataset.name}
        
        if args.scores == 'godin':
            ood_scores = get_scores(deconf_net, ood_loader, args.magnitude, args.score_func)
        else:
            ood_scores = get_scores(deconf_net, ood_loader)
        
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
    parser.add_argument('--feature_extractor', type=str, default='wide_resnet')
    parser.add_argument('--h', type=str, default='cosine')
    parser.add_argument('--deconf_path', type=str, default='./snapshots/w-c.pth')
    parser.add_argument('--id', type=str, default='cifar10')
    parser.add_argument('--oods', nargs='+', default=['svhn', 'cifar100', 'tinc', 'tinr', 'lsunc', 'lsunr', 'dtd', 'places365_10k', 'isun'])
    parser.add_argument('--scores', type=str, default='godin')
    parser.add_argument('--magnitude', type=float, default=0.0)
    parser.add_argument('--score_func', type=str, default='h')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument('--gpu_idx', type=int, default=0)

    args = parser.parse_args()

    main(args)