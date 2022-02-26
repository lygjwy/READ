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
from models import get_ae, get_deconf_net
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
    

def get_hybrid_scores(ae, deconf_net, data_loader, normalize, h='cosine'):
    ae.eval()
    deconf_net.eval()
    
    ori_scores, rec_scores, similarity_scores, hybrid_scores = [], [], [], []
    
    for sample in data_loader:
        if data_loader.dataset.labeled:
            data, _ = sample
        else:
            data = sample
        data = data.cuda()
        
        with torch.no_grad():
            rec_data = ae(data)
            
            data = torch.stack([normalize(img) for img in data], dim=0)
            rec_data = torch.stack([normalize(img) for img in rec_data], dim=0)
        
            penultimate_feature = deconf_net.intermediate_forward(data)
            h = deconf_net.h(penultimate_feature)
            rec_penultimate_feature = deconf_net.intermediate_forward(rec_data)
            rec_h = deconf_net.h(rec_penultimate_feature)
        
        ori_score, cla_idx = torch.max(h, dim=1)
        ori_scores.extend(ori_score.tolist())
        
        # the same category
        cla_idx = torch.unsqueeze(cla_idx, 0).t()  # column vector
        rec_score = torch.gather(rec_h, dim=1, index=cla_idx)
        rec_scores.extend(torch.squeeze(rec_score).tolist())
        
        # calculate the difference between penulitimate_feature & rec_penultimate_feature
        if args.h == 'cosine':
            similarity = torch.cosine_similarity(penultimate_feature, rec_penultimate_feature, dim=1)
        elif args.h == 'euclidean':
            similarity = -((penultimate_feature - rec_penultimate_feature).pow(2)).mean(1)
        elif args.h == 'inner':
            similarity = torch.bmm(penultimate_feature.view(args.batch_size, 1, -1), rec_penultimate_feature.view(args.batch_size, -1, 1))
            similarity = torch.squeeze(similarity)
        else:
            raise RuntimeError('<--- invalid h: '.format(args.h))
        
        similarity_scores.extend(similarity.tolist())
    
    # combine ori_scores & similarity_scores
    # ? how to use image complexity as weight
    for ori_score, similarity_score in zip(ori_scores, similarity_scores):
        hybrid_scores.append(ori_score + 0.5 * similarity_score)
    # for ori_score, rec_score in zip(ori_scores, rec_scores):
    #     hybrid_scores.append(ori_score + rec_score)
    return hybrid_scores


scores_dic = {
    'hybrid': get_hybrid_scores
}


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
        ood_ae_transform = get_ae_ood_transforms(ood, 'test')
        ood_loaders.append(get_dataloader_default(name=ood, transform=ood_ae_transform))

    #  -------------------- ae & deconf -------------------- #
    print('>>> Ae: {}'.format(args.ae))
    ae = get_ae(args.ae)
    ae_path = Path(args.ae_path)
    num_classes = len(get_dataset_info(args.id.split('-')[0], 'classes'))
    print('>>> Deconf: {} - {}'.format(args.feature_extractor, args.h))
    deconf_net = get_deconf_net(args.feature_extractor, args.h, num_classes)
    deconf_path = Path(args.deconf_path)
    
    if ae_path.exists():
        ae_params = torch.load(str(ae_path))
        rec_err = ae_params['rec_err']
        ae.load_state_dict(ae_params['state_dict'])
        print('>>> load ae from {} (rec err {})'.format(str(ae_path), rec_err))
    else:
        raise RuntimeError('---> invalid ae path: {}'.format(str(ae_path)))
    
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
        ae.cuda()
        deconf_net.cuda()
    cudnn.benchmark = True

    get_scores = scores_dic[args.scores]
    result_dic_list = []
    
    id_scores = get_scores(ae, deconf_net, id_loader, normalize, h=args.h)
    id_label = np.zeros(len(id_scores))
    
    for ood_loader in ood_loaders:
        result_dic = {'name': ood_loader.dataset.name}
        
        ood_scores = get_scores(ae, deconf_net, ood_loader, normalize, h=args.h)
        ood_label = np.ones(len(ood_scores))
        
        scores = np.concatenate([id_scores, ood_scores])
        labels = np.concatenate([id_label, ood_label])
        
        result_dic['fpr_at_tpr'], result_dic['auroc'], result_dic['aupr_in'], result_dic['aupr_out'] = compute_all_metrics(scores, labels, verbose=False)
        result_dic_list.append(result_dic)
        
        print('---> [ID: {:7s} - OOD: {:9s}] [auroc: {:3.3f}%, aupr_in: {:3.3f}%, aupr_out: {:3.3f}%, fpr@95tpr: {:3.3f}%]'.format(
            id_loader.dataset.name, ood_loader.dataset.name, 100. * result_dic['auroc'], 100. * result_dic['aupr_in'], 100. * result_dic['aupr_out'], 100. * result_dic['fpr_at_tpr']))
        
    # save result
    result = pd.DataFrame(result_dic_list)
    log_path = output_path / (args.scores + '-' + args.h + '.csv')
    result.to_csv(str(log_path), index=False, header=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='detect ood')
    parser.add_argument('--data_dir', type=str, default='/home/iip/datasets')
    parser.add_argument('--id', type=str, default='cifar10')
    parser.add_argument('--oods', nargs='+', default=['svhn', 'cifar100', 'tinc', 'tinr', 'lsunc', 'lsunr', 'dtd', 'places365_10k', 'isun'])
    parser.add_argument('--output_dir', help='dir to store log', default='logs')
    parser.add_argument('--output_sub_dir', help='sub dir to store log', default='tmp')
    parser.add_argument('--ae', type=str, default='res_ae')
    parser.add_argument('--ae_path', type=str, default='./snapshots/r.pth')
    parser.add_argument('--feature_extractor', type=str, default='wide_resnet')
    parser.add_argument('--h', type=str, default='cosine')
    parser.add_argument('--deconf_path', type=str, default='./snapshots/w-c.pth')
    parser.add_argument('--scores', type=str, default='hybrid')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument('--gpu_idx', type=int, default=0)

    args = parser.parse_args()

    main(args)