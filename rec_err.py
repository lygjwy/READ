from pathlib import Path
from functools import partial
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn

from datasets import get_ae_transform, get_ae_normal_transform, get_dataloader, get_corrupt_dataloader
from models import get_ae


def get_rec_errs(ae, data_loader):
    ae.eval()
    
    total_loss = 0.0
    rec_errs = []
    for sample in data_loader:
        if type(sample) == list:
            # labeled
            data, _ = sample
        else:
            # unlabeled
            data = sample
        data = data.cuda()
        with torch.no_grad():
            rec_data = ae(data)
        # rec_err = F.mse_loss(rec_data, data, reduction='sum')
        rec_err = torch.sum(F.mse_loss(rec_data, data, reduction='none'), dim=[1, 2, 3])
        rec_errs.extend(rec_err.tolist())
        total_loss += rec_err.sum().item()

    if data_loader.dataset.name == 'cifar10':
        print('loss: {:.6f}'.format(total_loss / len(data_loader.dataset)))
    return rec_errs


def get_cor_rec_errs(ae, data_loader):
    ae.eval()
    
    rec_errs = []
    # rec_errs = []
    for sample in data_loader:
        if len(sample) == 3:
            # labeled
            cor_data, data, _ = sample
        elif len(sample) == 2:
            # unlabeled
            cor_data, data = sample
        else:
            raise RuntimeError('<--- invalid sample length: {}'.format(len(sample)))
        cor_data, data = cor_data.cuda(), data.cuda()
        with torch.no_grad():
            rec_data = ae(cor_data)
            # cor_rec_data = ae(data)
        rec_err = torch.sum(F.mse_loss(rec_data, data, reduction='none'), dim=[1, 2, 3])
        rec_errs.extend(rec_err.tolist())
        
    # return rec_errs, cor_rec_errs 
    return rec_errs


def save_rec_imgs(ae, data_loader, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    ae.eval()
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(data_loader):
            if type(sample) == list:
                # labeled
                data, _ = sample
            else:
                # unlabeled
                data = sample
            data = data.cuda()
            
            rec_data = ae(data)
            # save original-reconstruct images
            if batch_idx == 1:
                n = min(data.size(0), 16)
                comparison = torch.cat(
                    [data[:n], rec_data.view(-1, 3, 32, 32)[:n]]
                )
                rec_path = output_dir / ('-'.join([data_loader.dataset.name, 'rec_imgs']) + '.png')
                save_image(comparison.cpu(), rec_path, nrow=16)
                return None
            

def save_cor_rec_imgs(ae, data_loader, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    ae.eval()
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(data_loader):
            if len(sample) == 2:
                cor_data, data = sample # unlabeled
            elif len(sample) == 3:
                cor_data, data, _ = sample # labeled
            else:
                raise RuntimeError('---> invalid sample length: {}'.format(len(sample)))
            
            cor_data, data = cor_data.cuda(), data.cuda()
            rec_data = ae(cor_data)
            
            # save ori-cor-rec images
            if batch_idx == 1:
                n = min(data.size(0), 16)
                comparison = torch.cat(
                    [data[:n], cor_data[:n], rec_data.view(-1, 3, 32, 32)[:n]]
                )
                rec_path = output_dir / ('-'.join([data_loader.dataset.name, 'cor-rec_imgs']) + '.png')
                save_image(comparison.cpu(), rec_path, nrow=n)
                return None


def draw_hist(data, colors, labels, title, fig_path):
    plt.clf()
    bins = list(range(125))
    plt.hist(data, bins, density=True, histtype='bar', color=colors, label=labels, alpha=1)
    plt.xlabel('reconstruction error')
    plt.ylabel('density')
    plt.legend(prop={'size': 10})
    plt.title(title)
    plt.savefig(fig_path)
    plt.close()


def main(args):
    output_path = Path(args.output_dir)
    ae_transform = get_ae_transform('test')
    ae_normal_transform = get_ae_normal_transform()
    
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
            random=False,
            transform=ae_normal_transform,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.prefetch
        )
    else:
        raise RuntimeError('<--- invalid data mode: {}'.format(args.data_mode))
    
    id_loader = get_dataloader_default(name=args.id)
    ood_loaders = []
    for ood in args.oods:
        ood_loaders.append(get_dataloader_default(name=ood))
    
    #  load ae
    ae = get_ae(args.arch)
    ae_path = Path(args.ae_path)
    
    if ae_path.exists():
        ae_params = torch.load(str(ae_path))
        rec_err = ae_params['rec_err']
        ae.load_state_dict(ae_params['state_dict'])
        print('>>> load ae from {} (rec err {:.6f})'.format(str(ae_path), rec_err))
    else:
        raise RuntimeError('<--- invalid ae path: {}'.format(str(ae_path)))
    
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        ae.cuda()
    cudnn.benchmark = True
    
    if args.data_mode == 'original':
        #  save rec images
        save_rec_imgs(ae, id_loader, output_path)
        ori_id_rec_errs = get_rec_errs(ae, id_loader)
        
        for ood_loader in ood_loaders:
            # save rec images
            save_rec_imgs(ae, ood_loader, output_path)
            ori_ood_rec_errs = get_rec_errs(ae, ood_loader)
            # plot hist
            rec_errs = [ori_id_rec_errs, ori_ood_rec_errs]
            colors = ['lime', 'orangered']
            labels = ['id', 'ood']
            title = '-'.join([ood_loader.dataset.name, args.id, 'rec_err'])
            fig_path = output_path / (title + '.png')
            draw_hist(rec_errs, colors, labels, title, fig_path)
    elif args.data_mode == 'corrupt':
        #  save rec images
        save_cor_rec_imgs(ae, id_loader, output_path)
        id_rec_errs = get_cor_rec_errs(ae, id_loader)
        
        for ood_loader in ood_loaders:
            #  save rec_images
            save_cor_rec_imgs(ae, ood_loader, output_path)
            ood_rec_errs = get_cor_rec_errs(ae, ood_loader)
            # plot hist
            rec_errs = [id_rec_errs, ood_rec_errs]
            colors = ['lime', 'cyan']
            labels = ['id', 'ood']
            title = '-'.join([ood_loader.dataset.name, args.id, 'corrupt-rec_err'])
            fig_path = output_path / (title + '.png')
            draw_hist(rec_errs, colors, labels, title, fig_path)
    else:
        raise RuntimeError('---> invalid data mode: {}'.format(args.data_mode))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoencoder reconstruction')
    parser.add_argument('--data_dir', type=str, default='./datasets')
    parser.add_argument('--output_dir', help='dir to store experiment artifacts', default='outputs')
    parser.add_argument('--id', type=str, default='cifar10')
    parser.add_argument('--oods', nargs='+', default=['svhn', 'cifar100', 'tinc', 'tinr', 'lsunc', 'lsunr', 'dtd', 'places365', 'isun'])
    parser.add_argument('--data_mode', type=str, default='original')
    parser.add_argument('--corrupt', type=str, default='gaussian_noise')
    parser.add_argument('--severity', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument('--arch', type=str, default='res_ae')
    parser.add_argument('--ae_path', type=str, default='./outputs/res_ae/rec_best.pth')
    parser.add_argument('--gpu_idx', type=int, default=0)
    
    args = parser.parse_args()

    main(args)
    
    
        