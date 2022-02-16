import ast
from pathlib import Path
from functools import partial
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn

from datasets import get_ae_transform
from datasets.utils import get_dataloader, get_corrupt_dataloader
from models import get_ae


def main(args):
    dataset = args.dataset
    ae_transform = get_ae_transform('test')
    
    if args.data_mode == 'original':
        get_dataloader_default = partial(
            get_dataloader,
            root=args.data_dir,
            split=args.split,
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
            split=args.split,
            transform=ae_transform,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.prefetch
        )
    else:
        raise RuntimeError('---> invlaid data mode: {}'.format(args.data_mode))
    
    dataset_root = Path(args.data_dir)
    original_dataset_path = dataset_root / dataset
    original_dataset_classes_path = original_dataset_path / 'classes.txt'
    
    if original_dataset_classes_path.exists():
        with open(original_dataset_classes_path) as f:
            classes = sorted(ast.literal_eval(f.readline()))
    else:
        raise RuntimeError('---> non-existed classes.txt path: {}'.format(original_dataset_classes_path))
    
    if args.data_mode == 'original':
        dataset_dir = dataset_root / (dataset + '-or') / args.split
    elif args.data_mode == 'corrupt':
        dataset_dir = dataset_root / (dataset + '-cor') / args.split
    else:
        raise RuntimeError('---> invalid data mode: '.format(args.data_mode))
    
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # load dataset
    data_loader = get_dataloader_default(name=dataset)
    
    ae = get_ae(args.arch)
    ae_path = Path(args.ae_path)
    
    if ae_path.exists():
        ae_params = torch.load(str(ae_path))
        rec_err = ae_params['rec_err']
        ae.load_state_dict(ae_params['state_dict'])
        print('>>> load best ae from {} (rec err {:.4f})'.format(str(ae_path), rec_err))
    else:
        print('---> invalid ae path: {}'.format(str(ae_path)))
    
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        ae.cuda()
    cudnn.benchmark = True
    
    total_loss = 0.0
    ae.eval()
    
    # evaluator = Evaluator(ae)
    if args.data_mode == 'original':
        data_list, rec_data_list, target_list = [], [], []
        
        with torch.no_grad():
            for sample in data_loader:
                data, target = sample
                target_list.extend(target.tolist())
                data = data.cuda()
                rec_data = ae(data)
                
                rec_loss = F.mse_loss(rec_data, data, reduction='sum')
                total_loss += rec_loss.item()
                data_list.append(data)
                rec_data_list.append(rec_data)
            
            data = torch.cat(data_list, dim=0)
            rec_data = torch.cat(rec_data_list, dim=0)
        
        #  make dir
        for cla in classes:
            cla_path = dataset_dir / cla
            cla_path.mkdir(parents=True, exist_ok=True)
        cla_counts = {cla: 0 for cla in classes}
        
        for i, target in enumerate(target_list):
            category = classes[target]
            save_image(data[i].cpu(), str(dataset_dir / category / (str(cla_counts[category]) + '.png')))
            save_image(rec_data[i].cpu(), str(dataset_dir / category / (str(cla_counts[category]) + 'r.png')))
            cla_counts[category] += 1
    
    elif args.data_mode == 'corrupt':
        cor_data_list, data_list, rec_data_list, target_list = [], [], [], []
        with torch.no_grad():
            for sample in data_loader:
                cor_data, data, target = sample
                target_list.extend(target.tolist())
                cor_data, data = cor_data.cuda(), data.cuda()
                
                rec_data = ae(data)
                
                rec_loss = F.mse_loss(rec_data, data, reduction='sum')
                total_loss += rec_loss.item()
                cor_data_list.append(cor_data)
                data_list.append(data)
                rec_data_list.append(rec_data)
        
        cor_data = torch.cat(cor_data_list, dim=0)
        data = torch.cat(data_list, dim=0)
        rec_data = torch.cat(rec_data_list, dim=0)
        
        #  make dir
        for cla in classes:
            cla_path = dataset_dir / cla
            cla_path.mkdir(parents=True, exist_ok=True)
        cla_counts = {cla: 0 for cla in classes}
        
        for i, target in enumerate(target_list):
            category = classes[target]
            save_image(data[i].cpu(), str(dataset_dir / category / (str(cla_counts[category]) + '.png')))
            save_image(rec_data[i].cpu(), str(dataset_dir / category / (str(cla_counts[category]) + 'r.png')))
            save_image(cor_data[i].cpu(), str(dataset_dir / category / (str(cla_counts[category]) + 'c.png')))
            cla_counts[category] += 1
    else:
        raise RuntimeError('---> invalid data mode: {}'.format(args.data_mode))

    print('[rec_loss: {:.4f}]'.format(total_loss / len(data_loader.dataset)))
# end for loop

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoencoder reconstruction')
    parser.add_argument('--data_dir', type=str, default='./datasets')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--split', type=str, default='test')
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