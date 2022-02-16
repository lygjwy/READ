import random
from pathlib import Path
from functools import partial
import argparse
import numpy as np
import copy
import time

import torch
import torch.backends.cudnn as cudnn

from datasets import get_ae_transform, get_ae_normal_transform, get_dataloader, get_corrupt_dataloader
from models import get_ae
from trainers import get_ae_trainer
from evaluation import Evaluator
from utils import setup_logger


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def main(args):
    init_seeds(args.seed)
    
    exp_path = Path(args.output_dir) / args.output_sub_dir
    print('>>> Exp dir: {}'.format(str(exp_path)))
    exp_path.mkdir(parents=True, exist_ok=True)
    
    setup_logger(str(exp_path), 'console.log')
    
    # ------------------------------------ Init Dataset ------------------------------------
    train_transform = get_ae_transform('train')
    val_transform = get_ae_transform('test')
    normal_transform = get_ae_normal_transform()
    
    print('>>> Dataset: {} with data mode: {}'.format(args.dataset, args.data_mode))
    
    if args.data_mode == 'original':
        get_dataloader_default = partial(
            get_dataloader,
            root=args.data_dir,
            name=args.dataset,
            batch_size=args.batch_size,
            num_workers=args.prefetch
        )
        
        train_loader = get_dataloader_default(split='train', transform=train_transform, shuffle=True)
        val_loader = get_dataloader_default(split='test', transform=val_transform, shuffle=False)
    elif args.data_mode == 'corrupt':
        get_dataloader_default = partial(
            get_corrupt_dataloader,
            root=args.data_dir,
            name=args.dataset,
            corrupt=args.corrupt,
            severity=args.severity,
            batch_size=args.batch_size,
            num_workers=args.prefetch
        )
        
        train_loader = get_dataloader_default(split='train', random=True, transform=normal_transform, shuffle=True)
        val_loader = get_dataloader_default(split='test', random=False, transform=normal_transform, shuffle=False)
    else:
        raise RuntimeError('<--- invalid data mode: {}'.format(args.data_mode))
    
    # ------------------------------------ Init Network ------------------------------------
    print('>>> AutoEncoder: {}'.format(args.arch))
    ae = get_ae(args.arch)
    
    # ------------------------------------ Init Trainer ------------------------------------
    if args.optimizer == 'sgd':
        print('>>> Optimizer: {} | Lr: {:.4f} | Weight_decay: {:.4f} | Momentum: {:.4f}'.format(args.optimizer, args.lr, args.weight_decay, args.momentum))
        optimizer = torch.optim.SGD(ae.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optimizer == 'adam':
        betas = tuple([float(param) for param in args.betas])
        print('>>> Optimizer: {} | Lr: {:.4f} | Weight_decay: {:.4f} | Betas: {}'.format(args.optimizer, args.lr, args.weight_decay, args.betas))
        optimizer = torch.optim.Adam(ae.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=betas)
    else:
        raise RuntimeError('---> invalid optimizer: {}'.format(args.optimizer))
    
    if args.scheduler == 'lambdalr':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                args.epochs * len(train_loader),
                1,
                1e-6 / args.lr
            )
        )
    elif args.scheduler == 'none':
        scheduler = None
    else:
        raise RuntimeError('---> invalid scheduler: {}'.format(args.scheduler))
    
    trainer = get_ae_trainer(ae, train_loader, optimizer, scheduler, args.data_mode)
    
    # move net to gpu device
    rec_best_err = float('inf')
    start_epoch = 1
    
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        ae.cuda()
    cudnn.benchmark = True
    
    # ------------------------------------ Start Training ------------------------------------
    evaluator = Evaluator(ae)
    begin_time = time.time()
    
    rec_best_state, last_state = {}, {}
    
    for epoch in range(start_epoch, args.epochs+1): 
        trainer.train_epoch()
        #  save intermediate reconstruction status
        if args.data_mode == 'original':
            evaluator.eval_rec(train_loader, epoch, exp_path / 'rec-train-imgs')
            val_metrics = evaluator.eval_rec(val_loader, epoch, exp_path / 'rec-val-imgs')
        elif args.data_mode == 'corrupt':
            evaluator.eval_cor_rec(train_loader, epoch, exp_path / 'cor_rec-train-imgs')
            val_metrics = evaluator.eval_cor_rec(val_loader, epoch, exp_path / 'cor_rec-val-imgs')
        else:
            raise RuntimeError('<--- invalid data mode: {}'.format(args.data_mode))
        
        rec_best = val_metrics['rec_err'] < rec_best_err
        rec_best_err = min(val_metrics['rec_err'], rec_best_err)
        
        if epoch == args.epochs:
            last_state = {
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': ae.state_dict(),
                'rec_err': val_metrics['rec_err']
            }
        
        if rec_best:
            rec_best_state = {
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': copy.deepcopy(ae.state_dict()),
                'rec_err': val_metrics['rec_err']
            }

        print(
            "---> Epoch {:4d} | Time {:5d}s".format(
                epoch,
                int(time.time() - begin_time)
            ),
            flush=True
        )
    
    # ------------------------------------ Train Done Save Model ------------------------------------
    rec_best_path = exp_path / 'rec_best.pth'
    torch.save(rec_best_state, str(rec_best_path))
    last_path = exp_path / 'last.pth'
    torch.save(last_state, str(last_path))
    print('---> Best rec error: {:.6f}'.format(rec_best_err))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training AutoEncoder')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--output_sub_dir', type=str, default='res_ae')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--betas', nargs='+', default=[0.9, 0.999])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--scheduler', type=str, default='none')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--data_dir', help='directory to store datasets', default='data/datasets')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--data_mode', type=str, default='original')
    parser.add_argument('--corrupt', type=str, default='gaussian_noise')
    parser.add_argument('--severity', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--prefetch', type=int, default=4, help='number of dataloader workers')
    parser.add_argument('--arch', type=str, default='res_ae')
    parser.add_argument('--gpu_idx', type=int, default=0)
    args = parser.parse_args()
    main(args)