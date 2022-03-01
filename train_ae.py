import random
from pathlib import Path
from functools import partial
import argparse
import numpy as np
import copy
import time

import torch
import torch.backends.cudnn as cudnn

from datasets import get_ae_transforms, get_dataloader
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
    
    exp_path = Path(args.output_dir) / args.dataset / args.output_sub_dir
    print('>>> Exp dir: {}'.format(str(exp_path)))
    exp_path.mkdir(parents=True, exist_ok=True)
    
    setup_logger(str(exp_path), 'console.log')
    
    # ------------------------------------ Init Dataset ------------------------------------
    train_transform = get_ae_transforms('train')
    val_transform = get_ae_transforms('test')
    
    print('>>> Dataset: {}'.format(args.dataset))
    
    get_dataloader_default = partial(
        get_dataloader,
        root=args.data_dir,
        name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.prefetch
    )
        
    train_loader = get_dataloader_default(split='train', transform=train_transform, shuffle=True)
    val_loader = get_dataloader_default(split='test', transform=val_transform, shuffle=False)

    # ------------------------------------ Init Network ------------------------------------
    print('>>> AutoEncoder: {}'.format(args.arch))
    ae = get_ae(args.arch)
    
    # ------------------------------------ Init Trainer ------------------------------------
    print('>>> Optimizer: Adam  | Scheduler: None')
    betas = tuple([float(param) for param in args.betas])
    print('>>> Lr: {:.4f} | Weight_decay: {:.4f} | Betas: {}'.format(args.lr, args.weight_decay, args.betas))
    optimizer = torch.optim.Adam(ae.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=betas)
    scheduler = None
    
    trainer = get_ae_trainer(ae, train_loader, optimizer, scheduler)
    
    # move net to gpu device
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        ae.cuda()
    cudnn.benchmark = True

    # ------------------------------------ Start Training ------------------------------------
    evaluator = Evaluator(ae)
    begin_time = time.time()
    
    rec_best_err = float('inf')
    start_epoch = 1
    rec_best_state, last_state = {}, {}

    for epoch in range(start_epoch, args.epochs+1): 
        trainer.train_epoch()
        #  save intermediate reconstruction status
        evaluator.eval_rec(train_loader, epoch, exp_path / 'rec-train-imgs')
        val_metrics = evaluator.eval_rec(val_loader, epoch, exp_path / 'rec-val-imgs')
        
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
                'rec_err': val_metrics['rec_loss']
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
    parser.add_argument('--data_dir', help='directory to store datasets', default='/home/iip/datasets')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--output_sub_dir', type=str, default='res_ae')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--betas', nargs='+', default=[0.9, 0.999])
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--prefetch', type=int, default=4, help='number of dataloader workers')
    parser.add_argument('--arch', type=str, default='res_ae')
    parser.add_argument('--gpu_idx', type=int, default=0)
    args = parser.parse_args()
    
    main(args)