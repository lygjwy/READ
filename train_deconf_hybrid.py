# train deconf net using ori & rec imgs
import time
import copy
from pathlib import Path
import argparse
import random
import numpy as np
from functools import partial

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from datasets import get_transforms, get_dataset_info, get_hybrid_dataloader
from models import get_deconf_net
from trainers import get_deconf_hybrid_trainer
from evaluation import Evaluator
from utils import  setup_logger


def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    init_seeds(args.seed)
    
    # store net and console log by training method
    exp_path = Path(args.output_dir) / args.dataset / args.output_sub_dir
    print('>>> Exp dir: {} '.format(str(exp_path)))
    exp_path.mkdir(parents=True, exist_ok=True)

    # record console output
    setup_logger(str(exp_path), 'console.log')

    # ------------------------------------ Init Datasets ------------------------------------
    ## get dataset transform
    train_transform = get_transforms(args.dataset.split('-')[0], stage='train')
    val_transform = get_transforms(args.dataset.split('-')[0], stage='test')  # using train set's mean&std

    print('>>> Dataset: {}'.format(args.dataset))
    
    ## get dataloader
    get_dataloader_default = partial(
        get_hybrid_dataloader,
        root=args.data_dir,
        name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.prefetch
    )
    
    train_loader = get_dataloader_default(
        split='train',
        transform=train_transform,
        shuffle=True
    )

    test_loader = get_dataloader_default(
        split='test',
        transform=val_transform,
        shuffle=False
    )
    
    # ------------------------------------ Init Classifier ------------------------------------
    num_classes = len(get_dataset_info(args.dataset.split('-')[0], 'classes'))
    print('>>> Deconf: {} - {}'.format(args.feature_extractor, args.h))
    deconf_net = get_deconf_net(args.feature_extractor, args.h, num_classes)
    
    if args.pretrained:
        #  load pretrain model
        pretrain_path = Path(args.pretrain_path)
        if pretrain_path.exists():
            deconf_params = torch.load(str(pretrain_path))
            cla_acc = deconf_params['cla_acc']
            deconf_net.load_state_dict(deconf_params['state_dict'])
            print('>>> load pretrained deconf net from {} (classification acc {:.4f}%)'.format(str(pretrain_path), cla_acc))
        else:
            raise RuntimeError('<--- invalid pretrained deconf net path: {}'.format(str(pretrain_path)))
    
    # move deconf_net to gpu device
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        deconf_net.cuda()
    cudnn.benchmark = True
    
    parameters = []
    h_parameters = []
    for name, parameter in deconf_net.named_parameters():
        if name == 'h.h.weight' or name == 'h.h.bias':
            h_parameters.append(parameter)
        else:
            parameters.append(parameter)

    # ------------------------------------ Init Trainer ------------------------------------
    print('>>> Lr: {:.5f} | Weight_decay: {:.5f} | Momentum: {:.2f}'.format(args.lr, args.weight_decay, args.momentum))
    optimizer = optim.SGD(parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)], gamma=0.1)
    
    h_optimizer = optim.SGD(h_parameters, lr=args.lr, momentum=args.momentum) # no weight_decay
    h_scheduler = optim.lr_scheduler.MultiStepLR(h_optimizer, milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)], gamma = 0.1)
    
    trainer = get_deconf_hybrid_trainer(deconf_net, train_loader, optimizer, h_optimizer, scheduler, h_scheduler)
    
    # ------------------------------------ Start training ------------------------------------
    evaluator = Evaluator(deconf_net)
    begin_time = time.time()
    
    start_epoch = 1
    cla_best_acc, rec_cla_best_acc, hybrid_cla_best_acc = 0.0, 0.0, 0.0
    cla_best_state, rec_cla_best_state, hybrid_cla_best_state, last_state = {}, {}, {}, {}
    
    for epoch in range(start_epoch, args.epochs+1):
    
        trainer.train_epoch()
        val_metrics = evaluator.eval_deconf_hybrid_classification(test_loader)
        
        cla_best = val_metrics['cla_acc'] > cla_best_acc
        cla_best_acc = max(val_metrics['cla_acc'], cla_best_acc)
        
        rec_cla_best = val_metrics['rec_cla_acc'] > rec_cla_best_acc
        rec_cla_best_acc = max(val_metrics['rec_cla_acc'], rec_cla_best_acc)
        
        hybrid_cla_best = val_metrics['hybrid_cla_acc'] > hybrid_cla_best_acc
        hybrid_cla_best_acc = max(val_metrics['hybrid_cla_acc'], hybrid_cla_best_acc)
        
        if epoch == args.epochs:
            last_state = {
                'epoch': epoch,
                'feature_extractor': args.feature_extractor,
                'h': args.h,
                'state_dict': deconf_net.state_dict(),
                'cla_acc': val_metrics['cla_acc'],
                'rec_cla_acc': val_metrics['rec_cla_acc'],
                'hybrid_cla_acc': val_metrics['hybrid_cla_acc']
            }
        
        if cla_best:
            cla_best_state = {
                'epoch': epoch,
                'feature_extractor': args.feature_extractor,
                'h': args.h,
                'state_dict': copy.deepcopy(deconf_net.state_dict()),
                'cla_acc': val_metrics['cla_acc'],
                'rec_cla_acc': val_metrics['rec_cla_acc'],
                'hybrid_cla_acc': val_metrics['hybrid_cla_acc']
            }
        
        if rec_cla_best:
            rec_cla_best_state = {
                'epoch': epoch,
                'feature_extractor': args.feature_extractor,
                'h': args.h,
                'state_dict': copy.deepcopy(deconf_net.state_dict()),
                'cla_acc': val_metrics['cla_acc'],
                'rec_cla_acc': val_metrics['rec_cla_acc'],
                'hybrid_cla_acc': val_metrics['hybrid_cla_acc']
            }

        if hybrid_cla_best:
            hybrid_cla_best_state = {
                'epoch': epoch,
                'feature_extractor': args.feature_extractor,
                'h': args.h,
                'state_dict': copy.deepcopy(deconf_net.state_dict()),
                'cla_acc': val_metrics['cla_acc'],
                'rec_cla_acc': val_metrics['rec_cla_acc'],
                'hybrid_cla_acc': val_metrics['hybrid_cla_acc']
            }

        
        print(
            "---> Epoch {:4d} | Time {:5d}s".format(
                epoch,
                int(time.time() - begin_time)
            ),
            flush=True
        )

    # ------------------------------------ Trainig done, save model ------------------------------------
    cla_best_path = exp_path / 'cla_best.pth'
    torch.save(cla_best_state, str(cla_best_path))
    rec_cla_best_path = exp_path / 'rec_cla_best.pth'
    torch.save(rec_cla_best_state, str(rec_cla_best_path))
    hybrid_cla_best_path = exp_path / 'cla_best.pth'
    torch.save(hybrid_cla_best_state, str(hybrid_cla_best_path))
    last_path = exp_path / 'last.pth'
    torch.save(last_state, str(last_path))
    print('---> Best cla acc: {:.4f}% | rec cla acc: {:.4f}% | hybrid cla acc: {:.4f}%'.format(
        cla_best_acc,
        rec_cla_best_acc,
        hybrid_cla_best_acc
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Deconf-net')
    parser.add_argument('--seed', default=1, type=int, help='seed for initialize training')
    parser.add_argument('--data_dir', help='directory to store datasets', default='/home/iip/datasets')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--output_dir', help='dir to store experiment artifacts', default='outputs')
    parser.add_argument('--output_sub_dir', help='sub dir to store experiment artifacts', default='wide_resnet_euclidean-hybrid')
    parser.add_argument('--feature_extractor', type=str, default='wide_resnet')
    parser.add_argument('--h', type=str, default='euclidean')  # inner, euclidean, cosine
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--pretrain_path', type=str, default='./snapshots/cifar10/wrn_e.pth')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--prefetch', type=int, default=4, help='number of dataloader workers')
    parser.add_argument('--gpu_idx', help='used gpu idx', type=int, default=0)
    args = parser.parse_args()
    
    main(args)
