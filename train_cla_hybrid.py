import numpy as np
import argparse
import time
import copy
from functools import partial
from pathlib import Path
import random

import torch
import torch.backends.cudnn as cudnn


from datasets import get_dataset_info, get_transforms, get_hybrid_dataloader
from models import get_classifier
from trainers import get_classifier_hybrid_trainer
from evaluation import Evaluator
from utils import setup_logger


# scheduler
def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

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
    train_transform = get_transforms(args.dataset, stage='train')
    val_transform = get_transforms(args.dataset, stage='test')  # using train set's mean&std

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
    num_classes = len(get_dataset_info(args.dataset, 'classes'))
    print('>>> Classifier: {}'.format(args.arch))
    classifier = get_classifier(args.arch, num_classes)
    
    if args.pretrained:
        # load pretrained model
        pretrain_path = Path(args.pretrain_path)
        if pretrain_path.exists():
            cla_params = torch.load(str(pretrain_path))
            cla_acc = cla_params['cla_acc']
            classifier.load_state_dict(cla_params['state_dict'])
            print('>>> load pretrained classifier from {} (classification acc {:.4f}%)'.format(str(pretrain_path), cla_acc))
        else:
            raise RuntimeError('<--- invalid pretrained classifier path: {}'.format(str(pretrain_path)))
    
    # ------------------------------------ Init Trainer ------------------------------------
    print('>>> Optimizer: SGD  | Scheduler: LambdaLR')
    print('>>> Lr: {:.5f} | Weight_decay: {:.5f} | Momentum: {:.2f}'.format(args.lr, args.weight_decay, args.momentum))
    optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader),
            1,
            1e-6 / args.lr
        )
    )

    trainer = get_classifier_hybrid_trainer(classifier, train_loader, optimizer, scheduler)
    
    # move classifier to gpu device
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        classifier.cuda()
    cudnn.benchmark = True

    # ------------------------------------ Start training ------------------------------------
    evaluator = Evaluator(classifier)
    begin_time = time.time()
    
    start_epoch = 1
    cla_best_acc, rec_cla_best_acc, hybrid_cla_best_acc = 0.0, 0.0, 0.0
    cla_best_state, rec_cla_best_state, hybrid_cla_best_state, last_state = {}, {}, {}, {}
    
    for epoch in range(start_epoch, args.epochs+1):
        trainer.train_epoch()
        val_metrics = evaluator.eval_hybrid_classification(test_loader)
        
        cla_best = val_metrics['cla_acc'] > cla_best_acc
        cla_best_acc = max(val_metrics['cla_acc'], cla_best_acc)
        
        rec_cla_best = val_metrics['rec_cla_acc'] > rec_cla_best_acc
        rec_cla_best_acc = max(val_metrics['rec_cla_acc'], rec_cla_best_acc)
        
        hybrid_cla_best = val_metrics['hybrid_cla_acc'] > hybrid_cla_best_acc
        hybrid_cla_best_acc = max(val_metrics['hybrid_cla_acc'], hybrid_cla_best_acc)
        
        if epoch == args.epochs:
            last_state = {
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': classifier.state_dict(),
                'cla_acc': val_metrics['cla_acc'],
                'rec_cla_acc': val_metrics['rec_cla_acc'],
                'hybrid_cla_acc': val_metrics['hybrid_cla_acc']
            }

        if cla_best:
            cla_best_state = {
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': copy.deepcopy(classifier.state_dict()),
                'cla_acc': val_metrics['cla_acc'],
                'rec_cla_acc': val_metrics['rec_cla_acc'],
                'hybrid_cla_acc': val_metrics['hybrid_cla_acc']
            }
        
        if rec_cla_best:
            rec_cla_best_state = {
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': copy.deepcopy(classifier.state_dict()),
                'cla_acc': val_metrics['cla_acc'],
                'rec_cla_acc': val_metrics['rec_cla_acc'],
                'hybrid_cla_acc': val_metrics['hybrid_cla_acc']
            }
            
        if hybrid_cla_best:
            hybrid_cla_best_state = {
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': copy.deepcopy(classifier.state_dict()),
                'cla_acc': val_metrics['cla_acc'],
                'rec_cla_acc': val_metrics['rec_cla_acc'],
                'hybrid_cla_acc': val_metrics['hybrid_cla_acc']
            }

        print(
            "---> Epoch {:3d} | Time {:5d}s".format(
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
    print('---> Best cla acc: {:.4f}% | rec cla acc: {:.4f}% | cla acc: {:.4f}%'.format(
        cla_best_acc,
        rec_cla_best_acc,
        hybrid_cla_best_acc
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Classifier with hybrid images')
    parser.add_argument('--seed', default=1, type=int, help='seed for initialize training')
    parser.add_argument('--data_dir', help='directory to store datasets', default='/home/iip/datasets')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--pretrain_path', type=str, default='snapshots')
    parser.add_argument('--output_dir', help='dir to store experiment artifacts', default='outputs')
    parser.add_argument('--output_sub_dir', help='sub dir to store experiment artifacts', default='tmp')
    parser.add_argument('--arch', type=str, default='wide_resnet')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--prefetch', type=int, default=4, help='number of dataloader workers')
    parser.add_argument('--gpu_idx', help='used gpu idx', type=int, default=0)
    args = parser.parse_args()
    
    main(args)