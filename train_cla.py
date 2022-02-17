import argparse
import time
import numpy as np
from functools import partial
from pathlib import Path
import random
import copy

import torch
import torch.backends.cudnn as cudnn

from datasets import get_dataset_info, get_transforms, get_dataloader
from models import get_classifier
from trainers import get_classifier_trainer
from evaluation import Evaluator
from utils import  setup_logger


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
    exp_path = Path(args.output_dir) / args.output_sub_dir
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
        get_dataloader,
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
    print('>>> Classifier: {}'.format(args.arch))
    classifier = get_classifier(args.arch, num_classes)

    # ------------------------------------ Init Trainer ------------------------------------
    print('>>> Optimizer: SGD  | Scheduler: LambdaLR')
    print('>>> Lr: {:.5f} | Weight_decay: {:.5f} | Momentum: {:.2f}').format(args.lr, args.weight_decay, args.momentum)
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
    
    trainer = get_classifier_trainer(classifier, train_loader, optimizer, scheduler)
    
    # move classifier to gpu device
    best_cla_acc = 0.0
    start_epoch = 1

    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        classifier.cuda()
    cudnn.benchmark = True

    # ------------------------------------ Start training ------------------------------------
    evaluator = Evaluator(classifier)
    begin_time = time.time()
    
    last_state = {}
    
    for epoch in range(start_epoch, args.epochs+1):
    
        trainer.train_epoch()
        val_metrics = evaluator.eval_classification(test_loader)
        
        cla_best = val_metrics['cla_acc'] > best_cla_acc
        best_cla_acc = max(val_metrics['cla_acc'], best_cla_acc)
        
        if epoch == args.epochs:
            last_state = {
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': classifier.state_dict(),
                'cla_acc': val_metrics['cla_acc']
            }
        
        if cla_best:
            cla_best_state = {
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': copy.deepcopy(classifier.state_dict()),
                'cla_acc': best_cla_acc
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
    last_path = exp_path / 'last.pth'
    torch.save(last_state, str(last_path))
    print('---> Best classify acc: {:.4f}%'.format(best_cla_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Classifier')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initialize training')
    parser.add_argument('--arch', type=str, default='wide_resnet')
    parser.add_argument('--output_dir', help='dir to store experiment artifacts', default='outputs')
    parser.add_argument('--output_sub_dir', help='sub dir to store experiment artifacts', default='wide_resnet')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--data_dir', help='directory to store datasets', default='data/datasets')
    parser.add_argument('--dataset', type=str, default='cifar10')
    # parser.add_argument('--oods', nargs='+', default=['svhn', 'cifar100', 'tinc', 'tinr', 'lsunc', 'lsunr', 'dtd', 'places365', 'isun'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--prefetch', type=int, default=4, help='number of dataloader workers')
    parser.add_argument('--gpu_idx', help='used gpu idx', type=int, default=0)
    args = parser.parse_args()
    main(args)