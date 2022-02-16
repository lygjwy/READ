# test dataset classification acc, including corruption dataset
from pathlib import Path
import argparse
from functools import partial

import torch
import torch.backends.cudnn as cudnn

from datasets import get_dataset_info, get_normal_transform, get_transforms, get_dataloader, get_corrupt_dataloader
from models import get_classifier
from evaluation import Evaluator


def main(args):
    
    # -------------------- data loader -------------------- #
    normal_transform = get_normal_transform(args.dataset)
    transform = get_transforms(args.dataset, 'test')
    print('>>> Dataset: {} with {} data mode'.format(args.dataset, args.data_mode))
    if args.data_mode == 'original':
        get_dataloader_default = partial(
            get_dataloader,
            root=args.data_dir,
            name=args.dataset,
            transform=transform,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.prefetch
        )
    elif args.data_mode == 'corrupt':
        get_dataloader_default = partial(
            get_corrupt_dataloader,
            root=args.data_dir,
            name=args.dataset,
            corrupt=args.corrupt,
            severity=args.severity,
            random=False,
            transform=normal_transform,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.prefetch
        )
    else:
        raise RuntimeError('<--- invalid data mode: {}'.format(args.data_mode))
    
    test_loader_train = get_dataloader_default(split='train')
    test_loader_test = get_dataloader_default(split='test')
    
    # -------------------- classifier -------------------- #
    num_classes = len(get_dataset_info(args.dataset, 'classes'))
    classifier = get_classifier(args.classifier, num_classes)
    classifier_path = Path(args.classifier_path)
    
    if classifier_path.exists():
        cla_params = torch.load(str(classifier_path))
        cla_acc = cla_params['cla_acc']
        classifier.load_state_dict(cla_params['state_dict'])
        print('>>> load classifier from {} (classifiication acc {:.4f}%)'.format(str(classifier_path), cla_acc))
    else:
        raise RuntimeError('<--- invlaid classifier path: {}'.format(str(classifier_path)))
    
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        classifier.cuda()
    cudnn.benchmark = True
    
    classifier.eval()

    # -------------------- inference -------------------- #
    evaluator = Evaluator(classifier)
    
    if args.data_mode == 'original':
        test_train_cla_acc = evaluator.eval_classification(test_loader_train)['cla_acc']
        test_test_cla_acc = evaluator.eval_classification(test_loader_test)['cla_acc']
    
        print('[train set cla acc: {:.4f}% | test set cla acc: {:.4f}%]'.format(test_train_cla_acc, test_test_cla_acc))
    elif args.data_mode == 'corrupt':
        test_train_metrics = evaluator.eval_co_classification(test_loader_train)
        test_train_cor_cla_acc = test_train_metrics['cor_cla_acc']
        test_train_ori_cla_acc = test_train_metrics['ori_cla_acc']
        
        test_test_metrics = evaluator.eval_co_classification(test_loader_test)
        test_test_cor_cla_acc = test_test_metrics['cor_cla_acc']
        test_test_ori_cla_acc = test_test_metrics['ori_cla_acc']
        
        print('[train set cor cla acc: {:.4f}% | train set ori cla acc: {:.4f}%]'.format(test_train_cor_cla_acc, test_train_ori_cla_acc))
        print('[test set cor cla acc: {:.4f}% | test set ori cla acc: {:.4f}%]'.format(test_test_cor_cla_acc, test_test_ori_cla_acc))
    else:
        raise RuntimeError('<--- invalid data mode: {}'.format(args.data_mode))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='original & corruption dataset evaluation')
    parser.add_argument('--data_dir', type=str, default='./datasets')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--data_mode', type=str, default='original')
    parser.add_argument('--corrupt', type=str, default='gaussian_noise')
    parser.add_argument('--severity', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument('--classifier', type=str, default='resnet18')
    parser.add_argument('--classifier_path', type=str, default='./outputs/resnet18/cla_best.pth')
    parser.add_argument('--gpu_idx', type=int, default=0)
    
    args = parser.parse_args()
    
    main(args)
        
        
         
