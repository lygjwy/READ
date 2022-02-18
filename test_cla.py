from pathlib import Path
import argparse
from functools import partial

import torch
import torch.backends.cudnn as cudnn

from datasets import get_dataset_info, get_transforms, get_dataloader
from models import get_classifier
from evaluation import Evaluator


def main(args):
    
    # -------------------- data loader -------------------- #
    transform = get_transforms(args.dataset, 'test')
    print('>>> Dataset: {}'.format(args.dataset))
    get_dataloader_default = partial(
        get_dataloader,
        root=args.data_dir,
        name=args.dataset,
        transform=transform,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.prefetch
    )
    
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
        # classifier.load_state_dict(torch.load(str(classifier_path)))
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
    
    test_train_cla_acc = evaluator.eval_classification(test_loader_train)['cla_acc']
    test_test_cla_acc = evaluator.eval_classification(test_loader_test)['cla_acc']
    
    print('[train set cla acc: {:.4f}% | test set cla acc: {:.4f}%]'.format(test_train_cla_acc, test_test_cla_acc))

    # save in standard format
    # standard_state = {
    #     'epoch': 100,
    #     'arch': 'wide_resnet',
    #     'state_dict': classifier.state_dict(),
    #     'cla_acc': test_test_cla_acc
    # }
    
    # standard_path =  '/home/iip/AEA/snapshots/e-p.pth'
    # torch.save(standard_state, str(standard_path))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='id dataset classification evaluation')
    parser.add_argument('--data_dir', type=str, default='./datasets')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument('--classifier', type=str, default='wide_resnet')
    parser.add_argument('--classifier_path', type=str, default='./snapshots/p.pth')
    parser.add_argument('--gpu_idx', type=int, default=0)
    
    args = parser.parse_args()
    
    main(args)
