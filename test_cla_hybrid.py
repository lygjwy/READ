from pathlib import Path
import argparse
from functools import partial

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from datasets import get_ae_transforms, get_dataset_info, get_dataloader
from models import get_ae, get_classifier


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            # t.mul_(s).add_(m)
            t.sub_(m).div_(s)
        return tensor


def eval_hybrid_classification(ae, classifier, data_loader, normalize):
    ae.eval()
    classifier.eval()
    
    total, ori_correct, rec_correct = 0, 0, 0
    
    for sample in data_loader:
        data = sample['data'].cuda()
        target = sample['label'].cuda()
        
        with torch.no_grad():
            rec_data = ae(data)
        
        data = torch.stack([normalize(img) for img in data], dim=0)
        rec_data = torch.stack([normalize(img) for img in rec_data], dim=0)
        
        with torch.no_grad():
            ori_logit = classifier(data)
            rec_logit = classifier(rec_data)
            total += target.size(0)
            _, ori_pred = ori_logit.max(dim=1)
            _, rec_pred = rec_logit.max(dim=1)
            ori_correct += ori_pred.eq(target).sum().item()
            rec_correct += rec_pred.eq(target).sum().item()
        
    metrics = {
        'cla_acc': 100. * ori_correct / total,
        'rec_cla_acc': 100. * rec_correct / total
    }
    
    return metrics


def main(args):
    
    # -------------------- data loader -------------------- #
    ae_transform = get_ae_transforms('test')
    
    means, stds = get_dataset_info(args.dataset, 'mean_and_std')
    normalize = Normalize(means, stds)
    print('>>> Reconstruction Dataset: {}'.format(args.dataset))
    get_dataloader_default = partial(
        get_dataloader,
        root=args.data_dir,
        name=args.dataset,
        transform=ae_transform,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.prefetch
    )
    
    test_loader_train = get_dataloader_default(split='train')
    test_loader_test = get_dataloader_default(split='test')
    
    # -------------------- ae & classifier -------------------- #
    ae = get_ae(args.ae)
    num_classes = len(get_dataset_info(args.dataset, 'classes'))
    classifier = get_classifier(args.classifier, num_classes)
    ae_path = Path(args.ae_path)
    classifier_path = Path(args.classifier_path)
    
    if ae_path.exists():
        ae_params = torch.load(str(ae_path))
        rec_err = ae_params['rec_err']
        ae.load_state_dict(ae_params['state_dict'])
        print('>>> load ae from {} (rec err {})'.format(str(ae_path), rec_err))
    else:
        raise RuntimeError('<--- invalid ae path: {}'.format(str(ae_path)))
    
    if classifier_path.exists():
        cla_params = torch.load(str(classifier_path))
        cla_acc = cla_params['cla_acc']
        classifier.load_state_dict(cla_params['state_dict'])
        print('>>> load classifier from {} (classification acc {:.4f})'.format(classifier_path, cla_acc))
    else:
        raise RuntimeError('<--- invalid classifier path: {}'.format(str(classifier_path)))
    
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        ae.cuda()
        classifier.cuda()
    cudnn.benchmark = True
    
    ae.eval()
    classifier.eval()
    
    # -------------------- inference -------------------- #
    test_train_rec_cla_acc = eval_hybrid_classification(ae, classifier, test_loader_train, normalize)['rec_cla_acc']
    test_test_rec_cla_acc = eval_hybrid_classification(ae, classifier, test_loader_test, normalize)['rec_cla_acc']
    
    print('[train set rec cla acc: {:.4f}% | test set rec cla acc: {:.4f}%]'.format(test_train_rec_cla_acc, test_test_rec_cla_acc))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reconstruction image classify')
    parser.add_argument('--data_dir', type=str, default='/home/iip/datasets')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument('--ae', type=str, default='res_ae')
    parser.add_argument('--ae_path', type=str, default='./snapshots/cifar10/rec.pth')
    parser.add_argument('--classifier', type=str, default='wide_resnet')
    parser.add_argument('--classifier_path', type=str, default='./snapshots/cifar10/wrn.pth')
    parser.add_argument('--gpu_idx', type=int, default=0)
    
    args = parser.parse_args()
    
    main(args)