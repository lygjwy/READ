from pathlib import Path
from functools import partial
import argparse

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn

from datasets import get_ae_transforms
from datasets.utils import get_dataloader
from models import get_ae


def main(args):
    dataset_root = Path(args.data_dir)
    dataset = args.dataset
    ae_transform = get_ae_transforms('test')
    
    get_dataloader_default = partial(
        get_dataloader,
        root=args.data_dir,
        split=args.split,
        transform=ae_transform,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.prefetch
    )
    
    # load dataset
    data_loader = get_dataloader_default(name=dataset)
    samples = data_loader.dataset.samples
    
    ae = get_ae(args.arch)
    ae_path = Path(args.ae_path)
    
    if ae_path.exists():
        ae_params = torch.load(str(ae_path))
        rec_err = ae_params['rec_err']
        ae.load_state_dict(ae_params['state_dict'])
        print('>>> load ae from {} (rec err {:.4f})'.format(str(ae_path), rec_err))
    else:
        print('---> invalid ae path: {}'.format(str(ae_path)))
    
    gpu_idx = int(args.gpu_idx)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx)
        ae.cuda()
    cudnn.benchmark = True
    
    total_loss = 0.0
    ae.eval()
    
    rec_data_list, target_list = [], []
    
    with torch.no_grad():
        for sample in data_loader:
            data = sample['data'].cuda()
            target = sample['label']
            target_list.extend(target.tolist())

            rec_data = ae(data)
            
            rec_loss = F.mse_loss(rec_data, data, reduction='sum')
            total_loss += rec_loss.item()
            rec_data_list.append(rec_data)
            
        rec_data = torch.cat(rec_data_list, dim=0)
    
    for i, target in enumerate(target_list):
        sample = samples[i]
        
        image_path = sample[0]
        assert sample[1] == target
        
        image_path_prefix, image_path_suffix = image_path.strip().split('.')
        rec_image_path = image_path_prefix + 'r.' + image_path_suffix
        save_image(rec_data[i].cpu(), str(dataset_root / rec_image_path))

# end for loop

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoencoder reconstruction')
    parser.add_argument('--data_dir', type=str, default='/home/iip/datasets')
    parser.add_argument('--dataset', type=str, default='cifar10_hybrid')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument('--arch', type=str, default='res_ae')
    parser.add_argument('--ae_path', type=str, default='./snapshots/cifar10/rec.pth')
    parser.add_argument('--gpu_idx', type=int, default=0)
    
    args = parser.parse_args()

    main(args)