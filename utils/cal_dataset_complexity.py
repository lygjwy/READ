import argparse
import os
from functools import partial

from torchvision import transforms

from datasets import get_dataloader


transforms_dic = {
    'cifar10': [transforms.ToTensor()],
    'cifar100': [transforms.ToTensor()],
    'svhn': [transforms.ToTensor()],
    'tinc': [transforms.CenterCrop(32), transforms.ToTensor()],
    'tinr': [transforms.ToTensor()],
    'lsunc': [transforms.CenterCrop(32), transforms.ToTensor()],
    'lsunr': [transforms.ToTensor()],
    'dtd': [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()],
    'places365_10k': [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()],
    'isun': [transforms.ToTensor()]
}


def save_entry(image_paths, image_sizes, file_path):
    lines = [' '.join([image_path, str(image_size)]) for image_path, image_size in zip(image_paths, image_sizes)]
    content = '\n'.join(lines)
    with open(file_path, 'w') as f:
        f.write(content)


def cal_complexity(data_loader, split):
    idx = 0
    img_paths, img_sizes = [], []
    samples = data_loader.dataset.samples
    
    for sample in data_loader:
        
        
        if data_loader.dataset.labeled:
            data, data_target = sample
            img_path, target = samples[idx]
            
            assert int(target) == int(data_target)
        else:
            data = sample
            img_path = samples[idx][0]

        img = data.squeeze()
        to_pil = transforms.ToPILImage()
        pil_img = to_pil(img)
        pil_img.save('./tmp.png', 'PNG')
        img_size = os.path.getsize('./tmp.png')
        # os.remove('./tmp.png')
        img_paths.append(img_path)
        img_sizes.append(img_size)
        idx += 1
    
    file_path = '/home/iip/datasets/' + data_loader.dataset.name + '/' + split + '_complexity.txt'
    print(file_path)
    save_entry(img_paths, img_sizes, file_path)


def main(args):
    
    get_dataloader_default = partial(
        get_dataloader,
        root=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.prefetch
    )
    
    for dataset in args.datasets:
        dataset_transforms = transforms.Compose(transforms_dic[dataset])
        dataset_loader = get_dataloader_default(name=dataset, transform=dataset_transforms)
        cal_complexity(dataset_loader, args.split)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate Image Complexity')
    parser.add_argument('--data_dir', type=str, default='/home/iip/datasets')
    parser.add_argument('--datasets', nargs='+', default=['tinc', 'tinr', 'lsunc', 'lsunr', 'dtd', 'places365_10k', 'isun'])
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--prefetch', type=int, default=4)
    
    args = parser.parse_args()
    
    main(args)