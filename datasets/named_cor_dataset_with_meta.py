import ast
import os
import sys
from PIL import Image
from pathlib import Path

from torchvision.datasets import VisionDataset

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')

def default_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    

def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)


def is_original_img_file(file_path):
    file_prefix = file_path.strip().split('.')[0].lower()
    if file_prefix.endswith('r') or file_prefix.endswith('c'):
        return False
    
    if os.path.exists(file_path) and has_file_allowed_extension(file_path, IMG_EXTENSIONS):
        return True
    return False


class NamedCorDatasetWithMeta(VisionDataset):
    """return original image and corresponding reconstruciton image 

    """
    def __init__(self, root, name, split, transform, target_transform=None):
        root = os.path.expanduser(root)
        super(NamedCorDatasetWithMeta, self).__init__(root, transform=transform, target_transform=target_transform)
        
        self.root = Path(root)
        dataset_path = self.root / name
        
        if split == 'train':
            self.entry_path = dataset_path / 'train.txt'
        elif split == 'test':
            self.entry_path = dataset_path / 'test.txt'
        else:
            raise RuntimeError('<--- invalid split: {}'.format(split))
        
        if not self.entry_path.is_file():
            raise RuntimeError('<--- entry file: {} not exist'.format(str(self.entry_path)))
        
        self.classes, self.class_to_idx = self._find_classes(dataset_path)
        
        self.original_samples = self._parse_entry_file()
        self.name = name
        
    def _find_classes(self, dataset_path):
        classes_path = dataset_path / 'classes.txt'
        
        if classes_path.is_file():
            self.labeled = True
            with open(classes_path, 'r') as f:
                classes = sorted(ast.literal_eval(f.readline()))
            class_to_idx = {cla: idx for idx, cla in enumerate(classes)}
        else:
            self.labeled = False
            print('---> loading unlabeled dataset from {}'.format(dataset_path))
            classes = None
            class_to_idx = None
        return classes, class_to_idx
    
    def _parse_entry_file(self):
        
        with open(self.entry_path, 'r') as ef:
            entries = ef.readlines()
        
        tokens_list = [entry.strip().split(' ') for entry in entries]
        
        original_samples = []
        platform = sys.platform
        
        if self.labeled:
            if platform == 'linux':
                original_samples = [(str(self.root / tokens[0].replace('\\', '/')), tokens[1]) for tokens in tokens_list if is_original_img_file(tokens[0])]
                
            if platform == 'win32':
                original_samples = [(str(self.root / tokens[0].replace('/', '\\')), tokens[1]) for tokens in tokens_list if is_original_img_file(tokens[0])]
        
        else:
            if platform == 'linux':
                original_samples = [(str(self.root / tokens[0].replace('\\', '/')),) for tokens in tokens_list if is_original_img_file(tokens[0])]
            
            if platform == 'win32':
                original_samples = [(str(self.root / tokens[0].replace('/', '\\')),) for tokens in tokens_list if is_original_img_file(tokens[0])]
        
        return original_samples
    
    def __getitem__(self, index):
        original_sample = self.original_samples[index]
        
        ori_image_path = original_sample[0]
        ori_image_path_prefix, ori_image_path_suffix = ori_image_path.strip().split('.')
        rec_image_path = ori_image_path_prefix + 'r.' + ori_image_path_suffix
        cor_image_path = ori_image_path_prefix + 'c.' + ori_image_path_suffix
        
        cor_image = default_loader(cor_image_path)
        ori_image = default_loader(ori_image_path)
        rec_image = default_loader(rec_image_path)
        if self.transform is not None:
            cor_image = self.transform(cor_image)
            ori_image = self.transform(ori_image)
            rec_image = self.transform(rec_image)
        
        if self.labeled:
            target = original_sample[1]
            if self.target_transform is not None:
                target = self.target_transform(target)
            return cor_image, ori_image, rec_image, target
        else:
            return cor_image, ori_image, rec_image
        
    def __len__(self):
        return len(self.original_samples)
