import ast
import os
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
    if file_prefix.endswith('r'):
        return False
    
    if os.path.exists(file_path) and has_file_allowed_extension(file_path, IMG_EXTENSIONS):
        return True
    return False


class NamedOrDatasetWithMeta(VisionDataset):
    """return original image and corresponding reconstruciton image 

    """
    def __init__(self, root, name, split, transform, target_transform=None):
        root = os.path.expanduser(root)
        self.name = name + '-or'
        super(NamedOrDatasetWithMeta, self).__init__(root, transform=transform, target_transform=target_transform)
        
        self.root = Path(root)
        dataset_path = self.root / self.name
        
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

        if self.labeled:
            for tokens in tokens_list:
                img_path = str(self.root / tokens[0].replace('\\', '/'))
                target = int(tokens[1])
                if is_original_img_file(img_path):
                    original_samples.append((img_path, target))
         
        else:
            for tokens in tokens_list:
                img_path = str(self.root / tokens[0].replace('\\', '/'))
                if is_original_img_file(img_path):
                    original_samples.append((img_path,))
        
        return original_samples
    
    def __getitem__(self, index):
        original_sample = self.original_samples[index]
        
        ori_image_path = original_sample[0]
        ori_image_path_prefix, ori_image_path_suffix = ori_image_path.strip().split('.')
        rec_image_path = ori_image_path_prefix + 'r.' + ori_image_path_suffix
        
        ori_image = default_loader(ori_image_path)
        rec_image = default_loader(rec_image_path)
        if self.transform is not None:
            ori_image = self.transform(ori_image)
            rec_image = self.transform(rec_image)
        
        if self.labeled:
            target = original_sample[1]
            if self.target_transform is not None:
                target = self.target_transform(target)
            return ori_image, rec_image, target
        else:
            return ori_image, rec_image
        
    def __len__(self):
        return len(self.original_samples)
                
        
        


        