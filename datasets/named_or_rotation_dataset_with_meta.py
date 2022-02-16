import os
import numpy as np
from PIL import Image

from torchvision import transforms

from .named_or_dataset_with_meta import NamedOrDatasetWithMeta


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


class NamedOrRotationDatasetWithMeta(NamedOrDatasetWithMeta):
    """return rotation version of image & corresponding reconstruction image
    """
    def __init__(self, root, name, split, random, transform, target_transform):
        root = os.path.expanduser(root)
        super(NamedOrRotationDatasetWithMeta, self).__init__(root, name, split, transform, target_transform=target_transform)
        
        self.center_crop = transforms.CenterCrop(32)
        self.random = random
        self.random_horizontal_flip = transforms.RandomHorizontalFlip()
        self.random_crop = transforms.RandomCrop(32, padding=4)
        
        self.name = name + '-rot'
        
    
    def __getitem__(self, index):
        ori_sample = self.original_samples[index]
        
        ori_img_path = ori_sample[0]
        ori_img_path_prefix, ori_img_path_suffix = ori_img_path.strip().split('.')
        rec_img_path = ori_img_path_prefix + 'r.' + ori_img_path_suffix

        ori_img = default_loader(ori_img_path)
        rec_img = default_loader(rec_img_path)
        
        ori_img = self.center_crop(ori_img)
        rec_img = self.center_crop(rec_img)
        
        if self.random:
            ori_img = self.random_horizontal_flip(ori_img)
            ori_img = self.random_crop(ori_img)
            rec_img = self.random_horizontal_flip(rec_img)
            rec_img = self.random_crop(rec_img)
            
        # rotation process
        ori_img = np.array(ori_img)
        ori_img_0 = np.copy(ori_img)
        ori_img_1 = np.rot90(ori_img.copy(), k=1).copy()
        ori_img_2 = np.rot90(ori_img.copy(), k=2).copy()
        ori_img_3 = np.rot90(ori_img.copy(), k=3).copy()
        
        # data preprocess
        rec_img = np.array(rec_img)
        rec_img_0 = np.copy(rec_img)
        rec_img_1 = np.rot90(rec_img.copy(), k=1).copy()
        rec_img_2 = np.rot90(rec_img.copy(), k=2).copy()
        rec_img_3 = np.rot90(rec_img.copy(), k=3).copy()
        
        if self.transform is not None:
            ori_img_0, ori_img_1, ori_img_2, ori_img_3 = self.transform(ori_img_0), self.transform(ori_img_1), self.transform(ori_img_2), self.transform(ori_img_3)
            rec_img_0, rec_img_1, rec_img_2, rec_img_3 = self.transform(rec_img_0), self.transform(rec_img_1), self.transform(rec_img_2), self.transform(rec_img_3)
            
        if self.labeled:
            target = ori_sample[1]
            if self.target_transform is not None:
                target = self.target_transform(target)
            return ori_img_0, ori_img_1, ori_img_2, ori_img_3, rec_img_0, rec_img_1, rec_img_2, rec_img_3, target
        else:
            return  ori_img_0, ori_img_1, ori_img_2, ori_img_3, rec_img_0, rec_img_1, rec_img_2, rec_img_3
        
        
    def __len__(self):
        return len(self.original_samples)    
    
        
        
    
    