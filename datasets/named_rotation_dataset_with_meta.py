import os
import numpy as np
from PIL import Image

from torchvision import transforms

from .named_dataset_with_meta import NamedDatasetWithMeta


def default_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    

class NamedRotationDatasetWithMeta(NamedDatasetWithMeta):
    
    def __init__(self, root, name, split, random, transform, target_transform=None):
        root = os.path.expanduser(root) 
        super(NamedRotationDatasetWithMeta, self).__init__(root, name, split, transform=transform, target_transform=target_transform)
        self.center_crop = transforms.CenterCrop(32)
        self.random = random
        self.random_horizontal_flip = transforms.RandomHorizontalFlip()
        self.random_crop = transforms.RandomCrop(32, padding=4)
        
        self.name = name + '-rot'
        
    
    def __getitem__(self, index):
        sample = self.samples[index]
        
        img_path = sample[0]
        img = default_loader(img_path)
        
        img = self.center_crop(img)
        if self.random:
            #  dataset used for training model
            img = self.random_horizontal_flip(img)
            img = self.random_crop(img)
        
        # rotation process
        img = np.array(img)
        img_0 = np.copy(img)
        img_1 = np.rot90(img.copy(), k=1).copy()
        img_2 = np.rot90(img.copy(), k=2).copy()
        img_3 = np.rot90(img.copy(), k=3).copy()
        
        # data preprocess
        if self.transform is not None:
            img_0, img_1, img_2, img_3 = self.transform(img_0), self.transform(img_1), self.transform(img_2), self.transform(img_3)
            
        if self.labeled:
            target = sample[1]
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img_0, img_1, img_2, img_3, target
        else:
            return img_0, img_1, img_2, img_3
        
        
    def __len__(self):
        return len(self.samples)
            
        
        