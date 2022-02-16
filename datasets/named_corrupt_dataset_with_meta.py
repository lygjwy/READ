from PIL import Image
import numpy as np

from torchvision import transforms
from torchvision.datasets.folder import default_loader
from .named_dataset_with_meta import NamedDatasetWithMeta


class GaussianNoise(object):
    """Apply gaussian noise on PIL input

    Args:
        x (PIL Image): Image to be processed.
        
    Returns:
        x (PIL Image): Processed image.
    """
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
        
    def __call__(self, x):
        x = np.array(x) / 255.
        # must convert to uint8
        # PIL_image = Image.fromarray(np.uint8(numpy_image)).convert('RGB')
        np_image = np.clip(x + np.random.normal(size=x.shape, loc=self.mean, scale=self.std), 0, 1)
        return Image.fromarray(np.uint8(np_image * 255))

    def __repr__(self):
        return self.__class__.__name__ + 'mean={}, std={}'.format(self.mean, self.std)


class NamedCorruptDatasetWithMeta(NamedDatasetWithMeta):
    """load dataset with specified corruption
    """
    def __init__(self, root, name, corruption, severity, split, random, transform, target_transform=None):
        super(NamedCorruptDatasetWithMeta, self).__init__(root=root, name=name, split=split, transform=transform, target_transform=target_transform)
        
        self.random = random
        self.center_crop = transforms.CenterCrop(32)
        self.random_horizontal_flip = transforms.RandomHorizontalFlip()
        self.random_crop = transforms.RandomCrop(32, padding=4)
        self.corruption = corruption
        self.severity = severity
        
        # get corruption & severity transform
        self.corrupt = GaussianNoise(mean=0, std=self.severity)
        self.name = name + '-crpt'
        
    def __getitem__(self, index):
        sample = self.samples[index]
        
        img_path = sample[0]
        img = default_loader(img_path)
        img = self.center_crop(img)
        #  random preprocess
        if self.random:
            img = self.random_horizontal_flip(img)
            img = self.random_crop(img)
        
        #  corruption preprocess
        cor_img = self.corrupt(img)
        
        # data preprocess
        if self.transform is not None:
            img, cor_img = self.transform(img), self.transform(cor_img)
        
        if self.labeled:
            target = sample[1]
            if self.target_transform is not None:
                target = self.target_transform(target)
            return cor_img, img, target
        else:
            return cor_img, img
        
    
    def __len__(self):
        return len(self.samples)
        
        
        
    