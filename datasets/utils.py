import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchvision import transforms

from .named_dataset_with_meta import NamedDatasetWithMeta
from .named_or_dataset_with_meta import NamedOrDatasetWithMeta

from .transforms import get_shift_transform

CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

def get_dataset_info(ds_name, info_type):
    dataset_info = {
        'svhn': {
            'image_size': 224,
            'channel': 3,
            'classes': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'mean_and_std': [(0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)]
        },
        'cifar10': {
            'image_size': 32,
            'channel': 3,
            'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
            'mean_and_std': [(0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)]
        },
        'cifar100': {
            'image_size': 32,
            'channel': 3,
            'classes': CIFAR100_CLASSES,
            'mean_and_std': [(0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)]
        }
    }

    if ds_name not in dataset_info.keys():
        raise Exception('---> Dataset Info: {} not available'.format(ds_name))
    
    ds_info = dataset_info[ds_name]
    if info_type not in ds_info.keys():
        raise Exception('---> Dataset Info Type: {} not available'.format(info_type))
    return ds_info[info_type]


class Convert():
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


def get_ae_transforms(stage):
    if stage == 'train':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor()
        ])
    elif stage == 'test':
        return transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        raise Exception('---> Dataset Stage: {} invalid'.format(stage))


def get_transforms(name, stage):
    mean, std = get_dataset_info(name, 'mean_and_std')
    if stage == 'train':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif stage == 'test':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise Exception('---> Dataset Stage: {} invalid'.format(stage))


def get_ae_ood_transforms(ood, stage):
    
    if stage == 'train':
        ood_transforms = {
            'ti_300k': [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        }
    elif stage == 'test':
        ood_transforms = {
            'svhn': [transforms.ToTensor()],
            'places365_10k': [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()],
            'lsunc': [transforms.CenterCrop(32), transforms.ToTensor()],
            'lsunr': [transforms.ToTensor()],
            'tinc': [transforms.CenterCrop(32), transforms.ToTensor()],
            'tinr': [transforms.ToTensor()],
            'dtd': [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()],
            'isun': [transforms.ToTensor()],
            'cifar10': [transforms.ToTensor()],
            'cifar100': [transforms.ToTensor()]
        }
    else:
        raise Exception('---> Dataset Stage: {} invalid'.format(stage))
    
    return transforms.Compose(ood_transforms[ood])


def get_ood_transforms(id, ood, stage):
    mean, std = get_dataset_info(id, 'mean_and_std')
    
    if stage == 'train':
        ood_transforms = {
            'ti_300k': [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)]
        }
    elif stage == 'test':
        ood_transforms = {
            'svhn': [transforms.ToTensor(), transforms.Normalize(mean, std)],
            'places365_10k': [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize(mean, std)],
            'lsunc': [transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize(mean, std)],
            'lsunr': [transforms.ToTensor(), transforms.Normalize(mean, std)],
            'tinc': [transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize(mean, std)],
            'tinr': [transforms.ToTensor(), transforms.Normalize(mean, std)],
            'dtd': [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize(mean, std)],
            'isun': [transforms.ToTensor(), transforms.Normalize(mean, std)],
            'cifar10': [transforms.ToTensor(), transforms.Normalize(mean, std)],
            'cifar100': [transforms.ToTensor(), transforms.Normalize(mean, std)]
        }
    else:
        raise Exception('---> Dataset Stage: {} invalid'.format(stage))
    
    return transforms.Compose(ood_transforms[ood])


# get common dataset
def get_dataset(root, name, split, transform, target_transform=None):
    dataset = NamedDatasetWithMeta(
        root=root,
        name=name,
        split=split,
        transform=transform,
        target_transform=target_transform
    )
    
    return dataset


def get_hybrid_dataset(root, name, split, transform, target_transform=None):
    dataset = NamedOrDatasetWithMeta(
        root=root,
        name=name,
        split=split,
        transform=transform,
        target_transform=target_transform
    )
    
    return dataset


# get common dataloader
def get_dataloader(root, name, split, transform, batch_size, shuffle, num_workers):

    ds = get_dataset(
        root=root,
        name=name,
        split=split,
        transform=transform
    )

    return DataLoader(
        ds,
        batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def get_hybrid_dataloader(root, name, split, transform, batch_size, shuffle, num_workers):
    
    ds = get_hybrid_dataset(
        root=root,
        name=name,
        split=split,
        transform=transform
    )
    
    return DataLoader(
        ds,
        batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


# -------- Validation OOD datasets --------
def get_uniform_noise_dataset(num):
    dummpy_targets = torch.ones(num)
    uniform_noise_data = torch.from_numpy(
        np.random.uniform(size=(num, 3, 32, 32), low=-1.0, high=1.0).astype(np.float32))
    
    return TensorDataset(uniform_noise_data, dummpy_targets)

def get_uniform_noise_dataloader(num, batch_size, shuffle, num_workers):
    uniform_noise_dataset = get_uniform_noise_dataset(num)
    uniform_noise_dataset.name = 'uniform_noise'
    uniform_noise_dataset.labeled = True
    
    return DataLoader(
        uniform_noise_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )
    

class AvgOfPair(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.shuffle_indices = np.arange(len(dataset))
        np.random.shuffle(self.shuffle_indices)
        self.labeled = False
        
    def __getitem__(self, i):
        random_idx = np.random.choice(len(self.dataset))
        while random_idx == i:
            random_idx = np.random.choice(len(self.dataset))
            
        return self.dataset[i][0] / 2. + self.dataset[random_idx][0] / 2.
    
    def __len__(self):
        return len(self.dataset)


class GeoMeanOfPair(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.mean, self.std = get_dataset_info(dataset.name, 'mean_and_std')
        self.shuffle_indices = np.arange(len(dataset))
        np.random.shuffle(self.shuffle_indices)
        self.labeled = False
        
    def __getitem__(self, i):
        random_idx = np.random.choice(len(self.dataset))
        while random_idx == i:
            random_idx = np.random.choice(len(self.dataset))
        
        return transforms.Normalize(self.mean, self.std)(torch.sqrt(self.dataset[i][0] * self.dataset[random_idx][0]))
    
    def __len__(self):
        return len(self.dataset)
    
