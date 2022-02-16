import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .named_dataset_with_meta import NamedDatasetWithMeta
from .named_corrupt_dataset_with_meta import NamedCorruptDatasetWithMeta
from .named_or_dataset_with_meta import NamedOrDatasetWithMeta
from .named_rotation_dataset_with_meta import NamedRotationDatasetWithMeta
from .named_or_rotation_dataset_with_meta import NamedOrRotationDatasetWithMeta


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

def get_normal_transform(name):
    mean, std = get_dataset_info(name, 'mean_and_std')
    
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    

def get_ae_normal_transform():
    
    return transforms.Compose([
        transforms.ToTensor()
    ])


def get_ae_transform(stage):    
    if stage == 'train':
        return transforms.Compose([
            # Convert(color_mode),
            transforms.CenterCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor()
        ])
    elif stage == 'test':
        return transforms.Compose([
            # Convert(color_mode),
            transforms.CenterCrop(32),
            transforms.ToTensor()
        ])
    else:
        raise Exception('---> Dataset Stage: {} invalid'.format(stage))


def get_transforms(name, stage):
    mean, std = get_dataset_info(name, 'mean_and_std')
    if stage == 'train':
        return transforms.Compose([
            transforms.CenterCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif stage == 'test':
        return transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise Exception('---> Dataset Stage: {} invalid'.format(stage))


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


def get_corrupt_dataset(root, name, corrupt, severity, split, random, transform, target_transform=None):
    dataset = NamedCorruptDatasetWithMeta(
        root=root,
        name=name,
        corruption=corrupt,
        severity=severity,
        split=split,
        random=random,
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


def get_auxiliary_dataset(root, name, split, random, transform, target_transform=None):
    dataset = NamedRotationDatasetWithMeta(
        root=root,
        name=name,
        split=split,
        random=random,
        transform=transform,
        target_transform=target_transform
    )
    
    return dataset


def get_hybrid_auxiliary_dataset(root, name, split, random, transform, target_transform=None):
    dataset = NamedOrRotationDatasetWithMeta(
        root=root,
        name=name,
        split=split,
        random=random,
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
    
    
def get_corrupt_dataloader(root, name, corrupt, severity, split, random, transform, batch_size, shuffle, num_workers):
    
    ds = get_corrupt_dataset(
        root=root,
        name=name,
        corrupt=corrupt,
        severity=severity,
        split=split,
        random=random,
        transform=transform 
    )
    
    return DataLoader(
        ds,
        batch_size=batch_size,
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
    

def get_auxiliary_dataloader(root, name, split, random, transform, batch_size, shuffle, num_workers):
    
    ds = get_auxiliary_dataset(
        root=root,
        name=name,
        split=split,
        random=random,
        transform=transform
    )
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    

def get_hybrid_auxiliary_dataloader(root, name, split, random, transform, batch_size, shuffle, num_workers):
    
    ds = get_hybrid_auxiliary_dataset(
        root=root,
        name=name,
        split=split,
        random=random,
        transform=transform
    )
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )