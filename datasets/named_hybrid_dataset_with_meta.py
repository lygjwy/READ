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


def is_img_file(file_path):
    # check if the image path valid (file exists and has allowed extension)
    if os.path.exists(file_path) and has_file_allowed_extension(file_path, IMG_EXTENSIONS):
        return True
    return False


class NamedHybridDatasetWithMeta(VisionDataset):
    """return original image and corresponding reconstruciton image 

    """
    def __init__(self, root, name, split, transform, target_transform=None):
        root = os.path.expanduser(root)
        self.name = name + '_hybrid'
        super(NamedHybridDatasetWithMeta, self).__init__(root, transform=transform, target_transform=target_transform)
        
        self.root = Path(root)
        dataset_path = self.root / self.name
        
        if split == 'train':
            self.entry_path = dataset_path / 'train.txt'
            self.complexity_path = dataset_path / 'train_complexity.txt'
        elif split == 'test':
            self.entry_path = dataset_path / 'test.txt'
            self.complexity_path = dataset_path / 'test_complexity.txt'
        else:
            raise RuntimeError('<--- invalid split: {}'.format(split))
        
        if not self.entry_path.is_file():
            raise RuntimeError('<--- entry file: {} not exist'.format(str(self.entry_path)))
        
        if not self.complexity_path.is_file():
            raise RuntimeError('<--- Split complexity file: {} not exist.'.format(str(self.complexity_path)))
        
        self.classes, self.class_to_idx = self._find_classes(dataset_path)
        
        self.samples = self._parse_entry_file()
        self.complexities = self._parse_complexity_file()


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
        samples = []
        
        for tokens in tokens_list:
            img_path = str(self.root / tokens[0].replace('\\', '/'))
            
            if is_img_file(img_path):
                if self.labeled:
                    samples.append((img_path, int(tokens[1])))
                else:
                    samples.append((img_path,))
            else:
                raise RuntimeError('<--- invalid image path: {}'.format(img_path))
    
        return samples


    def _parse_complexity_file(self):
        
        with open(self.complexity_path, 'r') as cf:
            complexities = cf.readlines()
            
        tokens_list = [complexity.strip().split(' ') for complexity in complexities]
        complexities = []
        
        for tokens in tokens_list:
            img_path = str(self.root / tokens[0].replace('\\', '/'))
            
            if is_img_file(img_path):
                complexities.append((img_path, int(tokens[1])))
            else:
                raise RuntimeError('<--- invalid image path: {}'.format(img_path))
            
        return complexities


    def __getitem__(self, index):
        sample = self.samples[index]
        complexity = self.complexities[index]
        
        assert sample[0] == complexity[0]
        image_path = sample[0]
        image = default_loader(image_path)
        
        image_path_prefix, image_path_suffix = image_path.strip().split('.')
        rec_image_path = image_path_prefix + 'r.' + image_path_suffix
        rec_image = default_loader(rec_image_path)
        
        if self.transform is not None:
            image = self.transform(image)
            rec_image = self.transform(rec_image)
        
        if self.labeled:
            target = sample[1]
            if self.target_transform is not None:
                target = self.target_transform(target)
            return {'data': image, 'rec_data': rec_image, 'label': target, 'complexity': complexity[1] / 3072.}
        else:
            return {'data': image, 'rec_data': rec_image, 'complexity': complexity[1] / 3072.}


    def __len__(self):
        return len(self.samples)
