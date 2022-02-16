import torch
from torchvision import transforms

from ..datasets import get_dataloader

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


if __name__ == '__main__':
    select_idxs = list(range(10))
    transform = transforms.ToTensor()
    root = '/home/iip/Robust_OOD_Detection/data/datasets'
    dataloader = get_dataloader('/home/iip/Robust_OOD_Detection/data/datasets', 'cifar10', select_idxs, 'train', transform, 8, False, 4)
    print(get_mean_and_std(dataloader))