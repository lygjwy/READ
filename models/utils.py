from .resnet import get_resnet18
from .res_rot import get_resnet_rot
from .res_ae import get_res_ae
from .wide_resnet import get_wrn


def get_ae(name):
    
    if name == 'res_ae':
        ae = get_res_ae('resnet18')
    else:
        raise RuntimeError('---> invalid network name: {}'.format(name))
    
    return ae


def get_classifier(name, num_classes):
    if name == 'resnet18':
        classifier = get_resnet18(num_classes, True)
    elif name == 'wide_resnet':
        classifier = get_wrn(num_classes, layers=40, widen_factor=2, drop_rate=0.3)
    else:
        raise RuntimeError('---> invalid network name: {}'.format(name))
    
    return classifier


def get_classifier_rot(name, num_classes):
    if name == 'resnet_rot':
        classifier = get_resnet_rot('resnet18', num_classes)
    else:
        raise RuntimeError('<--- invalid model arch: {}'.format(name))

    return classifier