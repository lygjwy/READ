from .resnet import get_resnet18
from .res_rot import get_resnet_rot
from .res_ae import get_res_ae
from .wide_resnet import get_wrn
from .deconf_net import get_h, DeconfNet


def get_ae(name):
    
    if name == 'res_ae':
        ae = get_res_ae('resnet18')
    else:
        raise RuntimeError('---> invalid network name: {}'.format(name))
    
    return ae


def get_feature_extractor(name, num_classes):
    if name == 'resnet18':
        feature_extractor = get_resnet18(num_classes, False)    
        feature_extractor.output_size = 1 * 512
    elif name == 'wide_resnet':
        feature_extractor = get_wrn(num_classes, layers=40, widen_factor=2, drop_rate=0.3, include_top=False)
        feature_extractor.output_size = 2 * 64
    else:
        raise RuntimeError('---> invalid network name: {}'.format(name))
    
    return feature_extractor


def get_classifier(name, num_classes):
    if name == 'resnet18':
        classifier = get_resnet18(num_classes, True)
    elif name == 'wide_resnet':
        classifier = get_wrn(num_classes, layers=40, widen_factor=2, drop_rate=0.3)
    else:
        raise RuntimeError('---> invalid network name: {}'.format(name))
    
    return classifier


def get_deconf_net(fe_name, h_name, num_classes):
    feature_extractor  = get_feature_extractor(fe_name, num_classes)
    in_features = feature_extractor.output_size
    h = get_h(h_name, in_features, num_classes)
    
    return DeconfNet(feature_extractor, h)


def get_classifier_rot(name, num_classes):
    if name == 'resnet_rot':
        classifier = get_resnet_rot('resnet18', num_classes)
    else:
        raise RuntimeError('<--- invalid model arch: {}'.format(name))

    return classifier