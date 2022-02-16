# from .joint_trainer import JointTrainer
from .ae_trainer import AeTrainer
from .cla_trainer import ClassifierTrainer
from .cla_hybrid_trainer import ClassifierHybridTrainer
from .cla_auxiliary_trainer import ClassifierAuxiliaryTrainer
from .cla_hybrid_auxiliary_trainer import ClassifierHybridAuxiliaryTrainer


def get_ae_trainer(ae, train_loader, optimizer, scheduler, data_mode):
    return AeTrainer(ae, train_loader, optimizer, scheduler, data_mode)

def get_classifier_trainer(classifier, train_loader, optimizer, scheduler):
    return ClassifierTrainer(classifier, train_loader, optimizer, scheduler)

def get_classifier_hybrid_trainer(classifier, train_loader, optimizer, scheduler):
    return ClassifierHybridTrainer(classifier, train_loader, optimizer, scheduler)

def get_classifier_auxiliary_trainer(classifier, train_loader, optimizer, scheduler):
    return ClassifierAuxiliaryTrainer(classifier, train_loader, optimizer, scheduler)

def get_classifier_hybrid_auxiliary_trainer(classifier, train_loader, optimizer, scheduler):
    return ClassifierHybridAuxiliaryTrainer(classifier, train_loader, optimizer, scheduler)