# from .joint_trainer import JointTrainer
from .ae_trainer import AeTrainer
from .cla_trainer import ClassifierTrainer
from .cla_hybrid_trainer import ClassifierHybridTrainer
from .cla_oe_trainer import ClassifierOeTrainer
from .cla_hybrid_oe_trainer import ClassifierHybridOeTrainer


def get_ae_trainer(ae, train_loader, optimizer, scheduler, data_mode):
    return AeTrainer(ae, train_loader, optimizer, scheduler, data_mode)

def get_classifier_trainer(classifier, train_loader, optimizer, scheduler):
    return ClassifierTrainer(classifier, train_loader, optimizer, scheduler)

def get_classifier_hybrid_trainer(classifier, train_loader, optimizer, scheduler):
    return ClassifierHybridTrainer(classifier, train_loader, optimizer, scheduler)

def get_classifier_oe_trainer(classifier, train_loader_id, train_loader_ood, optimizer, scheduler):
    return ClassifierOeTrainer(classifier, train_loader_id, train_loader_ood, optimizer, scheduler)

def get_classifier_hybrid_oe_trainer(classifier, train_loader_id, train_loader_ood, optimizer, scheduler):
    return ClassifierHybridOeTrainer(classifier, train_loader_id, train_loader_ood, optimizer, scheduler)