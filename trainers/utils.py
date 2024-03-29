from .ae_trainer import AeTrainer
from .cla_trainer import ClassifierTrainer
from .deconf_trainer import DeconfTrainer
from .cla_hybrid_trainer import ClassifierHybridTrainer
from .deconf_hybrid_trainer import DeconfHybridTrainer


def get_ae_trainer(ae, train_loader, optimizer, scheduler):
    return AeTrainer(ae, train_loader, optimizer, scheduler)

def get_classifier_trainer(classifier, train_loader, optimizer, scheduler):
    return ClassifierTrainer(classifier, train_loader, optimizer, scheduler)

def get_classifier_hybrid_trainer(classifier, train_loader, optimizer, scheduler):
    return ClassifierHybridTrainer(classifier, train_loader, optimizer, scheduler)

def get_deconf_trainer(deconf_net, train_loader, optimizer, h_optimizer, scheduler, h_scheduler):
    return DeconfTrainer(deconf_net, train_loader, optimizer, h_optimizer, scheduler, h_scheduler)

def get_deconf_hybrid_trainer(deconf_net, train_loader, optimizer, h_optimizer, scheduler, h_scheduler):
    return DeconfHybridTrainer(deconf_net, train_loader, optimizer, h_optimizer, scheduler, h_scheduler)