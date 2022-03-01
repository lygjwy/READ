import torch
import torch.nn.functional as F


class ClassifierTrainer():
    
    def __init__(
        self,
        classifier,
        train_loader,
        optimizer,
        scheduler
    ):
        self.classifier = classifier
        self.train_loader = train_loader
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        
    def train_epoch(self):
        self.classifier.train()
        
        total, correct = 0, 0
        total_loss = 0.0
        
        for sample in self.train_loader:
            data = sample['data'].cuda()
            target = sample['label'].cuda()
            
            logit = self.classifier(data)
            
            loss = F.cross_entropy(logit, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            _, pred = logit.max(dim=1)
            with torch.no_grad():
                total_loss += loss.item()
                total += target.size(0)
                correct += pred.eq(target).sum().item()
        
        # average on batch
        print('[cla loss: {:.8f} | cla acc: {:.4f}%]'.format(total_loss / len(self.train_loader.dataset), 100. * correct / total))
        metrics = {
            'cla_loss': total_loss / len(self.train_loader.dataset),
            'cla_acc': 100. * correct / total
        }
        
        return metrics