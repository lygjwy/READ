import torch
import torch.nn.functional as F


class ClassifierHybridTrainer():
    
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
        
        total, correct, rec_correct = 0, 0, 0
        total_loss = 0.0
        
        for sample in self.train_loader:
            data = sample['data'].cuda()
            rec_data = sample['rec_data'].cuda()
            target = sample['label'].cuda()
            
            logit = self.classifier(data)
            rec_logit = self.classifier(rec_data)
            
            loss = F.cross_entropy(logit, target) + F.cross_entropy(rec_logit, target)
            
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            _, pred = logit.max(dim=1)
            _, rec_pred = rec_logit.max(dim=1)
            
            with torch.no_grad():
                total_loss += loss.item()
                total += target.size(0)
                correct += pred.eq(target).sum().item()
                rec_correct += rec_pred.eq(target).sum().item()
            
        # average on batch
        print("[cla loss: {:.8f} | cla acc: {:.4f}% | rec cla acc: {:.4f}% | hybrid cla acc: {:.4f}%]".format(
            total_loss / len(self.train_loader.dataset),
            100. * correct / total,
            100. * rec_correct / total,
            100. * (correct + rec_correct) / (2 * total)
            )
        )
        
        metrics = {
            'train_loss': total_loss / len(self.train_loader.dataset),
            'cla_acc': correct / total,
            'rec_cla_acc': rec_correct / total,
            'hybrid_cla_acc': (correct + rec_correct) / (2 * total) 
        }
        
        return metrics
            
        