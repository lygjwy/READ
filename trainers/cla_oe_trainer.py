import torch
import torch.nn.functional as F


class ClassifierOeTrainer():
    
    def __init__(self, classifier, train_loader_id, train_loader_ood, optimizer, scheduler):
        self.classifier = classifier
        self.train_loader_id = train_loader_id
        self.train_loader_ood = train_loader_ood
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    def train_epoch(self):
        self.classifier.train()
        
        total, correct = 0, 0
        total_loss = 0.0
        
        train_dataiter_id = iter(self.train_loader_id)
        train_dataiter_ood = iter(self.train_loader_ood)
        
        for train_step in range(1, len(train_dataiter_id) + 1):
            data, target = next(train_dataiter_id)
            data, target = data.cuda(), target.cuda()
            logits = self.classifier(data)
            
            loss = F.cross_entropy(logits, target)
            
            try:
                ood_data = next(train_dataiter_ood)
            except StopIteration:
                train_dataiter_ood = iter(self.train_loader_ood)
                ood_data = next(train_dataiter_ood)
            
            ood_data = ood_data.cuda()
            
            logits_oe = self.classifier(ood_data)
            loss_oe = -(logits_oe.mean(dim=1) - torch.logsumexp(logits_oe, dim=1)).mean()

            loss += 0.5 * loss_oe
            
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            _, pred = logits.max(dim=1)
            with torch.no_grad():
                total_loss += loss.item()
                total += target.size(0)
                correct += pred.eq(target).sum().item()
        
        # average on batch
        print('[cla oe loss: {:.4f} | cla acc: {:.4f}%]'.format(total_loss / len(self.train_loader_id), 100. * correct / total))
        metrics = {
            'train_cla_oe_loss': total_loss / len(self.train_loader_id),
            'train_cla_acc': correct / total
        }
        
        return metrics