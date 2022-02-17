import torch
import torch.nn.functional as F


class ClassifierHybridOeTrainer():
    
    def __init__(self, classifier, train_loader_id, train_loader_ood, optimizer, scheduler):
        self.classifier = classifier
        self.train_loader_id = train_loader_id
        self.train_loader_ood = train_loader_ood
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    def train_epoch(self):
        self.classifier.train()
        
        total, ori_correct, rec_correct = 0, 0, 0
        total_loss = 0.0
        
        train_dataiter_id = iter(self.train_loader_id)
        train_dataiter_ood = iter(self.train_loader_ood)
        
        for train_step in range(1, len(train_dataiter_id)):
            ori_data, rec_data, target = next(train_dataiter_id)
            ori_data, rec_data, target = ori_data.cuda(), rec_data.cuda(), target.cuda()
            
            ori_logits = self.classifier(ori_data)
            rec_logits = self.classifier(rec_data)
            
            loss = F.cross_entropy(ori_logits, target) + F.cross_entropy(rec_logits, target)
            
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
            
            _, ori_pred = ori_logits.max(dim=1)
            _, rec_pred = rec_logits.max(dim=1)
            
            with torch.no_grad():
                total_loss += loss.item()
                total += target.size(0)
                ori_correct += ori_pred.eq(target).sum().item()
                rec_correct += rec_pred.eq(target).sum().item()
        
        # average on batch
        print("[cla loss: {:.4f} | ori cla acc: {:.4f}% | rec cla acc: {:.4f}% | hybrid cla acc: {:.4f}%]".format(
            total_loss / len(self.train_loader_id),
            100. * ori_correct / total,
            100. * rec_correct / total,
            100. * (ori_correct + rec_correct) / (2 * total)
            )
        )
        
        metrics = {
            'train_loss': total_loss / len(self.train_loader_id),
            'ori_train_accuracy': ori_correct / total,
            'rec_train_accuracy': rec_correct / total,
            'hybrid_train_accuracy': (ori_correct + rec_correct) / (2 * total) 
        }
        
        return metrics