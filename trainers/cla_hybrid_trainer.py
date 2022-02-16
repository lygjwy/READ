import torch
import torch.nn.functional as F


# train on original & reconstruct dataset
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
        
        total, ori_correct, rec_correct = 0, 0, 0
        total_loss = 0.0
        
        for sample in self.train_loader:
            assert len(sample) == 3
            ori_data, rec_data, target = sample
            ori_data, rec_data, target = ori_data.cuda(), rec_data.cuda(), target.cuda()
            
            ori_logit = self.classifier(ori_data)
            rec_logit = self.classifier(rec_data)
            
            loss = F.cross_entropy(ori_logit, target) + F.cross_entropy(rec_logit, target)
            
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            _, ori_pred = ori_logit.max(dim=1)
            _, rec_pred = rec_logit.max(dim=1)
            
            with torch.no_grad():
                total_loss += loss.item()
                total += target.size(0)
                ori_correct += ori_pred.eq(target).sum().item()
                rec_correct += rec_pred.eq(target).sum().item()
            
        # average on batch
        print("[cla loss: {:.4f} | ori cla acc: {:.4f}% | rec cla acc: {:.4f}% | hybrid cla acc: {:.4f}%]".format(
            total_loss / len(self.train_loader),
            100. * ori_correct / total,
            100. * rec_correct / total,
            100. * (ori_correct + rec_correct) / (2 * total)
            )
        )
        
        metrics = {
            'train_loss': total_loss / len(self.train_loader),
            'ori_train_accuracy': ori_correct / total,
            'rec_train_accuracy': rec_correct / total,
            'hybrid_train_accuracy': (ori_correct + rec_correct) / (2 * total) 
        }
        
        return metrics
            
        