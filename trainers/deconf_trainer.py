import torch
import torch.nn.functional as F


class DeconfTrainer():
    
    def __init__(
        self,
        deconf_net,
        train_loader,
        optimizer,
        h_optimizer,
        scheduler,
        h_scheduler
    ):
        self.deconf_net = deconf_net
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.h_optimizer = h_optimizer
        self.scheduler = scheduler
        self.h_scheduler = h_scheduler
        
    def train_epoch(self):
        self.deconf_net.train()
        
        total, correct = 0, 0
        total_loss = 0.0
        
        for sample in self.train_loader:
            data = sample['data'].cuda()
            target = sample['label'].cuda()
            
            logit, _, _ = self.deconf_net(data)
            loss = F.cross_entropy(logit, target)
            
            self.optimizer.zero_grad()
            self.h_optimizer.zero_grad()
            
            loss.backward()
            
            self.optimizer.step()
            self.h_optimizer.step()
            
            _, pred = logit.max(dim=1)
            with torch.no_grad():
                total_loss += loss.item()
                total += target.size(0)
                correct += pred.eq(target).sum().item()
        
        self.h_scheduler.step()
        self.scheduler.step()
        
        # average on dataset
        print('[cla loss: {:.8f} | cla acc: {:.4f}%]'.format(total_loss / len(self.train_loader.dataset), 100. * correct / total))
        metrics = {
            'cla_loss': total_loss / len(self.train_loader.dataset),
            'cla_acc': correct / total
        }
        
        return metrics
      
            