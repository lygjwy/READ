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
            data, target = sample
            data, target = data.cuda(), target.cuda()
            
            logit, _, _ = self.deconf_net(data)
            loss = F.cross_entropy(logit, target)
            
            self.optimizer.zero_grad()
            self.h_optimizer.zero_grad()
            loss.backward()
            
            self.optimizer.step()
            self.scheduler.step()
            self.h_optimizer.step()
            self.h_scheduler.step()
            
            _, pred = logit.max(dim=1)
            with torch.no_grad():
                total_loss += loss.item()
                total += target.size(0)
                correct += pred.eq(target).sum().item()
        
        # average on batch
        print('[cla loss: {:.4f} | cla acc: {:.4f}%]'.format(total_loss / len(self.train_loader), 100. * correct / total))
        metrics = {
            'train_cla_loss': total_loss / len(self.train_loader),
            'train_cla_acc': correct / total
        }
        
        return metrics
      
            