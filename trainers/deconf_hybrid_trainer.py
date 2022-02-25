import torch
import torch.nn.functional as F


class DeconfHybridTrainer():
    
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
        
        total, ori_correct, rec_correct = 0, 0, 0
        total_loss = 0.0
        
        for sample in self.train_loader:
            assert len(sample) == 3
            ori_data, rec_data, target = sample
            ori_data, rec_data, target = ori_data.cuda(), rec_data.cuda(), target.cuda()
            
            ori_logit, _, _ = self.deconf_net(ori_data)
            rec_logit, _, _ = self.deconf_net(rec_data)
            # add extra diff loss item? ori & rec diff
            loss = 0.5 * F.cross_entropy(ori_logit, target) + 0.5 * F.cross_entropy(rec_logit, target)
            
            # backward
            self.optimizer.zero_grad()
            self.h_optimizer.zero_grad()
            
            loss.backward()
            
            self.optimizer.step()
            self.h_optimizer.step()
            
            _, ori_pred = ori_logit.max(dim=1)
            _, rec_pred = rec_logit.max(dim=1)
            
            with torch.no_grad():
                total_loss += loss.item()
                # print(loss.item())
                total += target.size(0)
                ori_correct += ori_pred.eq(target).sum().item()
                rec_correct += rec_pred.eq(target).sum().item() 
        
        self.h_scheduler.step()
        self.scheduler.step()
        
       # average on dataset
        print("[cla loss: {:.4f} | ori cla acc: {:.4f}% | rec cla acc: {:.4f}% | hybrid cla acc: {:.4f}%]".format(
            total_loss / len(self.train_loader.dataset),
            100. * ori_correct / total,
            100. * rec_correct / total,
            100. * (ori_correct + rec_correct) / (2 * total)
            )
        )
        
        metrics = {
            'train_loss': total_loss / len(self.train_loader.dataset),
            'ori_train_accuracy': ori_correct / total,
            'rec_train_accuracy': rec_correct / total,
            'hybrid_train_accuracy': (ori_correct + rec_correct) / (2 * total) 
        }
        
        return metrics
