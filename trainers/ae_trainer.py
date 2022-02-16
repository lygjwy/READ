import torch
import torch.nn.functional as F


class AeTrainer():
    
    def __init__(
        self,
        ae,
        data_loader,
        optimizer,
        scheduler,
        data_mode
    ):
        self.ae = ae
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_mode = data_mode
    
    
    def train_epoch(self):
        self.ae.train()
        
        total = 0
        total_loss = 0.0
        
        for sample in self.data_loader:
            if self.data_mode == 'original':
                data, target = sample
                data, target = data.cuda(), target.cuda()
                rec_data = self.ae(data)
            elif self.data_mode == 'corrupt':
                cor_data, data, target = sample
                cor_data, data, target = cor_data.cuda(), data.cuda(), target.cuda()
                rec_data = self.ae(cor_data)
            else:
                raise RuntimeError('<--- invalid data mode: {}'.format(self.data_mode))

            loss = F.mse_loss(rec_data, data, reduction='sum')
        
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
                
            with torch.no_grad():
                total_loss += loss.item()
                total += target.size(0)
        
        # average on sample
        print('[rec loss: {:.6f}]'.format(total_loss / len(self.data_loader.dataset)))
        
        metrics = {
            'rec_err': total_loss / len(self.data_loader.dataset)
        }
        
        return metrics