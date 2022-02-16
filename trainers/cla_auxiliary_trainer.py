import torch
import torch.nn.functional as F


class ClassifierAuxiliaryTrainer():
    
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
        
        total, cla_correct, rot_correct = 0, 0, 0
        total_loss = 0.0
        
        for sample in self.train_loader:
            assert len(sample) == 5
            img_0, img_1, img_2, img_3, cla_target = sample
            img_0, img_1, img_2, img_3, cla_target = img_0.cuda(), img_1.cuda(), img_2.cuda(), img_3.cuda(), cla_target.cuda()
            
            batch_size = cla_target.size(0)
            rot_target = torch.cat((
                torch.zeros(batch_size),
                torch.ones(batch_size),
                2 * torch.ones(batch_size),
                3 * torch.ones(batch_size)
            ), 0).long().cuda()
            
            cla_logits, i0_rot_logits = self.classifier(img_0)
            _, i1_rot_logits = self.classifier(img_1)
            _, i2_rot_logits = self.classifier(img_2)
            _, i3_rot_logits = self.classifier(img_3)
            
            rot_logits = torch.cat((
                i0_rot_logits,
                i1_rot_logits,
                i2_rot_logits,
                i3_rot_logits
            ), 0)
            
            cla_loss = F.cross_entropy(cla_logits, cla_target)
            rot_loss = F.cross_entropy(rot_logits, rot_target)
            
            loss = cla_loss + rot_loss
            
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
                
            _, cla_pred = cla_logits.max(dim=1)
            _, rot_pred = rot_logits.max(dim=1)
            
            with torch.no_grad():
                total_loss += loss.item()
                total += batch_size
                cla_correct += cla_pred.eq(cla_target).sum().item()
                rot_correct += rot_pred.eq(rot_target).sum().item()
                
        # average on batch
        print('[loss: {:.4f} | cla acc: {:.4f}% | rot acc: {:.4f}%]'.format(
            total_loss / len(self.train_loader),
            100. * cla_correct / total,
            100. * rot_correct / (total * 4)
        ))
        
        metrics = {
            'train_loss': total_loss / len(self.train_loader),
            'train_cla_acc': cla_correct / total
        }
        
        return metrics
