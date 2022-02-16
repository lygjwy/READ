import torch
import torch.nn.functional as F


class ClassifierHybridAuxiliaryTrainer():
    
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
        
        total = 0
        ori_cla_correct, ori_rot_correct = 0, 0
        rec_cla_correct, rec_rot_correct = 0, 0
        total_loss = 0.0
        
        for sample in self.train_loader:
            assert len(sample) == 9
            ori_img_0, ori_img_1, ori_img_2, ori_img_3, rec_img_0, rec_img_1, rec_img_2, rec_img_3, cla_target = sample
            ori_img_0, ori_img_1, ori_img_2, ori_img_3 = ori_img_0.cuda(), ori_img_1.cuda(), ori_img_2.cuda(), ori_img_3.cuda()
            rec_img_0, rec_img_1, rec_img_2, rec_img_3 = rec_img_0.cuda(), rec_img_1.cuda(), rec_img_2.cuda(), rec_img_3.cuda()
            cla_target = cla_target.cuda()
            
            batch_size = cla_target.size(0)
            rot_target = torch.cat((
                torch.zeros(batch_size),
                torch.ones(batch_size),
                2 * torch.ones(batch_size),
                3 * torch.ones(batch_size)
            ), 0).long().cuda()
            
            ori_cla_logits, ori_i0_rot_logits = self.classifier(ori_img_0)
            _, ori_i1_rot_logits = self.classifier(ori_img_1)
            _, ori_i2_rot_logits = self.classifier(ori_img_2)
            _, ori_i3_rot_logits = self.classifier(ori_img_3)
            
            ori_rot_logits = torch.cat((
                ori_i0_rot_logits,
                ori_i1_rot_logits,
                ori_i2_rot_logits,
                ori_i3_rot_logits
            ), 0)
            
            rec_cla_logits, rec_i0_rot_logits = self.classifier(rec_img_0)
            _, rec_i1_rot_logits = self.classifier(rec_img_1)
            _, rec_i2_rot_logits = self.classifier(rec_img_2)
            _, rec_i3_rot_logits = self.classifier(rec_img_3)
            
            rec_rot_logits = torch.cat((
                rec_i0_rot_logits,
                rec_i1_rot_logits,
                rec_i2_rot_logits,
                rec_i3_rot_logits
            ), 0)
            
            ori_cla_loss = F.cross_entropy(ori_cla_logits, cla_target)
            ori_rot_loss = F.cross_entropy(ori_rot_logits, rot_target)
            
            rec_cla_loss = F.cross_entropy(rec_cla_logits, cla_target)
            rec_rot_loss = F.cross_entropy(rec_rot_logits, rot_target)
            
            loss = ori_cla_loss + ori_rot_loss + rec_cla_loss + rec_rot_loss
            
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
                
            _, ori_cla_pred = ori_cla_logits.max(dim=1)
            _, ori_rot_pred = ori_rot_logits.max(dim=1)
            
            _, rec_cla_pred = rec_cla_logits.max(dim=1)
            _, rec_rot_pred = rec_rot_logits.max(dim=1)
            
            with torch.no_grad():
                total_loss += loss.item()
                total += batch_size
                ori_cla_correct += ori_cla_pred.eq(cla_target).sum().item()
                ori_rot_correct += ori_rot_pred.eq(rot_target).sum().item()
                rec_cla_correct += rec_cla_pred.eq(cla_target).sum().item()
                rec_rot_correct += rec_rot_pred.eq(rot_target).sum().item()
                
        # average on batch
        print('[loss: {:.4f} | ori cla acc: {:.4f}% | ori rot acc: {:.4f}% | rec cla acc {:.4f}% | rec rot acc: {:.4f}%]'.format(
            total_loss / len(self.train_loader),
            100. * ori_cla_correct / total,
            100. * ori_rot_correct / (total * 4),
            100. * rec_cla_correct / total,
            100. * rec_rot_correct / (total * 4)
        ))
        
        metrics = {
            'train_loss': total_loss / len(self.train_loader),
            'ori_train_cla_acc': ori_cla_correct / total,
            'rec_train_cla_acc': rec_cla_correct / total
        }
        
        return metrics