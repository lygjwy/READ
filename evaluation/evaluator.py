import csv
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from .metrics import compute_all_metrics


class Evaluator():
    def __init__(self, net):
        self.net = net

    def inference(self, data_loader, post_processor):
        pred_list, conf_list = [], []

        for sample in data_loader:
            if type(sample) == list:
                # labeled
                data, _ = sample
            else:
                # unlabeled
                data = sample
            data = data.cuda()

            pred, conf = post_processor(self.net, data)
            
            pred_list.extend(pred.tolist())
            conf_list.extend(conf.tolist())

        pred_list = np.array(pred_list, int)
        conf_list = np.array(conf_list)

        return pred_list, conf_list
    
    def eval_classification(self, data_loader):
        self.net.eval()

        total = 0
        correct = 0
        cla_total_loss = 0.0
        
        with torch.no_grad():
            for sample in data_loader:
                data, target = sample
                data, target = data.cuda(), target.cuda()
                logit = self.net(data)
                cla_loss = F.cross_entropy(logit, target)
                
                cla_total_loss += cla_loss.item()
                
                pred = logit.data.max(dim=1)[1]
                total += target.size(dim=0)
                correct += pred.eq(target.data).sum().item()
    
        # average on batch
        print("[cla_loss: {:.4f} | cla_acc: {:.4f}%]".format(
                cla_total_loss / len(data_loader),
                100. * correct / total
            )
        )
        
        metrics = {
            'cla_acc': 100. * correct / total,
            'cla_loss': cla_total_loss / len(data_loader)
        }

        return metrics
    
    def eval_co_classification(self, data_loader):
        self.net.eval()
        
        total = 0
        cor_correct, ori_correct = 0, 0
        
        with torch.no_grad():
            for sample in data_loader:
                assert len(sample) == 3
                cor_data, data, target = sample
                cor_data, data, target = cor_data.cuda(), data.cuda(), target.cuda()
                cor_logit = self.net(cor_data)
                ori_logit = self.net(data)
                
                cor_pred = cor_logit.data.max(dim=1)[1]
                ori_pred = ori_logit.data.max(dim=1)[1]
                total += target.size(dim=0)
                cor_correct += cor_pred.eq(target.data).sum().item()
                ori_correct += ori_pred.eq(target.data).sum().item()
            
        # average on sample
        print('[cor cla acc: {:.4f}% | ori cla acc: {:.4f}%]'.format(
            100. * cor_correct / total,
            100. * ori_correct / total
            )
        )
        
        metrics = {
            'cor_cla_acc': 100. * cor_correct / total,
            'ori_cla_acc': 100. * ori_correct / total
        }
        
        return metrics


    def eval_hybrid_classification(self, data_loader):
        self.net.eval()
        
        total = 0
        ori_correct, rec_correct = 0, 0
        total_loss = 0.0
        
        with torch.no_grad():
            for sample in data_loader:
                assert len(sample) == 3
                ori_data, rec_data, target = sample
                ori_data, rec_data, target = ori_data.cuda(), rec_data.cuda(), target.cuda()
                ori_logit = self.net(ori_data)
                rec_logit = self.net(rec_data)
                
                loss = F.cross_entropy(ori_logit, target) + F.cross_entropy(rec_logit, target)
                total_loss += loss.item()
                ori_pred = ori_logit.data.max(dim=1)[1]
                rec_pred = rec_logit.data.max(dim=1)[1]
                total += target.size(dim=0)
                ori_correct += ori_pred.eq(target.data).sum().item()
                rec_correct += rec_pred.eq(target.data).sum().item()
        
        # average on sample
        print("[cla loss: {:.4f} | ori cla acc: {:.4f}% | rec cla acc: {:.4f}% | hybrid cla acc: {:.4f}%]".format(
            total_loss / len(data_loader.dataset),
            100. * ori_correct / total,
            100. * rec_correct / total,
            100. * (ori_correct + rec_correct) / (2 * total)
            )
        )
        
        metrics = {
            'test_loss': total_loss / len(data_loader),
            'ori_test_accuracy': 100. * ori_correct / total,
            'rec_test_accuracy': 100. * rec_correct / total,
            'hybrid_test_accuracy': 100. * (ori_correct + rec_correct) / (2 * total)
        }
        
        return metrics
    
    
    def visualize_reconstruction(self, x, n, rec_path):
        save_image(x.cpu(), rec_path, nrow=n)

    def eval_rec(self, data_loader, epoch, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.net.eval()
        
        rec_total_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, sample in enumerate(data_loader):
                if type(sample) == list:
                    # labeled
                    data, _ = sample
                else:
                    # unlabeld
                    data = sample
                data = data.cuda()
                
                rec_data = self.net(data)
                rec_loss = F.mse_loss(rec_data, data, reduction='sum')

                # save original-reconstruct images
                if batch_idx == 0:
                    n = min(data.size(0), 16)
                    if epoch % 20 == 0:
                        comparison = torch.cat(
                            [data[:n], rec_data.view(-1, 3, 32, 32)[:n]]
                        )
                        rec_path = output_dir / ('rec-' + str(epoch) + '.png')
                        self.visualize_reconstruction(comparison, n, rec_path)
                rec_total_loss += rec_loss.item()
        
        # average on sample
        print('[rec_loss: {:.6f}]'.format(rec_total_loss / len(data_loader.dataset)))
        metrics = {
            'rec_err': rec_total_loss / len(data_loader.dataset)
        }
        
        return metrics
    
    def eval_cor_rec(self, data_loader, epoch, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.net.eval()
        
        rec_total_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, sample in enumerate(data_loader):
                if len(sample) == 2:
                    cor_data, data = sample  # unlabeled
                elif len(sample) == 3:
                    cor_data, data, _ = sample  #  labeled
                else:
                    raise RuntimeError('<--- invalid sample length: {}'.format(len(sample)))
                cor_data, data = cor_data.cuda(), data.cuda()
                
                rec_data = self.net(cor_data)
                rec_loss = F.mse_loss(rec_data, data, reduction='sum')
                
                if batch_idx == 0:
                    n = min(data.size(0), 16)
                    if epoch % 20 == 0:
                        comparison = torch.cat(
                            [data[:n], cor_data[:n], rec_data.view(-1, 3, 32, 32)[:n]]
                        )
                        rec_path = output_dir / ('cor_rec-' + str(epoch) + '.png')
                        self.visualize_reconstruction(comparison, n, rec_path)
                rec_total_loss += rec_loss.item()
            
        # average on sample
        print('[rec_loss: {:.6f}]'.format(rec_total_loss / len(data_loader.dataset)))
        metrics = {
            'rec_err': rec_total_loss / len(data_loader.dataset)
        }
        
        return metrics
    
    def eval_ood(self, id_loader, ood_loaders, post_processor, epoch, output_dir):
        self.net.eval()
        result_dic_list = []

        # id_name = id_data_loader.dataset.name
        # print('---> Inference on ID: {} dataset ...'.format(id_name))
        _, id_conf = self.inference(id_loader, post_processor)
        id_ood_label = np.zeros(len(id_loader.dataset))

        for ood_data_loader in ood_loaders:
            result_dic = {'OOD': ood_data_loader.dataset.name}

            # print('---> Inference on ID: {} - OOD: {} dataset ...'.format(id_name, ood_name))
            _, ood_conf = self.inference(ood_data_loader, post_processor)
            ood_ood_label = np.ones(len(ood_data_loader.dataset))

            # pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            ood_label = np.concatenate([id_ood_label, ood_ood_label])
            # plot id & ood confidence distribution
            if epoch % 10 == 0:
                plt_path = output_dir / '-'.join([result_dic['OOD'], str(epoch), 'confidence.png'])
                self.visualize_conf(id_conf, ood_conf, plt_path)
            
            # print('---> Computing metrics  on ID: {} - OOD: {} dataset ...'.format(id_name, ood_name))
            fpr_at_tpr, auroc, aupr_in, aupr_out = compute_all_metrics(conf, ood_label)

            result_dic['FPR_TPR'] = fpr_at_tpr
            result_dic['AUROC'] = auroc
            result_dic['AUPR_IN'] = aupr_in
            result_dic['AUPR_OUT'] = aupr_out
        
            result_dic_list.append(result_dic)
            
        return result_dic_list
    
    def eval_cor_ood(self, id_loader, ood_loaders, post_processor, epoch, output_dir):
        self.net.eval()
        result_dic_list = []
        
        id_pred, id_conf = self.corrupt_inference(id_loader, post_processor)
        id_ood_label = np.zeros(len(id_loader.dataset))

        for ood_loader in ood_loaders:
            result_dic = {'OOD': ood_loader.dataset.name}
            ood_pred, ood_conf = self.corrupt_inference(ood_loader, post_processor)
            ood_ood_label = np.ones(len(ood_loader.dataset))
            
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            ood_label = np.concatenate([id_ood_label, ood_ood_label])
            # plot id & ood confidence distribution
            if epoch % 10 == 0:
                plt_path = output_dir / '-'.join([result_dic['OOD'], str(epoch), 'confidence.png'])
                self.visualize_conf(id_conf, ood_conf, plt_path)
            
            fpr_at_tpr, auroc, aupr_in, aupr_out = compute_all_metrics(conf, ood_label)

            result_dic['FPR_TPR'] = fpr_at_tpr
            result_dic['AUROC'] = auroc
            result_dic['AUPR_IN'] = aupr_in
            result_dic['AUPR_OUT'] = aupr_out
        
            result_dic_list.append(result_dic)
            
        return result_dic_list