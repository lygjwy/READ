import numpy as np
from sklearn import metrics


# FPR@TPR
def fpr_tpr(conf, label, tpr):
    # FPR@{tpr}TPR
    conf_ind = conf[label == 0]
    conf_ood = conf[label == 1]
    len_ind = len(conf_ind)
    len_ood = len(conf_ood)

    num_tp = int(np.floor(tpr * len_ind))
    thresh = np.sort(conf_ind)[-num_tp]

    num_fp = np.sum(conf_ood > thresh)
    fpr = num_fp / len_ood
    return fpr


def auc_roc_pr(conf, label):
    indicator_ind = np.zeros_like(label)
    indicator_ind[label == 0] = 1

    fpr, tpr, thresholds = metrics.roc_curve(indicator_ind, conf)
    precision_in, recall_in, thresholds_in = metrics.precision_recall_curve(indicator_ind, conf)
    precision_out, recall_out, thresholds_out = metrics.precision_recall_curve(1 - indicator_ind, 1 - conf)

    auroc = metrics.auc(fpr, tpr)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out


# def acc(pred, label):


def compute_all_metrics(conf, label, verbose=True):
    tpr = 0.95
    fpr_at_tpr = fpr_tpr(conf, label, tpr)
    auroc, aupr_in, aupr_out = auc_roc_pr(conf, label)
    # acc = 

    if verbose:
        # print(fpr_at_tpr, auroc, aupr_in, aupr_out)
        # print("FPR@{}TPR: {:.2f}, AUROC: {:.2f}, AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}".format(tpr, 100 * fpr_at_tpr, 100 * auroc, 100 * aupr_in, 100 * aupr_out))
        print('[auroc: {:.4f}, aupr_in: {:.4f}, aupr_out: {:.4f}, fpr@95tpr: {:.4f}]'.format(auroc, aupr_in, aupr_out, fpr_at_tpr))
    results = [fpr_at_tpr, auroc, aupr_in, aupr_out]
    return results
