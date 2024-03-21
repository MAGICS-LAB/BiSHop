import numpy as np
import torch
from torcheval.metrics import BinaryAccuracy, BinaryAUROC, BinaryF1Score
from torchmetrics.classification import MulticlassF1Score, MulticlassAUROC
from torcheval.metrics.functional import multiclass_accuracy
from sklearn.metrics import confusion_matrix
# from torchmetrics.regression import R2Score
from sklearn.metrics import r2_score
import torch.nn.functional as F

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))
def ACC(pred, true):
    pred = torch.tensor(pred.argmax(axis=1))
    metric = BinaryAccuracy()
    metric.update(pred, true)
    return metric.compute()

def F1(pred, true):
    metric = BinaryF1Score()
    metric.update(pred, true)
    return metric.compute()

def AUC(pred, true):
    metric = BinaryAUROC()
    metric.update(pred, true)
    return metric.compute()

def MultiACC(pred, true, num_classes):
    pred = torch.tensor(pred.argmax(axis=1))
    return multiclass_accuracy(pred, true, average="macro", num_classes=num_classes)

def MultiF1(pred, true, num_classes):
    metric = MulticlassF1Score(num_classes=num_classes)
    metric.update(pred, true)
    return metric.compute()

def MutiAUC(pred, true, num_classes):
    metric = MulticlassAUROC(num_classes=num_classes)
    metric.update(pred, true)
    return metric.compute()

def confusion(pred, true):
    pred = torch.tensor(pred.argmax(axis=1))
    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
    return (tn, fp, fn, tp)

def metric(pred, true, num_classes):
    true = torch.tensor(true)
    acc = MultiACC(pred, true, num_classes)
    f1 = MultiF1(pred, true, num_classes)
    auc = MutiAUC(pred, true, num_classes)
    return acc, f1, auc

def regre_metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    r2 = r2_score(true, pred)
    
    return mae,mse,rmse,mape,mspe,r2