import os
import torch
from data.data_loader import Adult, Bank, Blastchar, Income_1995, SeismicBumps, Shrutime
from data.data_loader import Spambase, Qsar, California, Jannis
from data.data_loader import OpenML
from exp.exp_basic import Exp_Basic
from models.model import BiSHop

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, regre_metric
from utils.metrics import confusion
from utils.tools import sec2hhmmss

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.nn import DataParallel
import torch.utils.data as data_utils
import torch.optim.lr_scheduler as lr_scheduler

import os
import time
import json
import pickle
import wandb

import warnings
warnings.filterwarnings('ignore')

classification_list = ['categorical_classification', 'categorical_classification_small', 'categorical_classification_large',
                        'numerical_classification', 'numerical_classification_small', 'numerical_classification_large']
regression_list = ['categorical_regression', 'categorical_regression_small', 'categorical_regression_large',
                    'numerical_regression', 'numerical_regression_small', 'numerical_regression_large']

class Exp_BiSHop(Exp_Basic):
  def __init__(self, args, extra):
    super(Exp_BiSHop, self).__init__(args)
    self.extra = extra

  def _acquire_device(self):
    if self.args.use_gpu:
      os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
      device = torch.device('cuda:{}'.format(self.args.gpu))
      print('Use GPU: cuda:{}'.format(self.args.gpu))
    else:
      device = torch.device('cpu')
      print('Use CPU')
    return device

  def _build_model(self):
    if self.args.benchmark_name in classification_list or self.args.task == 'classification': 
      is_class = True
      print('classification task')
    elif self.args.benchmark_name in regression_list or self.args.task == 'regression': 
      is_class = False
      print('regression task')
    if self.args.d_layers == 1: 
      d_layer = self.args.e_layers + 1
    else: 
      d_layer = self.args.d_layers
    print('d layers:', d_layer)
    model = BiSHop(
    n_cat=self.args.n_cat,
    n_num=self.args.n_num,
    n_out=self.args.output_size,
    emb_dim=self.args.emb_dim, 
    out_dim=self.args.out_len, 
    patch_dim=self.args.patch_dim, 
    factor=self.args.factor, 
    flip=True, 
    n_agg=self.args.n_agg, 
    actv=self.args.mode, 
    hopfield=True, 
    d_model=self.args.d_model, 
    d_ff=self.args.d_ff, 
    n_heads=self.args.n_heads, 
    e_layer=self.args.e_layers, 
    d_layer=d_layer,
    dropout=self.args.dropout, 
    share=True, 
    share_div=8, 
    share_add=False, 
    full_dropout=False, 
    emb_dropout=0.1, 
    mlp_actv=torch.nn.ReLU(), 
    mlp_bn=False, 
    mlp_bn_final=False, 
    mlp_dropout=0.2, 
    mlp_hidden=(4,2,1), 
    mlp_skip=False, 
    mlp_softmax=False, 
    device=torch.device('cuda:0')
    ).float()

    # print('# of params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    if self.args.use_multi_gpu and self.args.use_gpu:
      model = nn.DataParallel(model, device_ids=self.args.device_ids)
    return model

  def _get_data(self):
    args = self.args
    # Define a dictionary to map args.data to data classes
    data_class_map = {
        'adult': Adult,
        'bank': Bank,
        'blastchar': Blastchar,
        '1995_income': Income_1995,
        'SeismicBumps': SeismicBumps,
        'Shrutime': Shrutime,
        'Spambase': Spambase,
        'Qsar': Qsar,
        'California': California,
        'Jannis': Jannis,
        'OpenML': lambda: OpenML(task_id=args.task_id, benchmark_name=args.benchmark_name, splits=[.7, .2, .1])
    }

    # Check if args.data is in the dictionary, and if not, set a default value (e.g., None)
    data_class = data_class_map.get(args.data, None)

    train_val_test = []
    if data_class:
      if args.data == 'OpenML':
        data = data_class()
        regre = self.args.task == 'regression' or self.args.benchmark_name in regression_list
        for flag in ['train', 'val', 'test']:
          X_cat, X_num, target, data_set, data_loader = data._get_cat_num(flag, args.batch_size, regre)
          train_val_test.append([data_set, data_loader, X_cat, X_num, target])
      else:
        for flag in ['train', 'val', 'test']:
          data = data_class(root_path=args.root_path, data_path=args.data_path, splits=[.7, .2, .1])
          X_cat, X_num, target, data_set, data_loader = data._get_raw(batch_size=args.batch_size, flag=flag, 
                                                                    extra=self.extra, seed=args.seed)
          train_val_test.append([data_set, data_loader, X_cat, X_num, target])
    else:
        print(f"Unsupported data: {args.data}")

    return train_val_test

  def _select_optimizer(self):
    model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
    return model_optim

  def _select_criterion(self):
    if self.args.task == 'classification' or self.args.benchmark_name in classification_list:
      criterion =  nn.CrossEntropyLoss()
      print('criterion: CrossEntropy')
    elif self.args.task == 'regression' or self.args.benchmark_name in regression_list:
      criterion =  nn.MSELoss()
      print('criterion: MSELoss')
    return criterion

  def vali(self, vali_loader, criterion):
    self.model.eval()
    total_loss = []
    with torch.no_grad():
      for i, (batch_cat,batch_num,batch_y) in enumerate(vali_loader):
        pred, true = self._process_one_batch(batch_cat, batch_num, batch_y)
        loss = criterion(pred.detach().cpu(), true.detach().cpu())
        total_loss.append(loss.detach().item())
    total_loss = np.average(total_loss)
    self.model.train()
    return total_loss

  def train(self, setting):
    train_val_test = self._get_data()
    [_, train_loader, _, X_num, _] = train_val_test[0]
    [_, vali_loader, _, _, _] = train_val_test[1]
    [_, test_loader, _, _, _] = train_val_test[2]

    self.model.get_bins(X_num.float())

    path = os.path.join(self.args.checkpoints, setting)
    if not os.path.exists(path):
      os.makedirs(path)
    # with open(os.path.join(path, "args.json"), 'w') as f:
    #   json.dump(vars(self.args), f, indent=True)
      
    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
      
    model_optim = self._select_optimizer()
    criterion =  self._select_criterion()

    # learning rate scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(model_optim, factor=0.1, eps=1e-6, verbose=True)

    for epoch in range(self.args.train_epochs):
      time_now = time.time()
      iter_count = 0
      train_loss = []
        
      self.model.train()
      epoch_time = time.time()
      for i, (batch_cat,batch_num,batch_y) in enumerate(train_loader):
        iter_count += 1
            
        model_optim.zero_grad()
        pred, true = self._process_one_batch(batch_cat, batch_num, batch_y)
        loss = criterion(pred, true)
        train_loss.append(loss.item())
            
        if (i+1) % 10 == 0:
          print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
          speed = (time.time()-time_now)/iter_count
          left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
          print('\tspeed: {:.4f}s/iter; left time: '.format(speed) + sec2hhmmss(left_time))
          iter_count = 0
          time_now = time.time()
            
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        model_optim.step()
          
      print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
      train_loss = np.average(train_loss)
      vali_loss = self.vali(vali_loader, criterion)
      test_loss = self.vali(test_loader, criterion)
      scheduler.step(vali_loss)
      # test on the fly
      if self.args.task == 'regression' or self.args.benchmark_name in regression_list:
        mse, mae, rmse, mape, mspe, r2 = self.test(setting, test_loader)
      elif self.args.task == 'classification' or self.args.benchmark_name in classification_list:
        acc, f1, auc = self.test(setting, test_loader)

      print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
        epoch + 1, train_steps, train_loss, vali_loss, test_loss))

      if self.args.record:
        if self.args.task == 'classification' or self.args.benchmark_name in classification_list:
          wandb.log({"Train Loss": train_loss, "Validation Loss": vali_loss,
                    "Test Loss": test_loss, "Test ACC": acc, "Test AUC": auc,
                    "Test F1": f1})
        elif self.args.task == 'regression' or self.args.benchmark_name in regression_list:
          wandb.log({"Train Loss": train_loss, "Validation Loss": vali_loss,
                    "Test Loss": test_loss, "MSE": mse, "MAE": mae, "RMSE": rmse,
                    "MAPE": mape, "MSPE": mspe, "R2": r2})


      early_stopping(vali_loss, self.model, path)
      if early_stopping.early_stop:
        print("Early stopping")
        break

    adjust_learning_rate(model_optim, epoch+1, self.args)
        
    best_model_path = path+'/'+'checkpoint.pth'
    self.model.load_state_dict(torch.load(best_model_path))
    state_dict = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
    torch.save(state_dict, path+'/'+'checkpoint.pth')
    return self.model, train_loader, vali_loader, test_loader

  def test(self, setting, test_loader, save_pred = False, inverse = False):
        
    self.model.eval()
        
    preds = []
    trues = []
    metrics_all = []
    instance_num = 0
        
    with torch.no_grad():
      for _, (batch_cat,batch_num,batch_y) in enumerate(test_loader):
        pred, true = self._process_one_batch(batch_cat, batch_num, batch_y, inverse)
        preds += pred.detach().cpu().numpy().tolist()
        trues += true.detach().cpu().numpy().tolist()
        if self.args.task == 'regression' or self.args.benchmark_name in regression_list:
          batch_size = pred.shape[0]
          instance_num += batch_size
          batch_metric = np.array(regre_metric(pred.detach().cpu().numpy().reshape(-1), true.detach().cpu().numpy().reshape(-1))) * batch_size
          metrics_all.append(batch_metric)
          if (save_pred):
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
    if self.args.task == 'regression' or self.args.benchmark_name in regression_list:
      metrics_all = np.stack(metrics_all, axis = 0)
      metrics_mean = metrics_all.sum(axis = 0) / instance_num
      mean_true = np.mean(np.array(trues))
      # sst = np.sum((np.array(trues) - mean_true)**2)
      # ssr = np.sum((np.array(trues) - np.array(preds))**2)
      # print(mean_true, sst, ssr, 1-(ssr/sst))

    # result save
    folder_path = './results/' + setting +'/'
    if not os.path.exists(folder_path):
      os.makedirs(folder_path)

    if self.args.task == 'regression' or self.args.benchmark_name in regression_list:
      mae, mse, rmse, mape, mspe, r2 = metrics_mean
      print('mse:{}, mae:{}, rmse:{}, r2:{}'.format(mse, mae, rmse, r2))
      np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe, r2]))

    if self.args.task == 'classification' or self.args.benchmark_name in classification_list:
      if self.args.output_size == 2:
        print('(tn, fp, fn, tp)', confusion(torch.tensor(preds), torch.tensor(trues)))
      # print(preds[:10], trues[:10])
      acc, f1, auc = metric(torch.tensor(preds), torch.tensor(trues), self.args.output_size)
      print('acc:{}, f1:{}, auc:{}'.format(acc, f1, auc))
      np.save(folder_path+'metrics.npy', np.array([acc, f1, auc]))

    if (save_pred):
      preds = np.concatenate(preds, axis = 0)
      trues = np.concatenate(trues, axis = 0)
      np.save(folder_path+'pred.npy', preds)
      np.save(folder_path+'true.npy', trues)

    if self.args.task == 'regression' or self.args.benchmark_name in regression_list: return mse, mae, rmse, mape, mspe, r2
    if self.args.task == 'classification' or self.args.benchmark_name in classification_list: return acc, f1, auc

  def _process_one_batch(self, batch_cat, batch_num, batch_y, inverse = False):
    if self.args.task == 'classification' or self.args.benchmark_name in classification_list:
      batch_cat = batch_cat.long().to(self.device)
      batch_num = batch_num.float().to(self.device)
      batch_y = batch_y.long().to(self.device)
    elif self.args.task == 'regression' or self.args.benchmark_name in regression_list:
      batch_cat = batch_cat.long().to(self.device)
      batch_num = batch_num.float().to(self.device)
      batch_y = batch_y.float().to(self.device)

    outputs = self.model(batch_cat, batch_num)

    return outputs, batch_y