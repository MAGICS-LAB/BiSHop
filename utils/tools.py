import numpy as np
import torch
import json
import torch
import pandas
import random
import openml

from sklearn.ensemble import RandomForestClassifier
from data.data_loader import Adult, Bank, Blastchar, Income_1995, SeismicBumps, Shrutime
from data.data_loader import Spambase, Qsar, California, Jannis, ForestCoverType

def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj=='type1':
        lr_adjust = {2: args.learning_rate * 0.5 ** 1, 4: args.learning_rate * 0.5 ** 2,
                     6: args.learning_rate * 0.5 ** 3, 8: args.learning_rate * 0.5 ** 4,
                     10: args.learning_rate * 0.5 ** 5}
    elif args.lradj=='type2':
        lr_adjust = {5: args.learning_rate * 0.5 ** 1, 10: args.learning_rate * 0.5 ** 2,
                     15: args.learning_rate * 0.5 ** 3, 20: args.learning_rate * 0.5 ** 4,
                     25: args.learning_rate * 0.5 ** 5}
    else:
        lr_adjust = {}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss

class StandardScaler():
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean

def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)
    return args

def string_split(str_for_split):
    str_no_space = str_for_split.replace(' ', '')
    str_split = str_no_space.split(',')
    value_list = [eval(x) for x in str_split]

    return value_list

def sec2hhmmss(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%02i:%02i:%02i" % (hours, minutes, seconds)

def get_feature_importance(args, mode):
    X, Y = _get_XY(args, 'train')
    feature_names = [i for i in range(X.shape[1])]
    forest = RandomForestClassifier()
    forest.fit(X, Y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    forest_importances = pandas.Series(importances, index=feature_names)
    forest_sort = forest_importances.sort_values(ascending=False).index.to_numpy()
    print(forest_sort)
    if mode == 'most': remove_percent = args.rf_most / 100
    elif mode == 'least': remove_percent = args.rf_least / 100

    number_remove = np.round(len(forest_sort) * remove_percent).astype(np.int)
    if mode == 'most': index_remove = forest_sort[:number_remove]
    if mode == 'least': index_remove = forest_sort[-number_remove:]
    print('removing ', number_remove, ' features')
    print('index remove ', index_remove)

    rm_cat = sum(i < args.n_cat for i in index_remove)
    rm_num = number_remove - sum(i < args.n_cat for i in index_remove)
    print('remove cat features:', rm_cat)
    print('remove num features', rm_num)

    cat_idx = [i for i in index_remove if i < args.n_cat]
    num_idx = [i-args.n_cat for i in index_remove if i >= args.n_cat]

    n_cat = args.n_cat - rm_cat 
    n_num = args.n_num - rm_num
    return n_cat, n_num, number_remove, cat_idx, num_idx, index_remove

def get_random_feature(args):
    X, Y = _get_XY(args, 'train')
    remove_percent = args.rf_rand / 100
    number_remove = np.round(X.shape[1] * remove_percent).astype(np.int)
    index_remove = np.random.choice(np.arange(0, X.shape[1]+1), size=number_remove, replace=False)
    print('removing', number_remove, ' features')
    print('index remove ', index_remove)

    rm_cat = sum(i < args.n_cat for i in index_remove)
    rm_num = number_remove - sum(i < args.n_cat for i in index_remove)
    print('remove cat features:', rm_cat)
    print('remove num features', rm_num)

    cat_idx = [i for i in index_remove if i < args.n_cat]
    num_idx = [i-args.n_cat for i in index_remove if i >= args.n_cat]

    n_cat = args.n_cat - rm_cat 
    n_num = args.n_num - rm_num
    return n_cat, n_num, number_remove, cat_idx, num_idx, index_remove

def _get_XY(args, flag):
    data_classes = {
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
    }

    data_class = data_classes.get(args.data)
    if data_class is None:
        raise ValueError(f"Data source '{args.data}' is not supported.")

    data = data_class(root_path=args.root_path, data_path=args.data_path, splits=[0.7, 0.2, 0.1])
    X_cat, X_num, Y, _, _ = data._get_raw(batch_size=args.batch_size, flag=flag, extra=None)
    X = torch.cat((X_cat, X_num), 1)
    return X, Y

def _openml_get_info(task_id):
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    cat_cols = []
    cat_idx = []
    num_cols = []

    for idx, boolean_value in enumerate(categorical_indicator):
        if boolean_value:
          cat_idx.append(idx)
          cat_cols.append(attribute_names[idx])
        else:
          num_cols.append(attribute_names[idx])
    cat_cols = cat_cols
    num_cols = num_cols
    n_cat = len(cat_cols)
    n_num = len(num_cols)
    cat_idx = cat_idx
    n_classes = len(y.unique())
    class_suite = [337, 334]
    regre_suite = [336, 336]
    class_ids = []
    regre_ids = []
    for SUITE_ID in class_suite:
        benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite
        for idd in benchmark_suite.tasks:  # iterate over all tasks
            class_ids.append(idd)
    for SUITE_ID in regre_suite:
        benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite
        for idd in benchmark_suite.tasks:  # iterate over all tasks
            regre_ids.append(idd)
    if task_id in class_ids: 
        print('Task: Classification')
        task = 'classification'
    if task_id in regre_ids: 
        print('Task: Regression')
        task = 'regression'
    return n_cat, n_num, n_classes, task

def _check_data(args):
    data_parser = {
        'adult': {'data': 'adult.csv', 'n_cat': 8, 'n_num': 6, 'task': 'classification', 'n_classes': 2},
        'bank': {'data': 'bank-full.csv', 'n_cat': 9, 'n_num': 7, 'task': 'classification', 'n_classes': 2},
        'blastchar': {'data': 'blastchar.csv', 'n_cat': 16, 'n_num': 3, 'task': 'classification', 'n_classes': 2},
        '1995_income': {'data': 'income_1995.data', 'n_cat': 8, 'n_num': 6, 'task': 'classification', 'n_classes': 2},
        'SeismicBumps': {'data': 'seismic-bumps.arff', 'n_cat': 4, 'n_num': 14, 'task': 'classification', 'n_classes': 2},
        'Shrutime': {'data': 'shrutime.csv', 'n_cat': 4, 'n_num': 6, 'task': 'classification', 'n_classes': 2},
        'Spambase': {'data': 'spambase.data', 'n_cat': 0, 'n_num': 58, 'task': 'classification', 'n_classes': 2},
        'Qsar': {'data': 'biodeg.csv', 'n_cat': 0, 'n_num': 41, 'task': 'classification', 'n_classes': 2},
        'California': {'data': 'housing.csv', 'n_cat': 1, 'n_num': 9, 'task': 'regression'},
        'Jannis': {'data': 'jannis.arff', 'n_cat': 0, 'n_num': 54, 'task': 'classification', 'n_classes': 4},
        'ForestCoverType': {'data': 'covtype.data', 'n_cat': 0, 'n_num': 54}
    }
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.n_cat = data_info['n_cat']
        args.n_num = data_info['n_num']
        args.task = data_info['task']
        args.n_classes = data_info['n_classes']
        args.data_path = data_info['data']
        args.input_size = args.n_cat + args.n_num
        args.output_size = args.n_classes

    if args.data == 'OpenML':
        n_cat, n_num, n_classes, task = _openml_get_info(args.task_id)
        args.n_cat = n_cat
        args.n_num = n_num
        args.task = task
        args.n_classes = n_classes
        args.input_size = args.n_cat + args.n_num
        if task == 'classification': args.output_size = args.n_classes
        else: args.output_size = 1

    extra = None
    if args.rf_most != 0:
        print('Before...')
        print('cat_dim:', args.n_cat, 'num_dim', args.n_num)
        n_cat, n_num, remove_num, cat_idx, num_idx, all_idx = get_feature_importance(args, 'most')
        args.n_cat = int(n_cat)
        args.n_num = int(n_num)
        args.input_size = args.n_cat + args.n_num
        print('After...')
        extra = {'remove_num': remove_num, 'cat_idx': cat_idx, 'num_idx': num_idx, 'all_idx': all_idx}
        print('cat_dim:', args.n_cat, 'num_dim', args.n_num)
    if args.rf_least != 0:
        print('Before...')
        print('cat_dim:', args.n_cat, 'num_dim', args.n_num)
        n_cat, n_num, remove_num, cat_idx, num_idx, all_idx = get_feature_importance(args, 'least')
        args.n_cat = int(n_cat)
        args.n_num = int(n_num)
        args.input_size = args.n_cat + args.n_num
        extra = {'remove_num': remove_num, 'cat_idx': cat_idx, 'num_idx': num_idx, 'all_idx': all_idx}
        print('After...')
        print('cat_dim:', args.n_cat, 'num_dim', args.n_num)
    if args.rf_rand != 0:
        print('Before...')
        print('cat_dim:', args.n_cat, 'num_dim', args.n_num)
        n_cat, n_num, remove_num, cat_idx, num_idx, all_idx = get_random_feature(args)
        args.n_cat = int(n_cat)
        args.n_num = int(n_num)
        args.input_size = args.n_cat + args.n_num
        extra = {'remove_num': remove_num, 'cat_idx': cat_idx, 'num_idx': num_idx, 'all_idx': all_idx}
        print('After...')
        print('cat_dim:', args.n_cat, 'num_dim', args.n_num)
    return args, extra