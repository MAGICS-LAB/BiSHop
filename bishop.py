import argparse
import os
import torch

from exp.exp_bishop import Exp_BiSHop
from utils.tools import get_feature_importance, get_random_feature, _openml_get_info, _check_data
from utils.wandb_config import _login, _sweep_config, _log_config

import wandb

classification_list = ['categorical_classification', 'categorical_classification_small', 'categorical_classification_large',
                        'numerical_classification', 'numerical_classification_small', 'numerical_classification_large']
regression_list = ['categorical_regression', 'categorical_regression_small', 'categorical_regression_large',
                    'numerical_regression', 'numerical_regression_small', 'numerical_regression_large']

def main():
    parser = argparse.ArgumentParser(description='BiSHop')

    parser.add_argument('--data', type=str, default='OpenML', help='data')
    parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='adult.csv', help='data file')    
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location to store model checkpoints')
    parser.add_argument('--task_id', type=int, help='OpenML taskid')
    parser.add_argument('--benchmark_name', type=str, help='name of the benchmark')

    parser.add_argument('--out_len', type=int, default=24, help='length of the output sequence')
    parser.add_argument('--patch_dim', type=int, default=8, help='length of the segment')
    parser.add_argument('--emb_dim', type=int, default=32, help='embedding dimension')
    parser.add_argument('--n_agg', type=int, default=4, help='window size for segment merge')
    parser.add_argument('--factor', type=int, default=10, help='factor for the TwoStageAttentionLayer')

    parser.add_argument('--d_model', type=int, default=256, help='dimension of feed-forwar network')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of MLP in transformer')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers respect to encoder layer')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')

    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
    parser.add_argument('--patience', type=int, default=40, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='optimizer initial learning rate')
    parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--mode', type=str, default='entmax', help='mode')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

    parser.add_argument('--record', default=False, action='store_true', help='Wandb Record')

    # please only use one of these three argument
    parser.add_argument('--rf_most', type=int, default=0, help='Percentage of important feature remove')
    parser.add_argument('--rf_least', type=int, default=0, help='Percentage of unimportant feature remove')
    parser.add_argument('--rf_rand', type=int, default=0, help='Percentage of randomly remove features')

    parser.add_argument('--project', type=str, default='BiSHop', help='project name')
    parser.add_argument('--sweep', default=False, action='store_true', help='HPO Sweep')

    parser.add_argument('--seed', type=int, default=66, help='Seed')

    args = parser.parse_args()

    if args.record: _login()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ','')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.sweep: _sweep_config(args)
    args, extra = _check_data(args)

    print('Args in experiment:')
    print(args)

    Exp = Exp_BiSHop

    for ii in range(args.itr):

        if args.data == 'OpenML': dataname = args.task_id
        else: dataname = args.data
        if not args.sweep and args.record: _log_config(dataname, args, ii)

        # setting record of experiments
        setting = 'BiSHop_{}_sl{}_win{}_fa{}_dm{}_nh{}_el{}_eb{}_rfm{}_rfl{}_rfr{}_itr{}'.format(dataname, 
                    args.patch_dim, args.n_agg, args.factor,
                    args.d_model, args.n_heads, args.e_layers,
                    args.emb_dim, args.rf_most, args.rf_least,
                    args.rf_rand, ii)

        exp = Exp(args, extra) # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        _, _, _, test_loader = exp.train(setting)
        
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))

        if args.benchmark_name in classification_list or args.task == 'classification': 
            acc, f1, auc = exp.test(setting, test_loader)
            if args.record: wandb.log({"Accuracy": acc, "F1": f1, "AUC": auc})
        elif args.benchmark_name in regression_list or args.task == 'regression': 
            mse, mae, rmse, mape, mspe, r2 = exp.test(setting, test_loader)
            if args.record: wandb.log({"MSE": mse, "MAE": mae, "RMSE": rmse, "MAPE": mape, "MSPE": mspe, "R2":r2})

        torch.cuda.empty_cache()
        wandb.finish()

main()