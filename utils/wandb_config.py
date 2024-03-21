import os
import wandb

def _login():
    key_file_path = 'wandb_api_key.txt' 
    with open(key_file_path, 'r') as key_file:
        key = key_file.read().strip()
    os.environ["WANDB_API_KEY"] = key
    wandb.login()

def _sweep_config(args):
    wandb.init(
        project=args.project,
        config={"model": "BiSHop"}
    )
    # load from config file
    args.out_len = wandb.config.out_len
    args.patch_dim = wandb.config.patch_dim
    args.emb_dim = wandb.config.emb_dim
    args.n_agg = wandb.config.n_agg
    args.factor = wandb.config.factor
    args.d_model = wandb.config.d_model
    args.d_ff = wandb.config.d_ff
    args.n_heads = wandb.config.n_heads
    args.e_layers = wandb.config.e_layers
    args.d_layers = wandb.config.d_layers
    args.dropout = wandb.config.dropout
    args.learning_rate = wandb.config.learning_rate
    args.seed = wandb.config.seed
    if args.data == 'OpenML':
        args.task_id = wandb.config.dataset
        args.benchmark_name = wandb.config.benchmark_name

def _log_config(dataname, args, ii):
    # tracking
    wandb.init(
        project=args.project,
        config={
            "model": "BDSHop", "dataset": dataname, "mode": args.mode,
            "input size": args.input_size, "output len": args.out_len,
            "patch dimension": args.patch_dim, "output size": args.output_size,
            "embedding dimension": args.emb_dim, "number of aggregation": args.n_agg,
            "factor for BAModule": args.factor,
            "number of feature in GSH": args.d_model,
            "dimension of feedforward": args.d_ff,
            "number of heads": args.n_heads,
            "number of encoder layers": args.e_layers,
            "number of decoder layers respect to encoder layer": args.d_layers,
            "dropout": args.dropout,
            "batch size": args.batch_size, "epochs": args.train_epochs,
            "patience": args.patience, "learning rate": args.learning_rate, 
            "adjust learning rate": args.lradj, 
            "Remove most important features": args.rf_most,
            "Remove least importance features": args.rf_least, 
            "Remove random importance features": args.rf_rand,
            "iteration": ii, "task": args.task, "task id": args.task_id, "benchmark_name": args.benchmark_name,
            "Seed": args.seed
        }
    )