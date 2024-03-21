import yaml
import argparse
import sys
import subprocess

valid_dataset_names = [
    'adult', 'bank', 'blastchar', '1995_income', 'SeismicBumps', 
    'Shrutime', 'Spambase', 'Qsar', 'Jannis'
]

parser = argparse.ArgumentParser(description='Create a YAML file with a given dataset name.')
parser.add_argument('--data', type=str, help='The name of the dataset to include in the YAML file.')
parser.add_argument('--project', type=str, help='The name of wandb project')
args = parser.parse_args()

if args.data not in valid_dataset_names:
    print(f"Error: '{args.data}' is not a valid dataset name. Please choose from {valid_dataset_names}.")
    sys.exit(1)

dataset_name = args.data
project = args.project

content = {
    "name": dataset_name,  # Use the dataset name from the command-line argument
    "method": "random",
    "metric": {
        "goal": "maximize",
        "name": "AUC"
    },
    "parameters": {
        "out_len": {
            "values": [2, 4, 8, 16, 24, 32, 48, 64, 128, 256, 320]
        },
        "patch_dim": {
            "values": [1, 2, 4, 6, 8, 12, 16, 24]
        },
        "emb_dim": {
            "values": [16, 24, 32, 48, 64, 128, 256, 320]
        },
        "n_agg": {
            "values": [2, 3, 4, 5, 6, 7, 8]
        },
        "factor": {
            "values": [5, 10, 15]
        },
        "d_model": {
            "values": [64, 128, 256, 512, 1024]
        },
        "d_ff": {
            "values": [64, 128, 256, 512, 1024]
        },
        "n_heads": {
            "values": [2, 4, 6, 8, 10, 12]
        },
        "e_layers": {
            "values": [2, 3, 4, 5]
        },
        "d_layers": {
            "values": [0, 1]
        },
        "dropout": {
            "values": [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        },
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-4
        },
        "seed": {
            "value": 66
        },
        # "dataset": {
        #     "value": 361282
        # },
        # "benchmark_name": {
        #     "value": 'categorical_classification'
        # }
    },
    "command": [
        "${env}",
        "python",
        "${program}",
        "--data",
        dataset_name,
        "--record",
        "--sweep",
        "--train_epochs",
        "200"
    ],
    "program": "bishop.py"
}

yaml_str = yaml.dump(content, sort_keys=False)

yaml_str = yaml_str.replace(f'\n {dataset_name}\n', f'\n \'{dataset_name}\'\n')


with open('bishop.yaml', 'w') as file:
    yaml.dump(content, file, sort_keys=False)

sweep_command = f'wandb sweep bishop.yaml --project {project}'

# Execute the wandb sweep command
subprocess.run(sweep_command, shell=True)