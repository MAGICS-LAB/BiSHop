import yaml
import argparse
import sys
import subprocess

dataset_mapping = {
    'categorical_classification': [361110, 361111, 361113, 361282, 361283, 361285, 361286],
    'categorical_regression': [361093, 361094, 361096, 361097, 361098, 361099, 361101, 361102, 361103, 361104, 361287, 361288, 361289, 361291, 361292, 361293, 361294],
    'numerical_regression': [361072, 361073, 361074, 361076, 361077, 361078, 361079, 361080, 361081, 361082, 361083, 361084, 361085, 361086, 361087, 361088, 361279, 361280, 361281],
    'numerical_classification': [361055, 361060, 361061, 361062, 361063, 361065, 361066, 361068, 361069, 361070, 361273, 361274, 361275, 361276, 361277, 361278]
}

parser = argparse.ArgumentParser(description='Create a YAML file with a given dataset ID.')
parser.add_argument('--data', type=int, help='The ID of the dataset to include in the YAML file.')
parser.add_argument('--project', type=str, default='bishop', help='The name of the wandb project.')
args = parser.parse_args()

benchmark_name = None
goal = None
for key, ids in dataset_mapping.items():
    if args.data in ids:
        benchmark_name = key
        break
if benchmark_name == 'categorical_classification' or benchmark_name == 'numerical_classification':
    goal = 'AUC'
else: goal = 'R2'

if not benchmark_name:
    print(f"Error: Dataset ID '{args.data}' not found in the dataset mapping.")
    sys.exit(1)

dataset_id = args.data
project = args.project

content = {
    "name": benchmark_name,
    "method": "random",
    "metric": {
        "goal": "maximize",
        "name": goal
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
        "dataset": {
            "value": dataset_id
        },
        "benchmark_name": {
            "value": benchmark_name
        }
    },

    "command": [
        "${env}",
        "python",
        "${program}",
        "--data",
        "OpenML", 
        "--record",
        "--sweep",
        "--train_epochs",
        "200"
    ],
    "program": "bishop.py"
}

with open('bishop.yaml', 'w') as file:
    yaml.dump(content, file, sort_keys=False)

sweep_command = f'wandb sweep bishop.yaml --project {project}'

subprocess.run(sweep_command, shell=True)