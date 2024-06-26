# BiSHop: Bi-Directional Cellular Learning for Tabular Data with Generalized Sparse Modern Hopfield Model

<!---
Author: [Chenwei Xu](https://chenwei-1999.github.io/), Yu-Chao Huang, Jerry Yao-Chieh Hu, Weijian Li, Ammar Gilani, Hsi-Sheng Goan, Han Liu
Paper: 
-->
## Introduction
BiSHop leverages a sparse Hopfield model with adaptable sparsity, enhanced by column-wise and row-wise modules. It's specifically designed to address challenges in processing rotationally invariant and sparse tabular data.

## Installation
### Install the Environment
For setting up Conda environments and installing necessary packages, refer to the commands provided below (*Please install PyTorch according to the specific version of CUDA on your system*).

 ```
conda create -n BiSHop python=3.10
conda activate BiSHop
pip3 install torch --index-url https://download.pytorch.org/whl/cu121 # please install based on corresponding version
pip3 install -r requirements.txt
```

### Download Code and Datasets
To clone the project repository to your local machine, execute the following command:
```bash
git clone https://github.com/MAGICS-LAB/Bi-SHop.git
cd Bi-SHop
```
For the datasets for Baseline I, please first create the `datasets` folder
```bash
mkdir datasets
```
Please download them from the link below and place them in the `datasets` directory:
[Download Baseline I Datasets](https://drive.google.com/drive/folders/1T3oIYKXqnxyXhs-bHpGKABjR3tOHsAyr?usp=sharing)

## Reproduce Experiments
### Recording Runs
To record run details, update your API key in `utils/wandb_api_key.txt` and use the `--record` argument.


### Baseline I
Run the file launch_sweep.py and change to the dataset_name in Baseline I.
```bash 
python launch_baseline1.py --data [name of the data] --project [name of the wandb sweep project]
```
For example
```bash 
python launch_baseline1.py --data adult --project bishop_baseline1
```

### Baseline II
Run the file launch_sweep.py and change to the dataset_ID in Baseline II.
```bash 
python launch_baseline2.py --data [data_id] --project [name of the wandb sweep project]
```
For example
```bash 
python launch_baseline2.py --data 361110 --project bishop_baseline2
```

### Start Running
Upon initiating the process, you'll receive a prompt for the Wandb agent that reads: `wandb: Run sweep agent with: wandb agent [Agent Name]`. To proceed, execute the following command:
```bash
wandb agent [Agent Name]
```
## Reproduce Baselines and Ablation
To reproduce benchmark results, please checkout other available `Branches`.
<!---
 ## Citation
 -->

## Citation
If you find our work useful, please consider citing our paper:
```
@inproceedings{xu2024bishop,
  title={BiSHop: Bi-Directional Cellular Learning for Tabular Data with Generalized Sparse Modern Hopfield Model},
  author={Xu, Chenwei and Huang, Yu-Chao and Hu, Jerry Yao-Chieh and Li, Weijian and Gilani, Ammar and Goan, Hsi-Sheng and Liu, Han},
  booktitle={Forty-first International Conference on Machine Learning (ICML)},
  year={2024},
  url={https://arxiv.org/abs/2404.03830}
}
```
