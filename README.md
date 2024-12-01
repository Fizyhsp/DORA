# **DORA(Dynamic Optimization Prompt for Continuous Reflection of LLM-based Agent)**

Code for paper Dynamic Optimization Prompt for Continuous Reflection of LLM-based Agent.

<p align="center">
<img src=pics/DORA_Framework.png  width="80%" height="60%">
</p>

## Getting Started

## Experiments Setup

Install `Miniwob++` following instructions [here](https://github.com/alfworld/alfworld).

Install `ALFWorld` following instructions [here](https://github.com/1746104160/lawen).

## Installation

- Create env and download all the packages required as follows:

```
conda create -n DORA
conda activate DORA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install botorch -c pytorch -c gpytorch -c conda-forge
pip install -r requirements.txt
```

## Usage

1. Firstly, you need to prepare your OPENAI KEY.

2. Secondly, run the script to reproduce the experiments.

   Miniwob++:

   `cd Miniwob++`

   `bash run_dora.sh`

   ALFWorld:

   `cd ALFWorld`

   `bash run_dora.sh`

## Hyperparameters

- task: tasks performed by agents
- n_prompt_tokens: the length of the tunable prompt embeddings
- instrinsic_dim: the dimension of the projection matrix
- model_hf_dir: the open-source model used

## Results

The following are the experimental results of DORA method in Miniwob++and ALFWorld environments.

<p align="left">
<img src=pics/Miniwob++.png  width="80%" height="60%">
</p>


<p align="left">
<img src=pics/ALFWorld.png  width="80%" height="60%">
</p>
