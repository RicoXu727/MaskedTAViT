# MaskedTAViT

## Introduction

## Files
```bash
|-- main.py
|-- masktavit
|   |-- data
|   |   |-- bair
|   |-- models
|   |   |-- vqvae.py
|   |   |-- gpt.py
|   |   |-- attention.py
|   |   |-- utils.py
|   |-- train
|   |   |-- train_videogpt.py
|   |   |-- train_vqvae.py
|   |-- test
|   |   |-- test_videogpt.py
|   |   |-- test_vqvae.py
|   |-- metric
|   |   |-- fvd.py
|-- requirements.txt
|-- .gitignore
```

## Instructions

### Setup Environment
Build a new env (Please install version 3.8, as some packages are not available in more advanced versions of python)
```
conda create -name <your env name> python=3.8
```
Activate new env
```
conda activate <your env name>
```
Clone the repository
```
git clone https://github.com/RicoXu727/MaskedTAViT.git
```

Install the CUDA Toolkit and cuDNN  
```
conda install --yes -c conda-forge cudatoolkit=11.0 cudnn
```

Install required packages
```
pip install -r requirements.txt
```

### Data Download and Preprocess
Download and Preprocess BAIR Robot data 
```
sh masktavit/data/bair/bair_dataset.sh datasets/bair
```


### Train and Test

### Evaluation

## Maintainers
```
Longcong Xu lx2305@columbia.edu
Shiwen Tang st3510@columbia.edu
```

## References
[1] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, Aravind Srinivas: “VideoGPT: Video Generation using VQ-VAE and Transformers”, 2021; <a href='http://arxiv.org/abs/2104.10157'>arXiv:2104.10157</a>.

[2] Krzysztof Choromanski, Han Lin, Haoxian Chen, Tianyi Zhang, Arijit Sehanobish, Valerii Likhosherstov, Jack Parker-Holder, Tamas Sarlos, Adrian Weller, Thomas Weingarten: “From block-Toeplitz matrices to differential equations on graphs: towards a general theory for scalable masked Transformers”, 2021; <a href='http://arxiv.org/abs/2107.07999'>arXiv:2107.07999</a>.

