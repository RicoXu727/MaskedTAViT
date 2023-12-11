# MaskedTAViT

## Introduction

## Files
```bash
|-- masktavit
|   |-- data
|   |   |-- bair
|   |   |   |-- bair_dataset.sh
|   |   |   |-- bair_extract_images.py
|   |   |   |-- bair_image_to_hdf5.py
|   |-- models
|   |   |-- data.py
|   |   |-- vqvae.py
|   |   |-- gpt.py
|   |   |-- attention.py
|   |   |-- resnet.py
|   |   |-- utils.py
|   |   |-- i3d_pretrained_400.pt
|   |-- metric
|   |   |-- fvd.py
|   |   |-- convert_tf_pretrained.py
|   |   |-- pytorch_i3d.py
|   |-- train_vqvae.py
|   |-- train_videogpt.py
|   |-- compute_fvd.py
|-- run_script.sh
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

Install required packages
```
pip install -r requirements.txt
```

### Data Download and Preprocess
Download and Preprocess BAIR Robot data 
```
sh masktavit/data/bair/bair_dataset.sh 
```

### Train 
```
./run_script.sh
```
### Evaluation

## Maintainers
```
Longcong Xu lx2305@columbia.edu
Shiwen Tang st3510@columbia.edu
Ruochen Zhang rz2597@columbia.edu
```

## References
[1] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, Aravind Srinivas: “VideoGPT: Video Generation using VQ-VAE and Transformers”, 2021; <a href='http://arxiv.org/abs/2104.10157'>arXiv:2104.10157</a>.

[2] Krzysztof Choromanski, Han Lin, Haoxian Chen, Tianyi Zhang, Arijit Sehanobish, Valerii Likhosherstov, Jack Parker-Holder, Tamas Sarlos, Adrian Weller, Thomas Weingarten: “From block-Toeplitz matrices to differential equations on graphs: towards a general theory for scalable masked Transformers”, 2021; <a href='http://arxiv.org/abs/2107.07999'>arXiv:2107.07999</a>.

