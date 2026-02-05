TRACE
This repository provides an implementation of the paper: “TRACE: A Trajectory-Adaptive Context Encoder for Proactive POI Recommendation”

1. Environment
We report the hardware/software environment used in our experiments for reproducibility.

Hardware
GPU: NVIDIA RTX 5880 Ada Generation
CUDA: 12.9
Software
Python: 3.10.19
Pytorch: 2.8.0
First
You can pip install the requirements.txt to configure the environment.

2. Dataset
We conduct experiments on four public datasets (Foursquare-NYC, Foursquare-TKY, Gowalla-CA, Gowalla-TX, ). All datasets are preprocessed using the same pipeline. You can download them via the links below.

Foursquare：You can download the raw dataset from https://sites.google.com/site/yangdingqi/home/foursquare-dataset. If you want to know how to preprocess the data, please refer to scripts/check_ins_data_basic_preprocess.py

Gowalla：You can download the raw dataset from https://snap.stanford.edu/data/loc-Gowalla.html. If you want to know how to preprocess the data, please refer to .scripts/check_ins_data_basic_preprocess.py
3. Run
You can train the model after preprocessing the dataset.
```
$ cd main
$ python scripts/check_ins_data_basic_preprocess.py
$ python src/train.py experiment=trace
```
Acknowledgment
Our implementation is based on lightning-hydra-template: https://github.com/ashleve/lightning-hydra-template