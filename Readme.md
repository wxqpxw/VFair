# VFair

This repository provides a PyTorch implementation of the VFair.


## Setup
### Prerequisitess
```
Python and miniconda
```

### Requirements
Initiate conda environment
```
conda env create -f environment.yml
```

Activate env
```
conda activate vfair
```

# Datasets
The preprocessed datasets can be found in the folders:

 * ```./data/datasets/compas/```
 * ```./data/datasets/crime/```
 * ```./data/datasets/law_school/```

Due to the memory limitation of supplementary material, Uci Adult, CelebA, AgeDB are not included.



# Train and Test

Reproduce the training and testing by running:
```
python main.py
```

One can specify the experimental setting according to argparser.py
