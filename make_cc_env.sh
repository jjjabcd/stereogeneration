#!/bin/bash

# 3.8
module load python/3.8 scipy-stack
module load StdEnv/2020 gcc/9.3.0
module load rdkit/2021.09.3
module load openbabel

virtualenv --no-download ~/env/stereogeneration
source  ~/env/stereogeneration/bin/activate
pip install --upgrade pip

# pip install openbabel

pip install torch
pip install scikit-learn
pip install morfeus-ml
pip install selfies #==1.0.3
pip install pytorch-lightning
pip install pyyaml
pip install seaborn
pip install git+https://github.com/aspuru-guzik-group/group-selfies.git

deactivate

