#!/bin/bash

# 3.8
module load python/3.8 scipy-stack/2021a
module load StdEnv/2020 gcc/9.3.0
module load rdkit/2021.09.3
module load openbabel
module load xtb     # can also be loaded as binary

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
pip install pebble
pip install git+https://github.com/aspuru-guzik-group/group-selfies.git
pip install mapchiral   

# put in path for 
export STDAHOME='~/bin/stda'
export XTB4STDAHOME='~/bin/stda'
export PATH=$PATH:$STDAHOME
export PATH=$PATH:$XTB4STDAHOME

deactivate

