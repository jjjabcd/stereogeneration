#!/bin/bash

# 3.9
# module load python/3.9 scipy-stack
# module load StdEnv/2020 gcc/10.2.0
# module load rdkit/2021.09.4
# module load gcc/9.3.0
# module load openbabel

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
pip install morfeus-ml
pip install selfies==1.0.3
pip install pyyaml

deactivate

