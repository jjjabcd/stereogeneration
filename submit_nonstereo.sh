#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=12
#SBATCH --mem=125000M               # memory (per node)
#SBATCH --time=0-17:00
#SBATCH --job-name nonstereo

module load python/3.8 scipy-stack
module load StdEnv/2020 gcc/9.3.0
module load rdkit/2021.09.3
module load openbabel

source  ~/env/stereogeneration/bin/activate

time python main.py 0

deactivate

