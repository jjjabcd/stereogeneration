#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --ntasks-per-node=64
#SBATCH --mem=125000M               # memory (per node)
#SBATCH --time=0-05:00
#SBATCH --job-name smina

module load python/3.8 scipy-stack
module load StdEnv/2020 gcc/9.3.0
module load rdkit/2021.09.3
module load openbabel

source  ~/env/stereogeneration/bin/activate

time python run_smina.py --file=data/starting_smiles.txt --num_workers=$SLURM_NTASKS --target=1OYT

deactivate

