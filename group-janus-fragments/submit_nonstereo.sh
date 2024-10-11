#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --ntasks-per-node=32
#SBATCH --mem=12000M               # memory (per node)
#SBATCH --time=0-25:00
#SBATCH --job-name gjanus-nonstereo

start_time=$SECONDS

module load StdEnv/2020 gcc/9.3.0
module load python/3.8.10 scipy-stack/2022a
module load rdkit/2021.09.3
module load openbabel
module load xtb

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1,1
export OMP_STACKSIZE=4G
ulimit -s unlimited

export PATH=$PATH:~/bin/stda/
export XTB4STDAHOME=~/bin/stda/

source  ~/env/stereogeneration/bin/activate

echo Working on $1
time python main.py --target=$1 --num_workers=$SLURM_NTASKS  --starting_pop=best --use_fragments

deactivate

end_time=$SECONDS
duration=$(( $end_time - $start_time ))

echo "stuff took $duration seconds to complete"
