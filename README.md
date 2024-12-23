# Stereogeneration

Studying the effects of including stereisomeric information in generative models for molecules in optimizing stereochemistry-sensitive properties. We perform optimization on (1) rediscovery of R-albuterol and mestranol, (2) protein-ligand docking, and a stereochemistry-specific (3) CD peak spectra score. 

Preprint found on ChemRxiv: [Stereochemistry-aware string-based molecular generation](https://doi.org/10.26434/chemrxiv-2024-tkjr1). Data files are found on [Zenodo](https://doi.org/10.5281/zenodo.14545514)

## Getting started

Initialize a python environment, here we use conda, and install the required packages.
```bash
git clone git@github.com:aspuru-guzik-group/stereogeneration.git
cd stereogeneration

conda create -n stereogeneration python=3.8
conda activate stereogeneration
pip install -r requirements.txt
```

### Use of XTB
XTB will be installed in the `requirements.txt` files. Otherwise, you can install from source from [xtb](https://github.com/grimme-lab/xtb) from the Grimme Lab. You can also install using `conda`. Use the following environment variables:
```bash
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1,1
export OMP_STACKSIZE=4G
ulimit -s unlimited
```

### CD spectra setup

Use of CD spectra task will require [stda](https://github.com/grimme-lab/stda/releases/tag/v1.6.3) and [xtb4stda](https://github.com/grimme-lab/xtb4stda) from the Grimme Lab. The binary files are found in the [`stereogeneration/stda`](stereogeneration/stda) directory. The files will have to be made executable, and added to the `$PATH` variable:
```bash
cd stereogeneration/stda
chmod +x g_spec stda_v1.6.3 xtb4stda

# set file paths which will be used by stda
export PATH=$PATH:$PWD
export XTB4STDAHOME=$PWD
```

### Docking setup

Docking requires executable of the [`smina`](stereogeneration/docking) binary:
```bash
chmod +x stereogeneration/docking/smina.static
```

## Running the models
Scripts (`main.py`) for running each model are found in the respective folders: `reinvent`, `janus`, `group-janus`. The scripts have commandline arguments that control the fitness function task, and some of the parameters of the models.

```bash
python main.py \
  --target={1SYH, 1OYT, 6Y2F, cd, fp-albuterol, fp-mestranol} \    # specify task
  --stereo                                                         # turn on stereo-awareness
```

## Analysis of results

The experiments were repeated 10 times for each model each task. The result files are found in [Zenodo](https://doi.org/10.5281/zenodo.14545514). The individual runs for each task are saved in folders `{i}_stereo` and `{i}_nonstereo` for $i \in {0,...,9}$. The figures and statistics were generated using the `analysis_all.py`, which also requires the `zinc.csv` file (available in Zenodo) to be located in the repo directory:

```bash
python analysis_all.py \
  --target={1SYH, 1OYT, 6Y2F, cd, fp-albuterol, fp-mestranol}
  --root_dir='.'    # where the dataset and `stereogeneration` import are found
  --label='1SYH'    # name for target property label (defaults to 1SYH)
  --horizontal      # toggles horizontal subplots, exclude for vertical subplots
```

