# Stereogeneration

Studying the effects of including stereisomeric information in generative models for molecules in optimizing stereochemistry-sensitive properties. 

Preprint found on ChemRxiv: [Stereochemistry-aware string-based molecular generation](https://doi.org/10.26434/chemrxiv-2024-tkjr1). Data files are found on [Zenodo](https://doi.org/10.5281/zenodo.14545514)

## Getting started

Be sure to allow binaries to be executable. This includes
```bash
stereogeneration/docking/smina.static
~/bin/stda/g_spec
~/bin/stda/stda_v1.6.3
~/bin/stda/xtb4stda
```
otherwise there will be errors in using `cd` and `docking` fitnesses.

Initialize a python environment, here we use conda, and install the required packages.
```bash
git clone git@github.com:aspuru-guzik-group/stereogeneration.git
cd stereogeneration

conda create -n stereogeneration python=3.8
conda activate stereogeneration
pip install -r requirements.txt
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

