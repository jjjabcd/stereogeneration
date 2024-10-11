# Stereogeneration

Studying the effects of including stereisomeric information in generative models for molecules.

Benchmarks:
- Rediscovery benchmark
- Docking score
- CD Spectra

## Preliminaries

Be sure to allow binaries to be executable. This includes
```bash
stereogeneration/docking/smina.static
~/bin/stda/g_spec
~/bin/stda/stda_v1.6.3
~/bin/stda/xtb4stda
```
otherwise there will be errors in using cd and docking fitnesses.

## Using the models

Virutal environment used created using 
```bash
./make_cc_env.sh
```

## JANUS
Go into `janus` folder.

See `main.py` for running JANUS (with and without stereoinformation). Add custom fitness function in this file. The fitness currently assumes the use of 1 core (modification is necessary).

Use the submission scripts for stereo vs. non-stereo runs:
```bash
sbatch submit_nonstereo.sh  # submit_stereo.sh
```

## REINVENT
Go into `reinvent` folder.


