# Stereogeneration

Studying the effects of including stereisomeric information in generative models for molecules.

Benchmarks:
- Docking score
- CD Spectra

## Using the models

Virutal environment used created using 
```bash
./make_cc_env.sh
```

See `main.py` for running JANUS (with and without stereoinformation). Add custom fitness function in this file. The fitness currently assumes the use of 1 core (modification is necessary).

Use the submission scripts for stereo vs. non-stereo runs:
```bash
sbatch submit_nonstereo.sh  # submit_stereo.sh
```


