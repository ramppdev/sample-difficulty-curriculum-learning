# Does the Definition of Difficulty Matter? Scoring Functions and their Role for Curriculum Learning

This repository contains the configurations and code to reproduce the experiments and analyses of the paper
"Does the Definition of Difficulty Matter? Scoring Functions and their Role for Curriculum Learning".
The implementation is based on [aucurriculum `v0.1.0`](https://github.com/autrainer/aucurriculum) and [autrainer `v0.4.0`](https://github.com/autrainer/autrainer).

Implemented by [Simon David Noel Rampp](https://github.com/ramppdev) with contributions by [Manuel Milling](https://github.com/millinma) and [Andreas Triantafyllopoulos](https://github.com/ATriantafyllopoulos).

## Experiment Reproduction

After cloning the repository and navigating to the root directory, create a virtual environment and install the dependencies:

```bash
pip install aucurriculum==0.1.0
```

The experiments are organized in the `cifar` and `dcase` directories (the following steps should be executed for each dataset separately).
To reproduce the experiments, navigate to the respective directory:

```bash
cd cifar # for CIFAR-10
cd dcase # for DCASE2020
```

Fetch the data and models and preprocess the dataset:

```bash
aucurriculum fetch -cn train
aucurriculum preprocess -cn train # only for DCASE2020
```

Launch the intial grid search and seed baselines training:

```bash
aucurriculum train -cn train
aucurriculum train -cn train_baselines
```

Launch the curriculum scoring function calculation:

```bash
aucurriculum score -cn curriculum
aucurriculum score -cn curriculum_agg_seed
```

Launch the curriculum training:

```bash
aucurriculum train -cn cl
aucurriculum train -cn cl_agg_seed
```

Postprocess the results:

```bash
aucurriculum postprocess results <dataset> -a seed
```

To reproduce the analyses, tables, and figures, execute the respective notebooks in the root directory of the repository.

## Citation

If you use this code or _aucurriculum_ in your research, please cite the following paper:

(Soon to be updated).
