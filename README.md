# Does the Definition of Difficulty Matter? Scoring Functions and their Role for Curriculum Learning

This repository contains the configurations and code to reproduce the experiments and analyses of the paper
"Does the Definition of Difficulty Matter? Scoring Functions and their Role for Curriculum Learning"
published in [IJCNN 2026](https://doi.org/10.1109/IJCNN69854.2026.11729720) ([arXiv](https://doi.org/10.48550/arXiv.2411.00973)).
The implementation is based on [aucurriculum `v0.1.0`](https://github.com/autrainer/aucurriculum)
and [autrainer `v0.4.0`](https://github.com/autrainer/autrainer).

Implemented by [Simon David Noel Rampp](https://github.com/ramppdev)
with contributions by [Manuel Milling](https://github.com/millinma)
and [Andreas Triantafyllopoulos](https://github.com/ATriantafyllopoulos).

## Experiment Reproduction

All experiments were conducted using Python 3.10 and PyTorch 2.1.0,
and training was performed across a variety of GPU architectures,
including NVIDIA GeForce GTX TITAN X, GeForce RTX 3090, and A40 GPUs.

After cloning the repository and navigating to the root directory,
create a virtual environment and install the dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install aucurriculum==0.1.0
pip install efficientnet-pytorch==0.7.1
```

The experiments are organized in the `cifar` (for the CIFAR-10 experiments)
and `dcase` (for the DCASE2020 experiments) directories
(the following steps should be executed for each dataset separately).
To reproduce the experiments, navigate to the respective directory:

```bash
cd <dataset> # replace <dataset> with cifar or dcase
```

Fetch the data and models and preprocess the dataset:

```bash
aucurriculum fetch -cn train
aucurriculum preprocess -cn train # only for DCASE2020
```

Launch the initial grid search and seed baselines training:

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
aucurriculum postprocess results <dataset> -a seed # replace <dataset> with cifar or dcase
```

To reproduce the analyses, tables, and figures,
execute the respective notebooks in the root directory of the repository.

## Citation

If you use this code or _aucurriculum_ in your research, please cite the following paper:

```bibtex
@inproceedings{rampp2026difficulty,
  title         = {Does the Definition of Difficulty Matter? Scoring Functions and their Role for Curriculum Learning},
  author        = {Rampp, Simon and Milling, Manuel and Triantafyllopoulos, Andreas and Schuller, Bj{\"o}rn W.},
  booktitle     = {2026 International Joint Conference on Neural Networks (IJCNN)},
  year          = {2026},
  pages         = {1--8},
  doi           = {10.1109/IJCNN69854.2026.11729720},
  eprint        = {2411.00973},
  archivePrefix = {arXiv}
}
```
