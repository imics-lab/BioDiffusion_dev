<div align="center">


<!-- TITLE -->
# BioDiffusion
A Versatile Diffusion Model for Biomedical Signal Synthesis



[![Conda Test](https://github.com/ellisbrown/research-project/actions/workflows/conda-test.yml/badge.svg)](https://github.com/ellisbrown/research-project/actions/workflows/conda-test.yml)
[![arXiv](https://img.shields.io/badge/cs.LG-arXiv:1234.56789-b31b1b.svg)](https://arxiv.org/abs/1234.56789)
[![Conference](https://img.shields.io/badge/Conference-year-4b44ce.svg)](https://yourconference.org/2020)

</div>


<!-- DESCRIPTION -->
## Description
Biomedical signals are always suffering from insufficient data samples, imbalanced datasets, hard to label, and artificial noises which perturb machine learning implementations. In this study, we introduce BioDiffusion, a versatile diffusion-based probabilistic model designed for generating multivariate biomedical signals. The BioDiffusion model is capable of producing high-fidelity, non-stationary, and multivariate biomedical signals across various generation tasks such as unconditional, label-conditional, and signal-conditional generation. These generated signals can be employed to largely alleviate the aforementioned problems. Through qualitative and quantitative assessments, we evaluate the fidelity of the generated data and demonstrate its applicability of helping improve machine learning tasks accuracies on biomedical signals. Moreover, we compare our model with other state-of-the-art time-series generative models, with experimental results revealing that BioDiffusion outperforms its counterparts when generating biomedical signals.



### Conda Virtual Environment

Create the Conda virtual environment using the [requirements file](requirements.txt):
```bash
conda create --name <environment_name> --file requirements.txt


# activate the conda environment
conda activate <environment_name>

```


<!-- USAGE -->
## Usage

Please go to **src** folder.
The unconditional, label conditional, and signal conditional scripts and instrucitons on how to run are there.
In the **synthetic_data_exps** folder, there are some synthetic data examples and visualization examples. 






