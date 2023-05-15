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


<!-- SETUP -->
## Setup

> `### <<< DELETE ME:` ***Setup***
>  
> Below are some base instructions for setting up a conda environment. See this
> [guide](https://carpentries-incubator.github.io/introduction-to-conda-for-data-scientists/)
> to Conda to learn some great best practices!
> 
> Add any instructions necessary to setup the project. The best time to create
> this is ***as you are developing*** the project, while you remember the steps
> you have taken.
>
> *Brownie points*: try to follow your setup instructions to replicate the setup
> from scratch on another machine to ensure that it is truly reproducible.
> 
> `### DELETE ME >>>`


### Conda Virtual Environment

Create the Conda virtual environment using the [environment file](environment.yml):
```bash
conda env create -f environment.yml

# dynamically set python path for the environment
conda activate YOUR_PROJECT
conda env config vars set PYTHONPATH=$(pwd):$(pwd)/src
```


<!-- USAGE -->
## Usage
> `### <<< DELETE ME:` ***Usage***
>  
> Provide information on how to run your project. Delete the below example.
> 
> `### DELETE ME >>>`

```python
from foo import bar

bar.baz("hello world")
```

```bash
python -m foo.bar "hello world"
```


<!-- CONTRIBUTING -->
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.




