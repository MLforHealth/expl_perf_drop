# "Why did the Model Fail?": Attributing Model Performance Changes to Distribution Shifts

A method to attribute model performance changes to distribution shifts in causal mechanisms. For more details please see our [ICML 2023 paper](https://arxiv.org/pdf/2210.10769.pdf).


## Installation

Our package is available on [PyPI](https://pypi.org/project/expl_perf_drop/). Simply run the following with Python >= 3.7:

```
pip install expl_perf_drop
```

## Usage

We provide the following examples as Jupyter Notebooks:
1. [Spurious Synthetic Example](examples/synthetic.ipynb) 
2. More to come!


## Reproducing the Paper


If reproducing the experiments in the paper, we recommend creating a separate Conda environment:

```
git clone https://github.com/MLforHealth/expl_perf_drop
cd expl_perf_drop/
conda env create -f environment.yml
conda activate expl_perf_drop
```

To reproduce the experiments in the paper which involve training grids of models and then generating explanations for them, use `sweep.py` as follows:

```
python -m expl_perf_drop.sweep launch \
    --experiment {experiment_name} \
    --output_dir {output_root} \
    --command_launcher {launcher} 
```

where:
- `experiment_name` corresponds to experiments defined as classes in `experiments.py`
- `output_root` is a directory where experimental results will be stored.
- `launcher` is a string corresponding to a launcher defined in scripts/launchers.py (i.e. `slurm` or `local`).


The `train_model` experiment should be ran first. The remaining experiments can be ran in any order.

Alternatively, a single explanation can also be generated by calling `explain.py` with the appropriate arguments.



## Acknowledgements

The CausalGAN portion of our CelebA experiment is heavily based on an experiment in the [parametric-robustness-evaluation](https://github.com/clinicalml/parametric-robustness-evaluation) codebase. Our Shapley Value estimation functions are taken from the [DoWhy package](https://github.com/py-why/dowhy). 


## Citation

If you use this code or package in your research, please cite the following publication:

```
@inproceedings{zhang2023did,
  title={"Why did the Model Fail?": Attributing Model Performance Changes to Distribution Shifts},
  author={Zhang, Haoran and Singh, Harvineet and Ghassemi, Marzyeh and Joshi, Shalmali},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```