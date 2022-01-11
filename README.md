# Instance Weighting through Data Imprecisiation

This repository provides the Python implementation of the paper "Instance Weighting through Data 
Imprecisiation" by Julian Lienen and Eyke Hüllermeier published in *International Journal of Approximate Reasoning* (IJAR), 2021. Please cite this work as follows:

```
@article{DBLP:journals/ijar/LienenH21,
  author    = {Julian Lienen and
               Eyke H{\"{u}}llermeier},
  title     = {Instance weighting through data imprecisiation},
  journal   = {Int. J. Approx. Reason.},
  volume    = {134},
  pages     = {1--14},
  year      = {2021},
  url       = {https://doi.org/10.1016/j.ijar.2021.04.002},
  doi       = {10.1016/j.ijar.2021.04.002},
  timestamp = {Thu, 29 Jul 2021 13:39:54 +0200},
  biburl    = {https://dblp.org/rec/journals/ijar/LienenH21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Getting started

### Dependencies

The code uses the following dependencies (Python 3.7.x):
* Numpy
* Scipy
* Cvxpy (with Cvxopt solver)
* Mlflow
* Ray
* Scikit-learn

A detailed list (including version for reproducibility) can be found in `requirements.txt`. To install all dependencies, run the following command:

```
pip install -r requirements.txt
```


Note that due to the use of ray, Windows systems are not supported. To run the framework on Windows-based systems, we 
recommend to use the Windows Linux Subsystem.

### Basic workflow

This framework relies on [Mlflow](https://mlflow.org/) to track experimental results. To do so, it logs parameters
and resulting metrics for each run, such that they can be analyzed by Mlflow's UI or using SQL scripts. However, the
program itself also logs the results with the Python `logging` framework, such that you can still access results without
getting familiar with Mlflow, although it is not convenient on a large scale.

### Run experiments

In the original paper, two experimental settings are considered: Robust binary classification and (semi-supervised) 
self-training.

To run the first settings, one has to execute the following statement:

```
python data_imp/exp/model_exp.py <model_name> <dataset_id> <seed> <noise_level> [<debug_level> <config_path>]
```

The model name can be one of SVM (non-regularized), SVMReg (L2 regularized), RSVM, OWSVM or FLSVM (which is our SVM-DI). As dataset id, you can pass any OpenML ID 
you want. The seed must be an integer number, while the noise level is a float value in [0,1]. A debug level (integer) 
higher than 0 indicates logging debug outputs with increasing verbosity. The configuration path specifies a configuration 
file (see next subchapter) relative to the project root path. If the debug level and the configuration path are not 
specified, a default debug level of `0` and the default configuration file `conf/run.ini` is used.

For the second experiment, a run can be started by
```
python data_imp/exp/sem_sup.py <model_name> <dataset_id> <seed> <unlabeled_fraction> [<debug_level> <config_path>]
```

Here, the model name is one of SSWSVM (weighted SVM) or SSFLSVM (our SVM-DI). The unlabeled fraction is also a float
parameter between [0,1], while the other parameters match those used within the first setting.

### Configuration path

For both scenarios, a configuration file path has to be specified. It uses the default Python `configparser` and contains 
the following entries:

```
[MLFLOW]
MLFLOW_TRACKING_URI = <mlflow_tracking_uri>

[EXPERIMENTS]
NUM_CORES = 4
NUM_OUTER_FOLDS = 5
NUM_INNER_FOLDS = 5
EXP_PREFIX = <mlflow_experiment_prefix_for_first_setting>
SEMSUP_EXP_PREFIX = <mlflow_experiment_prefix_for_second_setting>
```

If Mlflow should not be used, just comment out the property `MLFLOW_TRACKING_URI`.

### Parameter search spaces

To specify which parameters are tuned within the internal hyperparameter optimization, the file in 
`exp/search_spaces.json` can be modified. This provides a generic syntax to specify both ranges and fixed parameters.