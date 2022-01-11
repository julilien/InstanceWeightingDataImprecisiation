from contextlib import ExitStack
import sys
import os
from datetime import datetime
import logging

import mlflow
import ray
from sklearn.metrics import mean_squared_error, zero_one_loss

from data_imp.clf.clf_meta import Classifier
from data_imp.clf.flsvm import FLSVM
from data_imp.clf.flsvm_semisup import SemiSupFLSVM
from data_imp.clf.owsvm import OWSVM
from data_imp.clf.reg_svm import RegularizedSVM
from data_imp.clf.rsvm import RSVM
from data_imp.clf.svm import SVMWrapper
from data_imp.clf.wsvm_semisup import SemiSupWSVM
from data_imp.exp.exp_utils import get_model_parameters
from data_imp.reg.huber import HuberWrapper
from data_imp.reg.interval_reg import IntervalBasedLinearRegression
from data_imp.reg.loess import LocallyWeightedLinearRegression
from data_imp.utils.eval_utils import perform_complex_cv, perform_cv_parameter_optimization
from data_imp.utils.io_utils import read_dataset_from_openml
from defs import get_config


def get_model_fn_for_name(name):
    if name == "LocallyWeightedLinearRegression":
        return LocallyWeightedLinearRegression
    elif name == "IntervalBasedLinearRegression":
        return IntervalBasedLinearRegression
    elif name == "HuberRegressor":
        return HuberWrapper
    elif name == "SVM":
        return SVMWrapper
    elif name == "RegSVM":
        return RegularizedSVM
    elif name == "OWSVM":
        return OWSVM
    elif name == "RSVM":
        return RSVM
    elif name == "FLSVM":
        return FLSVM
    elif name == "SSFLSVM":
        return SemiSupFLSVM
    elif name == "SSWSVM":
        return SemiSupWSVM
    else:
        logging.error("Error: Could not determine an implementation for model name '{}'. Aborting...".format(name))
        raise ValueError


def test_model(clf_fn, param_dict, dataset_id, seed, num_outer_folds, num_inner_folds, fuzzify_prob=None,
               target_selector=-1, track_mlflow=True, num_cores=None, config=None, exp_name=None):
    ray.init(num_cpus=num_cores, ignore_reinit_error=True)

    static_params = {}
    params_to_optimize = {}

    for key in param_dict:
        if not isinstance(param_dict[key], list):
            static_params[key] = param_dict[key]
        else:
            params_to_optimize[key] = param_dict[key]

    with ExitStack() as stack:
        if track_mlflow:
            stack.enter_context(mlflow.start_run())

        clf_task = False
        if isinstance(clf_fn(), Classifier):
            clf_task = True
            metric_fn = zero_one_loss
        else:
            # Ensure that no fuzzification is performed for regression tasks
            fuzzify_prob = None
            metric_fn = mean_squared_error
        X, y = read_dataset_from_openml(dataset_id, classification=clf_task, target_selector=target_selector)

        if clf_task:
            # Determine C parameter
            internal_c_opt = {}
            internal_c_opt["C"] = [0.01, 0.1, 0.5, 1, 5, 10, 25, 50, 100]
            logging.info("Determining C parameter for dataset...")
            internal_static_params_dict = {"random_state": seed}
            best_param_dict, _ = perform_cv_parameter_optimization(SVMWrapper, internal_c_opt, X,
                                                                   y, metric_fn, num_folds=num_inner_folds,
                                                                   min_is_better=True, seed=seed,
                                                                   static_param_dict=internal_static_params_dict,
                                                                   fuz_prob=None)
            static_params["C"] = best_param_dict["C"]
            logging.info("Done. Selected C parameter is {}.".format(best_param_dict["C"]))

        if track_mlflow:
            mlflow.log_param("model", clf_fn.__name__)
            mlflow.log_param("dataset_id", dataset_id)
            mlflow.log_param("static_params", static_params)
            mlflow.log_param("seed", seed)
            mlflow.log_param("outer_folds", num_outer_folds)
            mlflow.log_param("inner_folds", num_inner_folds)
            mlflow.log_param("params_to_opt", params_to_optimize)
            mlflow.log_param("fuzzify_prob", fuzzify_prob)
            parent_run_id = mlflow.active_run().info.run_id
        else:
            parent_run_id = None

        eval_result = perform_complex_cv(clf_fn, params_to_optimize, X, y, metric_fn, dataset_id=dataset_id,
                                         num_outer_folds=num_outer_folds, num_inner_folds=num_inner_folds,
                                         fuzzify_prob=fuzzify_prob, static_param_dict=static_params,
                                         seed=seed, track_mlflow=track_mlflow,
                                         parent_run_id=parent_run_id, config=config,
                                         exp_name=exp_name)
        if track_mlflow:
            mlflow.log_metric("err", eval_result)

        logging.info("Final score: {}".format(eval_result))


if __name__ == "__main__":
    # Disabling parallelization if not explicitly stated (instead, ray is used)
    os.environ['OMP_NUM_THREADS'] = '1'

    # Parameters: Classifier name, dataset id, seed, [if clf: fuzzify_prob]
    clf_name = sys.argv[1]
    clf_fn = get_model_fn_for_name(clf_name)
    if clf_fn is None:
        exit(1)

    dataset_id = int(sys.argv[2])
    seed = int(sys.argv[3])

    fuzzify_prob = None
    if len(sys.argv) > 4:
        fuzzify_prob = float(sys.argv[4])
        assert 0 <= fuzzify_prob <= 1.0

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 5:
        log_lvl = int(sys.argv[5])
        if log_lvl > 0:
            logging.basicConfig(level=logging.DEBUG)

    config_path = None
    if len(sys.argv) > 6:
        config_path = sys.argv[6]

    config = get_config(config_path)
    track_mlflow = True
    try:
        mlflow.set_tracking_uri(config["MLFLOW"]["MLFLOW_TRACKING_URI"])
    except KeyError:
        track_mlflow = False

    num_outer_folds = int(config["EXPERIMENTS"]["NUM_OUTER_FOLDS"])
    num_inner_folds = int(config["EXPERIMENTS"]["NUM_INNER_FOLDS"])

    num_cores = int(config["EXPERIMENTS"]["NUM_CORES"])
    if num_cores == -1:
        num_cores = None

    # Set Mlflow experiment
    exp_name = ""
    if track_mlflow:
        if isinstance(clf_fn(), Classifier):
            problem_type = "clf"
        else:
            problem_type = "reg"
        exp_name = config["EXPERIMENTS"]["EXP_PREFIX"] + "_{}_{}".format(problem_type, datetime.today().strftime('%Y%m%d'))
        mlflow.set_experiment(exp_name)

    # Get hyperparameter being optimized
    param_dict = get_model_parameters(clf_name)

    test_model(clf_fn, param_dict, dataset_id, seed, num_outer_folds, num_inner_folds, fuzzify_prob,
               num_cores=num_cores, config=config, exp_name=exp_name, track_mlflow=track_mlflow)
