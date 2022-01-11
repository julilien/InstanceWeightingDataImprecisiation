import itertools
import logging

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import numpy as np

from contextlib import ExitStack
import sys
import os
from datetime import datetime

import mlflow
import ray
from sklearn.metrics import zero_one_loss

from data_imp.clf.clf_meta import SemiSupervisedClassifier
from data_imp.clf.svm import SVMWrapper
from data_imp.exp.exp_utils import get_model_parameters
from data_imp.exp.model_exp import get_model_fn_for_name
from data_imp.utils.eval_utils import perform_cv_parameter_optimization, combine_params, init_best_score, \
    log_scores_artifact, get_fold_data, merge_params, update_best, init_mlflow_run
from data_imp.utils.io_utils import read_dataset_from_openml
from defs import get_config


def test_semi_sup(clf_fn, param_dict, dataset_id, seed, num_outer_folds, num_inner_folds, semi_part_prob,
                  track_mlflow, num_cores, config, exp_name, target_selector=-1, metric_fn=zero_one_loss):
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

        if not isinstance(clf_fn(), SemiSupervisedClassifier):
            raise NotImplementedError
        X, y = read_dataset_from_openml(dataset_id, classification=True, target_selector=target_selector)

        # Determine C parameter
        internal_c_opt = {}
        internal_c_opt["C"] = [0.01, 0.1, 0.5, 1, 5, 10, 25, 50, 100]
        logging.info("Determining C parameter for dataset...")
        internal_static_params_dict = {"random_state": seed}
        best_param_dict, _ = perform_cv_parameter_optimization(SVMWrapper, internal_c_opt, X, y, metric_fn,
                                                               num_folds=num_inner_folds, min_is_better=True, seed=seed,
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
            mlflow.log_param("semi_part_prob", semi_part_prob)

        cv = KFold(n_splits=num_outer_folds, shuffle=True, random_state=seed)
        fold_scores = []

        if track_mlflow:
            parent_run_id = mlflow.active_run().info.run_id
        else:
            parent_run_id = -1

        it = 0
        for train, test in cv.split(X, y):
            @ray.remote
            def perform_inner_fold(train, test, it, logging_lvl):
                logging.basicConfig(level=logging_lvl)

                if track_mlflow:
                    init_mlflow_run(config, exp_name, parent_run_id)

                with ExitStack() as stack:
                    if track_mlflow:
                        stack.enter_context(mlflow.start_run(nested=True))

                    if track_mlflow:
                        mlflow.log_param("model", clf_fn.__name__)
                        mlflow.log_param("seed", seed)
                        mlflow.log_param("fold", it)
                        mlflow.log_param("dataset_id", dataset_id)
                        mlflow.log_param("fuzzify_prob", semi_part_prob)
                    logging.debug("Starting fold {}...".format(it))

                    X_train, y_train, X_test, y_test = get_fold_data(X, y, train, test)

                    best_param_dict, scores = sem_supervised_ho(clf_fn, params_to_optimize, X_train, y_train,
                                                                semi_part_prob=semi_part_prob, metric_fn=metric_fn,
                                                                num_folds=num_inner_folds,
                                                                static_param_dict=static_params, seed=seed)

                    logging.debug(
                        "Finished hyperparameter optimization. Now training the classifier with best params {}...".format(
                            best_param_dict))

                    best_param_dict = merge_params(best_param_dict, static_params)
                    best_param_dict["random_state"] = seed

                    fold_score = sem_supervised_run(clf_fn, best_param_dict, X_train, y_train, X_test, y_test,
                                                    semi_part_prob, seed, metric_fn)

                    if track_mlflow:
                        log_scores_artifact(scores)
                        mlflow.log_param("best param", best_param_dict)
                        mlflow.log_metric("fold_err", fold_score)

                    logging.debug("Score for fold {}: {} (best params: {})".format(it, fold_score, best_param_dict))

                    # fold_scores.append(fold_score)
                    return fold_score

            fold_scores.append(perform_inner_fold.remote(train, test, it, logging.getLogger().level))
            it += 1

        fold_scores = ray.get(fold_scores)
        eval_result = np.mean(fold_scores)

        if track_mlflow:
            mlflow.log_metric("err", eval_result)

        logging.info("Final score: {}".format(eval_result))


def sem_supervised_ho(clf_fn, params_to_optimize, X, y, semi_part_prob, metric_fn, num_folds, min_is_better=True,
                      static_param_dict=None, seed=42):
    best_score = init_best_score(min_is_better)
    best_param_values = None
    scores = []

    param_values = []
    param_keys = []
    for key in params_to_optimize:
        param_values.append(params_to_optimize[key])
        param_keys.append(key)

    for param_comb in itertools.product(*param_values):
        # param_comb is list of values
        logging.debug("Optimizing parameters {}={}...".format(param_keys, param_comb))

        param_dict = combine_params(param_comb, param_keys, static_param_dict, seed)

        score = sem_supervised_cv(clf_fn, param_dict, X, y, semi_part_prob=semi_part_prob, metric_fn=metric_fn,
                                  agg_fn=np.mean, num_folds=num_folds, seed=seed)
        scores.append(score)
        logging.debug("CV score for parameters {}={}: {}".format(param_keys, param_comb, score))

        best_score, best_param_values = update_best(best_score, score, best_param_values, param_comb, min_is_better)

    assert best_param_values is not None
    best_param_dict = {}
    for i in range(len(param_keys)):
        best_param_dict[param_keys[i]] = best_param_values[i]

    return best_param_dict, scores


def sem_supervised_cv(clf_fn, param_dict, X, y, semi_part_prob, metric_fn, agg_fn=None, num_folds=5, seed=42):
    cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    fold_scores = []
    for train, test in cv.split(X, y):
        @ray.remote
        def perform_inner_fold(train, test, logging_lvl):
            logging.basicConfig(level=logging_lvl)

            X_train, y_train, X_test, y_test = get_fold_data(X, y, train, test)

            fold_score = sem_supervised_run(clf_fn, param_dict, X_train, y_train, X_test, y_test, semi_part_prob,
                                            seed=seed, metric_fn=metric_fn)

            logging.debug("Inner fold score: {}".format(fold_score))
            return fold_score

        fold_scores.append(perform_inner_fold.remote(train, test, logging.getLogger().level))
    fold_scores = ray.get(fold_scores)
    return agg_fn(fold_scores)


def sem_supervised_run(clf_fn, param_dict, X_train, y_train, X_test, y_test, semi_part_prob, seed, metric_fn):
    np.random.seed(seed)
    if semi_part_prob > 0.0:
        X_labeled, X_unlabeled, y_labeled, _ = train_test_split(X_train, y_train, test_size=semi_part_prob,
                                                                random_state=seed, stratify=y_train)
    else:
        X_labeled = X_train
        y_labeled = y_train
        X_unlabeled = np.empty(shape=[0, X_labeled.shape[1]])

    clf = clf_fn(**param_dict)
    clf.fit(X_labeled, y_labeled, X_unlabeled)

    preds = clf.predict(X_test)
    score = metric_fn(y_test, preds)

    return score


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

    semi_part_prob = None
    if len(sys.argv) > 4:
        semi_part_prob = float(sys.argv[4])
        assert 0 <= semi_part_prob <= 1.0

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
    if isinstance(clf_fn(), SemiSupervisedClassifier):
        problem_type = "ssclf"
    else:
        problem_type = "reg"
    exp_name = config["EXPERIMENTS"]["SEMSUP_EXP_PREFIX"] + "_{}_{}".format(problem_type,
                                                                            datetime.today().strftime('%Y%m%d'))
    if track_mlflow:
        mlflow.set_experiment(exp_name)

    # Get hyperparameter being optimized
    param_dict = get_model_parameters(clf_name)

    test_semi_sup(clf_fn, param_dict, dataset_id, seed, num_outer_folds, num_inner_folds, semi_part_prob,
                  num_cores=num_cores, config=config, exp_name=exp_name, track_mlflow=track_mlflow)
