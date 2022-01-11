from contextlib import ExitStack
import tempfile

import numpy as np
import ray
import mlflow

from sklearn.model_selection import KFold

from data_imp.exp.exp_utils import fuzzify_labels

import itertools
import logging


def perform_simple_cv(clf_fn, param_dict, x, y, metric_fn, fuz_prob, agg_fn=None, num_folds=5, seed=42):
    cv = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    fold_scores = []
    for train, test in cv.split(x, y):

        @ray.remote
        def perform_fold(train, test, logging_lvl):
            logging.basicConfig(level=logging_lvl)

            train_x = x[train]
            train_y = y[train]
            test_x = x[test]
            test_y = y[test]

            # Fuzzify training data
            if fuz_prob is not None:
                train_y = fuzzify_labels(train_y, fuz_prob, seed)

            clf = clf_fn(**param_dict)
            clf.fit(train_x, train_y)
            preds = clf.predict(test_x)
            fold_score = metric_fn(test_y, preds)
            # fold_scores.append(fold_score)

            logging.debug("Inner fold score: {}".format(fold_score))
            return fold_score

        fold_scores.append(perform_fold.remote(train, test, logging.getLogger().level))
    final_scores = ray.get(fold_scores)
    logging.debug("Final scores: {}".format(final_scores))
    return agg_fn(final_scores)


def perform_cv_parameter_optimization(clf_fn, params_to_optimize, x, y, metric_fn, fuz_prob, num_folds=5,
                                      min_is_better=True, static_param_dict=None, seed=42):
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

        score = perform_simple_cv(clf_fn, param_dict, x, y, metric_fn=metric_fn, agg_fn=np.mean, num_folds=num_folds,
                                  seed=seed, fuz_prob=fuz_prob)
        scores.append(score)
        logging.debug("CV score for parameters {}={}: {}".format(param_keys, param_comb, score))

        best_score, best_param_values = update_best(best_score, score, best_param_values, param_comb, min_is_better)

    assert best_param_values is not None
    best_param_dict = {}
    for i in range(len(param_keys)):
        best_param_dict[param_keys[i]] = best_param_values[i]

    return best_param_dict, scores


def perform_complex_cv(clf_fn, params_to_optimize, x, y, metric_fn, num_outer_folds=10, num_inner_folds=5,
                       fuzzify_prob=None, min_is_better=True, static_param_dict=None, seed=42, track_mlflow=False,
                       dataset_id=-1, parent_run_id=None, config=None, exp_name=None):
    cv = KFold(n_splits=num_outer_folds, shuffle=True, random_state=seed)
    fold_scores = []

    it = 0
    for train, test in cv.split(x, y):
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
                    if dataset_id != -1:
                        mlflow.log_param("dataset_id", dataset_id)
                    if fuzzify_prob is not None:
                        mlflow.log_param("fuzzify_prob", fuzzify_prob)

                logging.debug("Starting fold {}...".format(it))

                train_x, train_y, test_x, test_y = get_fold_data(x, y, train, test)

                best_param_dict, scores = perform_cv_parameter_optimization(clf_fn, params_to_optimize, train_x,
                                                                            train_y, metric_fn,
                                                                            num_folds=num_inner_folds,
                                                                            min_is_better=min_is_better,
                                                                            static_param_dict=static_param_dict,
                                                                            seed=seed, fuz_prob=fuzzify_prob)

                logging.debug(
                    "Finished hyperparameter optimization. Now training the classifier with best params {}...".format(
                        best_param_dict))

                # Fuzzify training data
                if fuzzify_prob is not None:
                    train_y = fuzzify_labels(train_y, fuzzify_prob, seed)

                best_param_dict = merge_params(best_param_dict, static_param_dict)
                best_param_dict["random_state"] = seed

                clf = clf_fn(**best_param_dict)
                clf.fit(train_x, train_y)

                preds = clf.predict(test_x)
                fold_score = metric_fn(test_y, preds)

                if track_mlflow:
                    log_scores_artifact(scores)
                    mlflow.log_param("best param", best_param_dict)
                    mlflow.log_metric("fold_err", fold_score)

                logging.debug("Score for fold {}: {} (best params: {})".format(it, fold_score, best_param_dict))

                return fold_score

        # fold_scores.append(fold_score)
        fold_scores.append(perform_inner_fold.remote(train, test, it, logging.getLogger().level))
        it += 1

    fold_scores = ray.get(fold_scores)
    logging.debug("Final fold scores: {}".format(fold_scores))
    return np.mean(fold_scores)


def combine_params(param_comb, param_keys, static_param_dict, seed):
    param_dict = {}
    for i in range(len(param_comb)):
        param_dict[param_keys[i]] = param_comb[i]

    param_dict = merge_params(param_dict, static_param_dict)
    param_dict["random_state"] = int(seed)
    return param_dict


def merge_params(params1, params2):
    if params1 is not None and params2 is not None:
        return {**params1, **params2}
    elif params1 is not None:
        return params1
    elif params2 is not None:
        return params2
    else:
        return None


def init_best_score(min_is_better):
    if min_is_better:
        return np.inf
    else:
        return -np.inf


def log_scores_artifact(scores):
    # Log optimal scores as artifact
    with tempfile.NamedTemporaryFile(mode='w', suffix=".txt") as temp:
        temp.write(str(scores))
        temp.flush()
        mlflow.log_artifact(temp.name)


def init_mlflow_run(config, exp_name, parent_run_id):
    mlflow.set_tracking_uri(config["MLFLOW"]["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(exp_name)
    if mlflow.active_run() is None:
        mlflow.start_run(run_id=parent_run_id)


def get_fold_data(X, y, train_idxs, test_idxs):
    train_x = X[train_idxs]
    train_y = y[train_idxs]
    test_x = X[test_idxs]
    test_y = y[test_idxs]

    return train_x, train_y, test_x, test_y


def update_best(best_score, score, former_best_params, params, min_is_better):
    if (min_is_better and score < best_score) or (not min_is_better and score > best_score):
        return score, params
    else:
        return best_score, former_best_params
