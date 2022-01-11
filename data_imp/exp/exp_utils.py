import json
import os

import numpy as np

from defs import ROOT_DIR


def fuzzify_labels(y, prob, random_state):
    np.random.seed(random_state)
    assert 0 <= prob <= 1

    draw = np.random.uniform(size=y.shape[0])
    y_result = np.copy(y)
    y_result[draw <= prob] *= -1
    return y_result


def get_model_parameters(model_name, space_file_path="exp/search_spaces.json"):
    parameter_space_file_path = os.path.join(ROOT_DIR, space_file_path)
    with open(parameter_space_file_path) as ssfile:
        parameter_space = json.load(ssfile)

    return parameter_space[model_name]
