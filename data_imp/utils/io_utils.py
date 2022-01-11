import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler


def read_dataset(path, header=False, delimiter=",", class_att_index=-1):
    with open(path) as file:
        # Skip header line
        if header:
            file.readline()

        result = np.loadtxt(file, delimiter=delimiter)

    feature_mask = np.ones(result.shape[1])
    feature_mask[class_att_index] = 0

    class_mask = np.zeros(result.shape[1])
    class_mask[class_att_index] = 1

    return result[:, feature_mask.astype(bool)], np.squeeze(result[:, class_mask.astype(bool)])


def read_dataset_from_openml(id, classification=True, target_selector=-1, standardize_output=True):
    X, y = fetch_openml(data_id=id, return_X_y=True, as_frame=False)

    # Impute
    if np.isnan(X).any():
        from sklearn.impute import SimpleImputer
        X = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(X)

    if target_selector != -1:
        assert y is None
        y = np.copy(X[:, target_selector])
        X = np.delete(X, target_selector, 1)

    # Transform binary dataset to targets {-1, +1} to work properly with approach
    if classification:
        if y.dtype == object:
            from sklearn.preprocessing import LabelBinarizer
            y = np.squeeze(LabelBinarizer().fit_transform(y))
        y = y.astype('int')
        min_val = np.min(y)
        max_val = np.max(y)
        y[y == min_val] = -1
        y[y == max_val] = +1
    else:
        y = y.astype('float')

        # Standardization of regression targets
        if standardize_output:
            y = np.squeeze(StandardScaler().fit_transform(np.expand_dims(y, axis=-1)))

    return X, y