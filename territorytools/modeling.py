import numpy as np
import sklearn
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def train_test_model(design_mat: np.ndarray, output_mat: np.ndarray,
            model: sklearn.linear_model = TweedieRegressor(power=1, alpha=0.5, link='log')):
    """
    Trains and tests a model using the given design and output matrices.

    Parameters
    ----------
    design_mat : np.ndarray
        Design matrix (input features).
    output_mat : np.ndarray
        Output matrix (target values).
    model : sklearn.linear_model, optional
        Model to train (default is TweedieRegressor(power=1, alpha=0.5, link='log')).

    Returns
    -------
    y_test : np.ndarray
        Test target values.
    pred : np.ndarray
        Predicted values for the test set.
    full_pred : np.ndarray
        Predicted values for the entire dataset.
    r2 : float
        R-squared score of the model.
    model : sklearn.linear_model
        Trained model.
    """
    output = output_mat
    if design_mat.ndim < 2:
        design_mat = np.expand_dims(design_mat, axis=1)
    if output_mat.ndim < 2:
        output = np.expand_dims(output_mat, axis=1)
    nans_x = np.any(np.isnan(design_mat), axis=1)
    nans_y = np.any(np.isnan(design_mat), axis=1)
    keep_inds = ~np.logical_or(nans_x, nans_y)
    design_mat = design_mat[keep_inds, :]
    output = output[keep_inds, :]
    X_train, X_test, y_train, y_test = train_test_split(design_mat, output, test_size=0.2, random_state=42)
    if y_train.shape[1] == 1:
        y_train = y_train.squeeze()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    full_pred = np.nan * np.ones_like(output)
    pred_vals = model.predict(design_mat)
    if pred_vals.ndim < 2:
        pred_vals = pred_vals[:, None]
    full_pred[keep_inds, :] = pred_vals
    return y_test, pred, full_pred, r2, model


def chunk_shuffle(data, chunk_size, rand_seed=42, num_shuffs=1, verbose=False):
    """
    Shuffles data in chunks.

    Parameters
    ----------
    data : np.ndarray
        Data to shuffle.
    chunk_size : int
        Size of each chunk.
    rand_seed : int, optional
        Random seed for reproducibility (default is 42).
    num_shuffs : int, optional
        Number of shuffles to perform (default is 1).
    verbose : bool, optional
        Whether to print verbose output (default is False).

    Returns
    -------
    np.ndarray
        Array of shuffled data.
    """
    data_sz = len(data)
    bin_ar = np.digitize(np.arange(data_sz), bins=np.arange(data_sz, step=chunk_size))
    bin_ids = np.unique(bin_ar)
    np.random.seed(rand_seed)
    shuff_array = []
    for s in range(num_shuffs):
        if verbose:
            print(fr'Shuffle {s} of {num_shuffs}')
        perm_inds = np.random.permutation(bin_ids)
        shuff_data = np.zeros_like(data)
        c = 0
        for p in perm_inds:
            this_d = data[bin_ar == p]
            shuff_data[c:(c+len(this_d))] = this_d
            c += len(this_d)
        shuff_array.append(shuff_data)
    return np.array(shuff_array)
