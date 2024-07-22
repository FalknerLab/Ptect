import numpy as np
import sklearn
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def train_test_model(design_mat: np.ndarray, output: np.ndarray,
            model: sklearn.linear_model = TweedieRegressor(power=1, alpha=0.5, link='log')):
    orig_len = np.shape(output)[0]
    if design_mat.ndim < 2:
        design_mat = np.expand_dims(design_mat, axis=1)
    # if output.ndim < 2:
    #     output = np.expand_dims(output, axis=1)
    nans_x = np.any(np.isnan(design_mat), axis=1)
    nans_y = np.isnan(output)
    keep_inds = ~np.logical_or(nans_x, nans_y)
    design_mat = design_mat[keep_inds, :]
    output = output[keep_inds]
    X_train, X_test, y_train, y_test = train_test_split(design_mat, output, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    full_pred = np.nan * np.ones(orig_len)
    pred_vals = model.predict(design_mat)
    full_pred[keep_inds] = pred_vals
    return y_test, pred, full_pred, r2, model