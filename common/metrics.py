import numpy as np


def calculate_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    avg_true_value: float = y_true.mean()
    ss_tot: float = ((y_true - avg_true_value) ** 2).sum()
    ss_res: float = ((y_true - y_pred) ** 2).sum()
    return 1 - ss_res / ss_tot


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(calculate_mse(y_pred=y_pred, y_true=y_true))


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return ((y_true - y_pred) ** 2).mean()
