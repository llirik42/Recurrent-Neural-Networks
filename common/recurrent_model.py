import numpy as np

from .layer import Layer
from .metrics import calculate_mse


class RecurrentModel:
    __layers: list[Layer]

    def __init__(self, layers: list[Layer]) -> None:
        self.__layers = layers

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        return self.__forward(sequence)

    def train(self, sequence: np.ndarray, y_true: np.ndarray, learning_rate: float) -> float:
        y_pred: np.ndarray = self.__forward(sequence)
        y_true_reshaped: np.ndarray = y_true.reshape(-1, 1)

        mse_error: float = calculate_mse(y_pred=y_pred, y_true=y_true_reshaped)
        mse_grad: np.ndarray = y_pred - y_true_reshaped

        self.__backward(gradient=mse_grad, learning_rate=learning_rate)

        return mse_error

    def __forward(self, sequence: np.ndarray) -> np.ndarray:
        result: np.ndarray = sequence

        for i, layer in enumerate(self.__layers):
            result = layer.forward(result)

        return result

    def __backward(self, gradient: np.ndarray, learning_rate: float) -> None:
        current_gradient: np.ndarray = gradient
        for layer in self.__layers[::-1]:
            current_gradient = layer.backward(current_gradient, learning_rate)
