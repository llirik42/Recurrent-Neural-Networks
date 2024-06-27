import numpy as np

from .activation import Activation


class Tanh(Activation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 1 - self(x) ** 2
