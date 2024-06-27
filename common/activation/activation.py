from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        pass
