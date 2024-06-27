import numpy as np


class FeatureNormalizer:
    __min_value: float
    __max_min_diff: float

    def __init__(self, values: np.ndarray):
        self.__min_value = values.min()
        self.__max_min_diff = values.max() - self.__min_value

    def normalize(self, value: float) -> float:
        return (value - self.__min_value) / self.__max_min_diff

    def denormalize(self, value: float) -> float:
        return self.__min_value + value * self.__max_min_diff
