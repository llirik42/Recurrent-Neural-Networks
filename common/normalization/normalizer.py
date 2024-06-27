import pandas as pd

from .feature_normalizer import FeatureNormalizer


class Normalizer:
    __feature_normalizers: dict[str, FeatureNormalizer]

    def __init__(self, df: pd.DataFrame):
        self.__feature_normalizers = {}
        for col in df.columns:
            self.__feature_normalizers[col] = FeatureNormalizer(df[col])

    def denormalize_feature(self, value: float, feature_name: str) -> float:
        return self.__feature_normalizers[feature_name].denormalize(value)

    def normalize(self, row: pd.Series) -> pd.Series:
        result: pd.Series = pd.Series()

        for col in row.index.tolist():
            result[col] = self.__feature_normalizers[col].normalize(row[col])

        return result

    def denormalize(self, row: pd.Series) -> pd.Series:
        result: pd.Series = pd.Series()

        for col, normalizer in self.__feature_normalizers.items():
            result[col] = normalizer.denormalize(row[col])

        return result
