import os
import pickle as pkl
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import opendatasets as od
import pandas as pd
from tqdm import tqdm

from common.activation import Tanh
from common.metrics import calculate_r2_score, calculate_rmse
from common.normalization import Normalizer
from common.preprocessing import preprocess
from common.recurrent_model import RecurrentModel
from rnn.rnn_layer import RNNLayer

TRAIN_SIZE: float = 0.8
PREPROCESSED_DATASET_PATH: str = "preprocessed.csv"
LEARNING_RATE: float = 0.005
EPOCHS: int = 10
SEQUENCE_LENGTH: int = 24
MODEL_PATH = "rnn.pkl"
DATASET_URL: str = "https://www.kaggle.com/datasets/budincsevity/szeged-weather"
TARGET: str = "Temperature (C)"
DATE: str = "Formatted Date"
SECONDS_IN_HOUR: int = 3600


def parse_datetime(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f %z")


def load_preprocessed_dataset() -> pd.DataFrame:
    return pd.read_csv(PREPROCESSED_DATASET_PATH)


def dump_preprocessed_dataset(df: pd.DataFrame) -> None:
    df.to_csv(PREPROCESSED_DATASET_PATH, index=False)


def get_preprocessed_dataset() -> pd.DataFrame:
    if os.path.isfile(PREPROCESSED_DATASET_PATH):
        print("Found preprocessed dataset!. Reading it ...")
        result: pd.DataFrame = load_preprocessed_dataset()
        print("Read preprocessed dataset!\n")
        return result

    od.download(DATASET_URL)
    result: pd.DataFrame = pd.read_csv("szeged-weather/weatherHistory.csv")

    print("Starting preprocessing...")
    result = preprocess(result)
    print("Preprocessing done!\n")

    print("Dumping preprocessed dataset ...")
    dump_preprocessed_dataset(result)
    print("Preprocessed dataset dumped!\n")

    return result


def load_model() -> RecurrentModel:
    with open(MODEL_PATH, "rb") as model_file:
        model: RecurrentModel = pkl.load(model_file)
        return model


def dump_model(model: RecurrentModel) -> None:
    with open(MODEL_PATH, "wb") as model_file:
        pkl.dump(model, model_file)


def train_model(model: RecurrentModel, normalizer: Normalizer, dates: pd.Series, train: pd.DataFrame) -> None:
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}")

        mse_error: float = 0
        sequence = []
        true_values = []
        prev_date: Optional[datetime] = None
        prev_y_true_normalized: float = 0

        for index in tqdm(range(train.shape[0])):
            normalized_row: pd.Series = normalizer.normalize(train.iloc[index])
            current_date: datetime = parse_datetime(dates.iloc[index])

            if prev_date is None:
                prev_date = current_date
                sequence.append(normalized_row.drop([TARGET]).to_numpy())
                true_values.append(normalized_row[TARGET])
                continue

            if (current_date - prev_date).seconds != SECONDS_IN_HOUR or len(sequence) == SEQUENCE_LENGTH:
                error_array = model.train(
                    sequence=np.array(sequence),
                    learning_rate=LEARNING_RATE,
                    y_true=np.array(true_values),
                )
                mse_error += error_array
                sequence = []
                true_values = []

            true_values.append(normalized_row[TARGET])
            sequence.append(normalized_row.drop([TARGET]).to_numpy())
            prev_date = current_date
            prev_y_true_normalized = normalized_row[TARGET]

        if sequence:
            error_array = model.train(
                sequence=np.array(sequence),
                learning_rate=LEARNING_RATE,
                y_true=np.array([prev_y_true_normalized]),
            )
            mse_error += error_array

        print(f"Epoch {epoch + 1} is finished. MSE: {mse_error}")


def get_trained_model(normalizer: Normalizer, train: pd.DataFrame, dates: pd.Series) -> RecurrentModel:
    if os.path.isfile(MODEL_PATH):
        print(f'Loading model from "{MODEL_PATH}" ...')
        model: RecurrentModel = load_model()
        print(f"Model is loaded!\n")
        return model

    model = RecurrentModel(
        layers=[
            RNNLayer(input_size=7, hidden_size=5, output_size=1, activation=Tanh()),
        ]
    )

    print("Starting training ...")
    train_model(model=model, normalizer=normalizer, dates=dates, train=train)
    print("Training finished!\n")

    print(f'Dumping model to "{MODEL_PATH}" ...')
    dump_model(model)
    print("Model is dumped!\n")

    return model


def test_model(
        model: RecurrentModel, normalizer: Normalizer, test: pd.DataFrame, dates: pd.Series, train_size: int
) -> None:
    sequence = []
    prev_date: Optional[datetime] = None
    y_pred: list[float] = []

    for index in tqdm(range(test.shape[0])):
        normalized_row: pd.Series = normalizer.normalize(test.iloc[index])
        current_date: datetime = parse_datetime(dates.iloc[index + train_size])

        if prev_date is None:
            prev_date = current_date
            sequence.append(normalized_row.drop([TARGET]).to_numpy())
            continue

        if (current_date - prev_date).seconds != SECONDS_IN_HOUR or len(sequence) == SEQUENCE_LENGTH:
            prediction: np.ndarray = model.predict(np.array(sequence))

            for p in prediction:
                y_pred.append(normalizer.denormalize_feature(p[0], TARGET))

            sequence = []

        sequence.append(normalized_row.drop([TARGET]).to_numpy())
        prev_date = current_date

    if sequence:
        prediction = model.predict(np.array(sequence))
        for p in prediction:
            y_pred.append(normalizer.denormalize_feature(p[0], TARGET))

    # Print some examples of (y_true, y_pred)
    print()
    for j in range(20):
        index = np.random.choice(len(test), size=1)[0]
        print(test[TARGET].to_numpy()[index], y_pred[index])
    print()

    plt.plot(test[TARGET].to_numpy(), "g")
    plt.plot(y_pred, "r")
    plt.show()

    print(f"RMSE: {calculate_rmse(y_true=test[TARGET].to_numpy(), y_pred=np.array(y_pred))}")
    print(f"R2-Score: {calculate_r2_score(y_true=test[TARGET].to_numpy(), y_pred=np.array(y_pred))}")


def main() -> None:
    preprocessed_df: pd.DataFrame = get_preprocessed_dataset()

    dates: pd.Series = preprocessed_df[DATE]
    preprocessed_df.drop([DATE], axis=1, inplace=True)
    normalizer: Normalizer = Normalizer(preprocessed_df)
    train_size: int = int(TRAIN_SIZE * len(preprocessed_df))
    train, test = (
        preprocessed_df[:train_size],
        preprocessed_df[train_size:],
    )

    model: RecurrentModel = get_trained_model(normalizer=normalizer, dates=dates, train=train)

    print("Starting test ...")
    test_model(model=model, normalizer=normalizer, test=test, dates=dates, train_size=train_size)
    print("\nTest is done!")


if __name__ == "__main__":
    main()
