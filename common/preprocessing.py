import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows where pressure equals 0
    result = df[df["Pressure (millibars)"] > 0]

    # Drop useless columns
    result = result.drop("Loud Cover", axis=1)
    result = result.drop("Daily Summary", axis=1)
    result = result.drop("Summary", axis=1)

    # Drop column that cannot be known without knowing real temperature
    result = result.drop("Apparent Temperature (C)", axis=1)

    # Encode recip type
    result = pd.get_dummies(result, columns=["Precip Type"], dtype=float)

    # Drop duplicated rows
    result = result.drop_duplicates()

    # Sort by date
    result = result.sort_values(by="Formatted Date")

    return result
