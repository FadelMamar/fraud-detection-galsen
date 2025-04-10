from .helpers import get_train_test_set, prequentialSplit
from .features import transform_data
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(data_path: str = "../data/training.csv") -> pd.DataFrame:
    print("step: load data")

    # load data
    df_data = pd.read_csv(data_path)
    df_data["TransactionStartTime"] = pd.to_datetime(
        df_data["TransactionStartTime"], dayfirst=True
    )
    # renaming columns
    rename_cols = {
        "FraudResult": "TX_FRAUD",
        "Amount": "TX_AMOUNT",
        "CustomerId": "CUSTOMER_ID",
        "TransactionStartTime": "TX_DATETIME",
        "TransactionId": "TRANSACTION_ID",
    }
    df_data.rename(columns=rename_cols, inplace=True)

    # necessary for splitting
    df_data["TX_TIME_DAYS"] = (
        df_data["TX_DATETIME"] - df_data["TX_DATETIME"].min()
    ).dt.days

    return df_data


def train_test_split(
    df_data: pd.DataFrame,
    delta_train: int = 40,
    delta_delay: int = 7,
    delta_test: int = 20,
    method: str = "hold-out",
    random_state: int = 41,
    n_folds: int = 5,
    sampling_ratio: float = 1.0,
) -> tuple:
    print(f"step: train-test-split using method={method}")

    df_data.sort_values("TX_DATETIME", inplace=True, ascending=True)
    start_date_training = df_data["TX_DATETIME"].iloc[-1]  # last date of the dataset
    start_date_training_with_valid = start_date_training + datetime.timedelta(
        days=-(delta_delay + delta_test + delta_train)
    )

    if method == "hold-out":
        train_df, test_df = get_train_test_set(
            df_data,
            start_date_training=start_date_training_with_valid,
            delta_train=delta_train,
            delta_test=delta_test,
            delta_delay=delta_delay,
            sampling_ratio=sampling_ratio,
            random_state=random_state,
        )

        return train_df, test_df

    elif method == "prequential":
        prequential_split_indices = prequentialSplit(
            df_data,
            start_date_training_with_valid,
            n_folds=n_folds,
            delta_train=delta_train,
            delta_delay=delta_delay,
            delta_assessment=delta_test,
        )
        return prequential_split_indices

    else:
        raise ValueError("Invalid method. Choose either 'hold-out' or 'prequential'.")


def data_loader(
    kwargs_tranform_data: dict,
    data_path: str = "../data/training.csv",
    split_method: str = "hold-out",
    delta_train=40,
    delta_delay=7,
    delta_test=20,
    n_folds=5,
    random_state=41,
    sampling_ratio=1.0,
):
    # load data
    df_data = load_data(data_path)

    # split data
    out = train_test_split(
        df_data,
        delta_train=delta_train,
        delta_delay=delta_delay,
        delta_test=delta_test,
        method=split_method,
        random_state=random_state,
        n_folds=n_folds,
        sampling_ratio=sampling_ratio,
    )

    if split_method == "hold-out":
        train_df, val_df = out
        (X_train, y_train, X_val, y_val), _ = transform_data(
            train_df=train_df, val_df=val_df, **kwargs_tranform_data
        )
        return X_train, y_train, X_val, y_val

    elif split_method == "prequential":
        (X_train, y_train), _ = transform_data(
            train_df=df_data, val_df=None, **kwargs_tranform_data
        )
        return (X_train, y_train), out
