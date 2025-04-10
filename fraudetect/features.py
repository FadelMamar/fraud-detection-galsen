from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pyod.models.base import BaseDetector
import pandas as pd
import numpy as np
from .sampling import get_sampler, build_samplers_pipeline
from tqdm import tqdm
from collections import OrderedDict
from functools import partial
from .detectors import get_detector, instantiate_detector


def get_customer_spending_behaviour_features(
    customer_transactions, windows_size_in_days=[1, 7, 30]
):
    # Let us first order transactions chronologically
    customer_transactions = customer_transactions.sort_values("TX_DATETIME")

    # The transaction date and time is set as the index, which will allow the use of the rolling function
    customer_transactions.index = customer_transactions.TX_DATETIME

    # For each window size
    for window_size in windows_size_in_days:
        # Compute the sum of the transaction amounts and the number of transactions for the given window size
        SUM_AMOUNT_TX_WINDOW = (
            customer_transactions["TX_AMOUNT"].rolling(str(window_size) + "d").sum()
        )
        NB_TX_WINDOW = (
            customer_transactions["TX_AMOUNT"].rolling(str(window_size) + "d").count()
        )

        # Compute the average transaction amount for the given window size
        # NB_TX_WINDOW is always >0 since current transaction is always included
        AVG_AMOUNT_TX_WINDOW = SUM_AMOUNT_TX_WINDOW / NB_TX_WINDOW

        # Save feature values
        customer_transactions[
            "CUSTOMER_ID_NB_TX_" + str(window_size) + "DAY_WINDOW"
        ] = list(NB_TX_WINDOW)
        customer_transactions[
            "CUSTOMER_ID_AVG_AMOUNT_" + str(window_size) + "DAY_WINDOW"
        ] = list(AVG_AMOUNT_TX_WINDOW)

    # Reindex according to transaction IDs
    customer_transactions.index = customer_transactions.TRANSACTION_ID

    # And return the dataframe with the new features
    return customer_transactions


def get_count_risk_rolling_window(
    terminal_transactions,
    delay_period=7,
    windows_size_in_days=[1, 7, 30],
    feature="AccountId",
):
    terminal_transactions = terminal_transactions.sort_values("TX_DATETIME")

    terminal_transactions.index = terminal_transactions.TX_DATETIME

    NB_FRAUD_DELAY = (
        terminal_transactions["TX_FRAUD"].rolling(str(delay_period) + "d").sum()
    )
    NB_TX_DELAY = (
        terminal_transactions["TX_FRAUD"].rolling(str(delay_period) + "d").count()
    )

    for window_size in windows_size_in_days:
        NB_FRAUD_DELAY_WINDOW = (
            terminal_transactions["TX_FRAUD"]
            .rolling(str(delay_period + window_size) + "d")
            .sum()
        )
        NB_TX_DELAY_WINDOW = (
            terminal_transactions["TX_FRAUD"]
            .rolling(str(delay_period + window_size) + "d")
            .count()
        )

        NB_FRAUD_WINDOW = NB_FRAUD_DELAY_WINDOW - NB_FRAUD_DELAY
        NB_TX_WINDOW = NB_TX_DELAY_WINDOW - NB_TX_DELAY

        RISK_WINDOW = NB_FRAUD_WINDOW / NB_TX_WINDOW

        terminal_transactions[feature + "_NB_TX_" + str(window_size) + "DAY_WINDOW"] = (
            list(NB_TX_WINDOW)
        )
        terminal_transactions[feature + "_RISK_" + str(window_size) + "DAY_WINDOW"] = (
            list(RISK_WINDOW)
        )

    terminal_transactions.index = terminal_transactions.TRANSACTION_ID

    # Replace NA values with 0 (all undefined risk scores where NB_TX_WINDOW is 0)
    terminal_transactions.fillna(0, inplace=True)

    return terminal_transactions


def clean_data(
    df_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean the data by dropping NA rows and duplicated rows
    """
    df = df_data.copy()
    df.drop_duplicates(inplace=True)
    df.dropna(axis=0, how="any", inplace=True)

    return df


def perform_feature_engineering(
    transactions_df,
    columns_to_drop: list,
    columns_to_onehot_encode: list,
    onehot_encoder: OneHotEncoder,
    scaler: StandardScaler,
    windows_size_in_days=[1, 7, 30],
    delay_period_accountid: int = 7,
    columns_to_scale: list = None,
    mode: str = "train",
) -> pd.DataFrame:
    """
    Feature engineering function to be used in the pipeline.
    """

    assert mode in ["train", "test", "val"], (
        "Error: mode should be either 'train' or 'test' or 'val'"
    )

    df_data = clean_data(transactions_df)

    # create TX_TIME_DAYS column
    df_data["TX_TIME_DAYS"] = (
        df_data["TX_DATETIME"] - df_data["TX_DATETIME"].min()
    ).dt.days
    # TX_DURING_WEEKEND
    df_data["TX_DURING_WEEKEND"] = (df_data["TX_DATETIME"].dt.dayofweek > 4) * 1
    # TX_DURING_NIGHT
    df_data["TX_DURING_NIGHT"] = (df_data["TX_DATETIME"].dt.hour < 6) * 1

    # Customer ID transformation
    df_data = df_data.groupby("CUSTOMER_ID").apply(
        lambda x: get_customer_spending_behaviour_features(
            x, windows_size_in_days=windows_size_in_days
        )
    )
    df_data = df_data.sort_values("TX_DATETIME").reset_index(drop=True)

    # Account ID transformation
    df_data = df_data.groupby("AccountId").apply(
        lambda x: get_count_risk_rolling_window(
            x,
            delay_period=delay_period_accountid,
            windows_size_in_days=windows_size_in_days,
            feature="AccountId",
        )
    )
    df_data = df_data.sort_values("TX_DATETIME").reset_index(drop=True)

    # Features
    X = df_data.drop(columns=["TX_FRAUD"])

    # Labels
    y = df_data["TX_FRAUD"]

    # get features for scaling and encoding
    X_one_hot = X[columns_to_onehot_encode].astype(str)

    if columns_to_scale is None:
        X_scale = X.drop(columns=columns_to_onehot_encode + columns_to_drop).astype(
            float
        )
        columns_to_scale = []
    else:
        X_scale = X[columns_to_scale].astype(float)

    # drop the columns that are not needed anymore
    X.drop(
        columns=columns_to_onehot_encode + columns_to_scale + columns_to_drop,
        inplace=True,
    )

    if mode == "train":
        # Fit on the training data
        X_one_hot = onehot_encoder.fit_transform(X_one_hot)
        X_scaled = scaler.fit_transform(X_scale)
        assert len(onehot_encoder.categories_) == len(columns_to_onehot_encode), (
            "Error: Number of categories in one-hot encoder does not match the number of columns to one-hot encode. Set drop=None in the encoder."
        )
        assert (
            df_data.nunique().loc[columns_to_onehot_encode].sum() == X_one_hot.shape[1]
        ), (
            "Error: Number of unique values in one-hot encoded columns does not match the number of columns to one-hot encode."
        )

    else:
        # Transform the test/val data
        X_one_hot = onehot_encoder.transform(X_one_hot)
        X_scaled = scaler.transform(X_scale)

    # Concatenate the one-hot encoded features with the scaled features
    if X.empty:
        X_preprocessed = np.hstack([X_one_hot, X_scaled])
    else:
        X_preprocessed = np.hstack([X_one_hot, X_scaled, X.to_numpy()])

    # check total numer of features

    return X_preprocessed, y


def transform_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    columns_to_drop: str,
    columns_to_onehot_encode: str,
    columns_to_scale: str,
    train_transform=None,
    val_transform=None,
    delay_period_accountid: int = 7,
    windows_size_in_days=[1, 7, 30],
) -> tuple:
    # scaler and encoders
    onehot_encoder = OneHotEncoder(
        sparse_output=False, handle_unknown="ignore", drop=None, dtype=np.float64
    )
    scaler = StandardScaler()

    X_train, y_train = perform_feature_engineering(
        train_df,
        columns_to_drop=columns_to_drop,
        columns_to_onehot_encode=columns_to_onehot_encode,
        columns_to_scale=columns_to_scale,
        onehot_encoder=onehot_encoder,
        scaler=scaler,
        mode="train",
        delay_period_accountid=delay_period_accountid,
        windows_size_in_days=windows_size_in_days,
    )
    if train_transform is not None:
        X_train, y_train = train_transform(X_train, y_train)

    if val_df is None:
        return (X_train, y_train), (onehot_encoder, scaler)

    X_val, y_val = perform_feature_engineering(
        val_df,
        columns_to_drop=columns_to_drop,
        columns_to_onehot_encode=columns_to_onehot_encode,
        columns_to_scale=columns_to_scale,
        onehot_encoder=onehot_encoder,
        scaler=scaler,
        mode="val",
        delay_period_accountid=delay_period_accountid,
        windows_size_in_days=windows_size_in_days,
    )
    if val_transform is not None:
        X_val, y_val = val_transform(X_val, y_val)

    return (X_train, y_train, X_val, y_val), (onehot_encoder, scaler)


def fit_outliers_detectors(
    detector_list: list[BaseDetector], X_train: np.ndarray
) -> list[BaseDetector]:
    model_list_ = list()

    for model in tqdm(detector_list, desc="fitting-outliers-det-pyod"):
        model.fit(X_train)
        model_list_.append(model)

    return model_list_


def concat_outliers_probs_pyod(
    fitted_detector_list: list[BaseDetector],
    X: np.ndarray,
    method="unify",
    add_confidence: bool = False,
):
    probs = []

    for model in tqdm(fitted_detector_list, desc="concat-outliers-probs-pyod"):
        score = model.predict_proba(X, method=method, return_confidence=add_confidence)
        if add_confidence:
            prob, cnf = score
            probs.append(cnf.reshape((-1, 1)))
        else:
            prob = score

        probs.append(
            prob[:, 1].reshape((-1, 1))
        )  # prob is shape[m,2] (prob normal, prob outlier)

    X_t = np.hstack([X] + probs)

    return X_t


def load_transforms_pyod(
    X_train: np.ndarray,
    outliers_det_configs: OrderedDict,
    method: str = "unify",
    add_confidence: bool = False,
    fitted_detector_list: list[BaseDetector] = None,
):
    if fitted_detector_list is not None:
        return partial(
            concat_outliers_probs_pyod,
            fitted_detector_list=fitted_detector_list,
            method=method,
            add_confidence=add_confidence,
        )

    assert isinstance(outliers_det_configs, OrderedDict)

    model_list = list()

    # instantiate detectors
    names = outliers_det_configs.keys()
    for name in names:
        detector, cfg = get_detector(name=name, config=outliers_det_configs)
        detector = instantiate_detector(detector, cfg)
        model_list.append(detector)

    # fit detectors
    model_list = fit_outliers_detectors(model_list, X_train)

    # transform func
    transform_func = partial(
        concat_outliers_probs_pyod,
        fitted_detector_list=model_list,
        method=method,
        add_confidence=add_confidence,
    )

    return transform_func
