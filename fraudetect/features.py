from collections.abc import Iterable
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pyod.models.base import BaseDetector
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from functools import partial
from .detectors import get_detector, instantiate_detector
from category_encoders import BinaryEncoder, CountEncoder, HashingEncoder,BaseNEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline



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
    drop_all_na:bool=False
) -> pd.DataFrame:
    """
    Clean the data by dropping NA rows and duplicated rows
    """
    df = df_data.copy()
    df.drop_duplicates(inplace=True)
    if drop_all_na:
        df.dropna(axis=0, how="any", inplace=True)

    return df


def load_cat_encoding(cat_encoding_method:str,**kwargs):
    
    cat_encodings = ['binary','count','hashing','base_n']
    
    if cat_encoding_method not in cat_encodings:
        raise KeyError(f"cat_encoding_method should be in {cat_encodings}")
    
    if cat_encoding_method == 'binary':
        return BinaryEncoder(handle_missing='value', drop_invariant=True, handle_unknown='value')
    
    elif cat_encoding_method == 'count':
        return CountEncoder(handle_missing='value', drop_invariant=True, handle_unknown='value')
    
    elif cat_encoding_method == 'hashing':
        return HashingEncoder(return_df=False, drop_invariant=True,**kwargs)
    
    elif cat_encoding_method == 'base_n':
        return BaseNEncoder(return_df=False,handle_missing='value', drop_invariant=True, handle_unknown='value',**kwargs)


def build_encoder_scalers(cols_onehot:list, 
                   cols_cat_encode:list,
                   cat_encoding_method:str,
                   cols_std:list,
                   cols_robust:list,
                   add_imputer:bool=False,
                   verbose:bool=False,
                   add_concat_features_transform:bool=False,
                   concat_features_encoding_kwargs:dict=None,
                   n_jobs=8,
                   **cat_encoding_kwargs
                   ):
    
    # Imputer
    imputer =  Pipeline(steps=[('imputer', KNNImputer(n_neighbors=5))
                             ]
                        )
    # cat variables
    onehot_encoder = Pipeline(steps=[
                                    ("onehot", OneHotEncoder())
                                ]
                            )
    cat_encoder = Pipeline(steps=[
                                ("cat_encode", load_cat_encoding(cat_encoding_method,**cat_encoding_kwargs))
                            ]
                        )
    # numeric variables
    robust_scaler = Pipeline(steps=[ ("robust", RobustScaler())],
                             )
    std_scaler = Pipeline(steps=[("standard", StandardScaler())],
                        )
    
    # compose encoders and scalers in a column transformer
    transformers=[
        ("cat", cat_encoder, cols_cat_encode),
        ("onehot", onehot_encoder, cols_onehot),
        ("std", std_scaler, cols_std)
    ]
    
    if len(cols_robust)>0:
        transformers.append(("robust", robust_scaler, cols_robust))
        
    if add_imputer:
        _imputer = [('imputer', imputer, cols_cat_encode + cols_onehot + cols_robust)]
        transformers = _imputer + transformers
        
    if add_concat_features_transform:
        _enc = Pipeline(steps=[("concat_features", 
                                load_cat_encoding(**concat_features_encoding_kwargs))
                                ]
                        )
        _transform = ('concat_features', _enc, ['concat_features',])
        transformers.append(_transform)
         
    transformer = ColumnTransformer(transformers, 
                                    remainder="passthrough",
                                    n_jobs=n_jobs,
                                    verbose=verbose
                                    )
    
    return transformer


def perform_feature_engineering(
    transactions_df:pd.DataFrame,
    col_transformer:ColumnTransformer,
    cols_to_drop:list|None,
    windows_size_in_days=[1, 7, 30],
    delay_period_accountid: int = 7,
    mode: str = "train",
    concat_features:Iterable=None,
) -> pd.DataFrame:
    """
    Feature engineering function to be used in the pipeline.
    """
    
    # checks
    assert mode in ["train", "test", "val"], (
        "Error: mode should be either 'train' or 'test' or 'val'"
    )
    assert isinstance(concat_features, Iterable) or (concat_features is None)
    if concat_features is not None :
        concat_features = list(concat_features)
        assert len(concat_features)>=2,"At least to columns should be given"
        for col in concat_features:
            assert col in transactions_df.columns
                
    # clean
    df_data = clean_data(transactions_df)

    # create TX_TIME_DAYS column
    df_data["TX_TIME_DAYS"] = (
        df_data["TX_DATETIME"] - df_data["TX_DATETIME"].min()
    ).dt.days
    # TX_DURING_WEEKEND
    df_data["TX_DURING_WEEKEND"] = (df_data["TX_DATETIME"].dt.dayofweek > 4) * 1
    # TX_DURING_NIGHT
    df_data["TX_DURING_NIGHT"] = (df_data["TX_DATETIME"].dt.hour < 6) * 1 + (df_data["TX_DATETIME"].dt.hour > 18) * 1
    # TX_HOUR
    df_data["TX_HOUR"] = df_data["TX_DATETIME"].dt.hour

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
    
    # concat_features
    if concat_features is not None:
        df_data['concat_features'] = df_data[concat_features].apply(lambda x: "+".join(x),
                                                                    axis=1,
                                                                    raw=False,
                                                                    result_type='reduce')
        df_data = df_data.groupby("concat_features").apply(
            lambda x: get_count_risk_rolling_window(
                x,
                delay_period=delay_period_accountid,
                windows_size_in_days=windows_size_in_days,
                feature="concat_features",
            )
        )
        
    # Labels
    y = df_data["TX_FRAUD"]
    
    # Features
    df_data.drop(columns=["TX_FRAUD"],inplace=True)
    X = df_data
    
    # Drop unneeded_columns
    if cols_to_drop is not None:
        X.drop(columns=cols_to_drop,inplace=True)

    
    
    if mode == "train":        
        X_preprocessed = col_transformer.fit_transform(X)
    else:
        # Transform the test/val data
        X_preprocessed = col_transformer.transform(X)
    
    # columns_of_transformed_data = list(
    #     map(lambda name: name.split('__')[1],
    #         list(col_transformer.get_feature_names_out())
    #     )
    # )
           

    return X_preprocessed, y

#TODO: fits feature selector on df_train
def feature_selector(df_train,)->callable:
     
    pass

def transform_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    col_transformer:ColumnTransformer,
    cols_to_drop:list|None,
    train_transform=None,
    val_transform=None,
    delay_period_accountid: int = 7,
    windows_size_in_days=[1, 7, 30],
    concat_features:list=None
) -> tuple:
    
    
    X_train, y_train = perform_feature_engineering(
        train_df,
        col_transformer=col_transformer,
        cols_to_drop=cols_to_drop,
        mode="train",
        delay_period_accountid=delay_period_accountid,
        windows_size_in_days=windows_size_in_days,
        concat_features=concat_features
    )
    if train_transform is not None:
        X_train, y_train = train_transform(X_train, y_train)

    if val_df is None:
        return (X_train, y_train), col_transformer

    X_val, y_val = perform_feature_engineering(
        val_df,
        col_transformer=col_transformer,
        cols_to_drop=cols_to_drop,
        mode="val",
        delay_period_accountid=delay_period_accountid,
        windows_size_in_days=windows_size_in_days,
        concat_features=concat_features
    )
    if val_transform is not None:
        X_val, y_val = val_transform(X_val, y_val)

    return (X_train, y_train, X_val, y_val), col_transformer


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
    return_fitted_models:bool=False
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
    
    if return_fitted_models:
        return transform_func, model_list
    
    return transform_func
