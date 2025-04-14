import datetime
from copy import deepcopy
import pandas as pd
from collections import OrderedDict
from .helpers import get_train_test_set, prequentialSplit
from .features import transform_data
from .config import (
    COLUMNS_TO_DROP,
    COLUMNS_TO_ONE_HOT_ENCODE,
    COLUMNS_TO_CAT_ENCODE,
    COLUMNS_TO_STD_SCALE,
    COLUMNS_TO_ROBUST_SCALE,
    Arguments,
)
from .features import load_transforms_pyod, build_encoder_scalers
from .sampling import data_resampling
from .preprocessing import FraudFeatureEngineer, FeatureEncoding


def load_data(data_path: str = "../data/training.csv") -> pd.DataFrame:
    print("step: load data")

    # load data
    df_data = pd.read_csv(data_path)
    df_data["TransactionStartTime"] = pd.to_datetime(
        df_data["TransactionStartTime"], dayfirst=True
    )
    # renaming columns
    rename_cols = {
        "Amount": "TX_AMOUNT",
        "CustomerId": "CUSTOMER_ID",
        "TransactionStartTime": "TX_DATETIME",
        "TransactionId": "TRANSACTION_ID",
    }
    df_data.rename(columns=rename_cols, inplace=True)
    
    try:
        df_data.rename(columns={"FraudResult": "TX_FRAUD",},inplace=True)    
        df_data["TX_FRAUD"] = df_data["TX_FRAUD"].astype("UInt8")
    except:
        print("There is no column FraudResult in loaded data.")

    # necessary for splitting
    df_data["TX_TIME_DAYS"] = (
        df_data["TX_DATETIME"] - df_data["TX_DATETIME"].min()
    ).dt.days

    # change dtype
    df_data = df_data.convert_dtypes()
    df_data["CountryCode"] = df_data["CountryCode"].astype(str)
    df_data["PricingStrategy"] = df_data["PricingStrategy"].astype(str)
    df_data["TX_TIME_DAYS"] = df_data["TX_TIME_DAYS"].astype("UInt8")

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
    mode:str='train',
    delta_train=40,
    delta_delay=7,
    delta_test=20,
    n_folds=5,
    random_state=41,
    sampling_ratio=1.0,
) -> tuple:
    
    # load data
    df_data = load_data(data_path)

    
    if mode == 'train':

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
            (X_train, y_train), col_transformer = transform_data(
                train_df=train_df, **kwargs_tranform_data
            )
            kwargs_tranform_data['col_transformer']=col_transformer
            (X_val, y_val), col_transformer = transform_data(
                val_df=val_df, **kwargs_tranform_data
            )
            return (X_train, y_train, X_val, y_val), col_transformer

        elif split_method == "prequential":
            (X_train, y_train), col_transformer = transform_data(
                train_df=df_data, **kwargs_tranform_data
            )
            return (X_train, y_train), out, col_transformer
        else:
            raise NotImplementedError
 
    elif mode == 'val':  
        (X_val, y_val), col_transformer = transform_data(
                val_df=df_data, **kwargs_tranform_data
        )  
        return (X_val, y_val), col_transformer
    
    elif mode == 'predict':
        (X_pred, y), col_transformer = transform_data(
                pred_df=df_data, **kwargs_tranform_data
        )  
        return (X_pred, y), col_transformer

    else:
        raise NotImplementedError


class MyDatamodule(object):
    def __init__(self,):
                
        self.encoder = None
        self.feature_engineer = None
                
                
    def setup(self,encoder:FeatureEncoding,feature_engineer:FraudFeatureEngineer):
        self.encoder = deepcopy(encoder)
        self.feature_engineer = deepcopy(feature_engineer)

    def get_train_dataset(self, data_path):
        
        raw_data = load_data(data_path)
        df_augmented = self.feature_engineer.fit_transform(raw_data) 
        X,y = self.encoder.fit_transform(X=df_augmented)
                  
        return X, y
    
    def get_predict_dataset(self, data_path:str):
        
        raw_data = load_data(data_path)
        df_augmented = self.feature_engineer.transform(raw_data) 
        X,y= self.encoder.transform(X=df_augmented)        
        
        return X, y

    def _resample_data(
        self,
        X,
        y,
        sampler_names: list[str],
        sampler_cfgs: list[dict],
    ):
        assert len(sampler_names) == len(sampler_cfgs), (
            "They should have the same length."
        )

        # Re-sample data
        X, y = data_resampling(
            X=X, y=y, sampler_names=sampler_names, sampler_cfgs=sampler_cfgs
        )
        print("Resampled data shape: ", X.shape, y.shape)

        return X, y

    def _concat_pyod_scores(
        self, X, outliers_det_configs: OrderedDict, fitted_detector_list: list = None
    ):
        # load pyod transform and apply it to X
        transform_pyod, fitted_models_pyod = load_transforms_pyod(
            X_train=X,
            outliers_det_configs=outliers_det_configs,
            fitted_detector_list=fitted_detector_list,
            return_fitted_models=True,
        )

        return transform_pyod(X=X), fitted_models_pyod

    # functions to augment data
    def augment_resample_dataset(
        self,
        X,
        y,
        outliers_det_configs: OrderedDict | None,
        sampler_names: list[str] | None,
        sampler_cfgs: list[dict] | None,
        fitted_detector_list: list = None,
    ):
        
        # augment data using outliers scores
        fitted_models_pyod = None
        if outliers_det_configs is not None:
            X, fitted_models_pyod = self._concat_pyod_scores(
                X,
                outliers_det_configs=outliers_det_configs,
                fitted_detector_list=fitted_detector_list,
            )

        # resample data
        if (sampler_names is not None) and (sampler_cfgs is not None):
            X, y = data_resampling(
                X=X, y=y, sampler_names=sampler_names, sampler_cfgs=sampler_cfgs
            )
            print("Resampled data shape: ", X.shape, y.shape)

        return (X, y), fitted_models_pyod






