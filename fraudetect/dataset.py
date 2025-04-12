import datetime
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

    # change dtype
    df_data = df_data.convert_dtypes()
    df_data["TX_FRAUD"] = df_data["TX_FRAUD"].astype("UInt8")
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
    delta_train=40,
    delta_delay=7,
    delta_test=20,
    n_folds=5,
    random_state=41,
    sampling_ratio=1.0,
) -> tuple:
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
        (X_train, y_train, X_val, y_val), col_transformer = transform_data(
            train_df=train_df, val_df=val_df, **kwargs_tranform_data
        )
        return X_train, y_train, X_val, y_val

    elif split_method == "prequential":
        (X_train, y_train), col_transformer = transform_data(
            train_df=df_data, val_df=None, **kwargs_tranform_data
        )
        return (X_train, y_train), out, col_transformer


class MyDataset(object):
    def __init__(self, 
                 args: Arguments,
                 cols_to_drop:list[str]=None,
                 cols_onehot:list[str]=None,
                 cols_cat_encode:list[str]=None,
                 cols_std_scale:list[str]=None,
                 cols_robust_scale:list[str]=None
                 ):
        self.args = args
        self.col_transformer=None
        self.fitted_models_pyod=None
        self.cols_to_drop=cols_to_drop
        self.cols_onehot=cols_onehot,
        self.cols_cat_encode=cols_cat_encode,
        self.cols_std=cols_std_scale,
        self.cols_robust=cols_robust_scale,

    def __prepare_data(
        self,
        data_path: str,
        kwargs_tranform_data: dict,
        delta_train=40,
        delta_delay=7,
        delta_test=20,
        random_state=41,
    ):
        # load and transform
        (X_train, y_train), prequential_split_indices, self.col_transformer = data_loader(
            kwargs_tranform_data=kwargs_tranform_data,
            data_path=data_path,
            split_method="prequential",
            delta_train=delta_train,
            delta_delay=delta_delay,
            delta_test=delta_test,
            n_folds=1,  # matters if prequential_split_indices are used
            random_state=random_state,
            sampling_ratio=1.0,
        )
        print("Raw data shape: ", X_train.shape, y_train.shape)

        return X_train, y_train

    def build_dataset(self, verbose=0):
        args = self.args

        if args.cat_encoding_method == "hashing":
            kwargs = dict(
                n_components=args.cat_encoding_hash_n_components,
                hash_method=args.cat_encoding_hash_method,
            )
        elif args.cat_encoding_method == "base_n":
            kwargs = dict(
                base=args.cat_encoding_base_n,
            )
        else:
            kwargs = dict()

        # load data & do preprocessing
        self.col_transformer = build_encoder_scalers(
            cols_onehot=self.cols_onehot,
            cols_cat_encode=self.cols_cat_encode,
            cols_std=self.cols_std,
            cols_robust=self.cols_robust,
            cat_encoding_method=args.cat_encoding_method,
            add_imputer=args.add_imputer,
            verbose=bool(verbose),
            add_concat_features_transform=args.concat_features is not None,
            n_jobs=args.n_jobs,
            concat_features_encoding_kwargs=args.concat_features_encoding_kwargs,
            **kwargs,
        )
        kwargs_tranform_data = dict(
            col_transformer=self.col_transformer,
            cols_to_drop=self.cols_to_drop,
            windows_size_in_days=args.windows_size_in_days,
            train_transform=None,  # some custom transform applied to X_train,y_train
            val_transform=None,  # some custom transform applied to X_val,y_val
            delay_period_accountid=args.delta_delay,
            concat_features=args.concat_features,
        )
        X_train, y_train = self.__prepare_data(
            data_path=args.data_path,
            kwargs_tranform_data=kwargs_tranform_data,
            delta_train=args.delta_train,
            delta_delay=args.delta_delay,
            delta_test=args.delta_test,
            random_state=args.random_state,
        )

        # transformed column names
        # columns_of_transformed_data = list(map(lambda name: name.split('__')[1],
        #                                         list(col_transformer.get_feature_names_out()))
        #                                     )
        columns_of_transformed_data = self.col_transformer.get_feature_names_out()
        df_train_preprocessed = pd.DataFrame(
            X_train, columns=columns_of_transformed_data
        )

        print("X_train_preprocessed columns: ", df_train_preprocessed.columns)

        return X_train, y_train

    def __resample_data(
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

    def __concat_pyod_scores(
        self, X, outliers_det_configs: OrderedDict, fitted_detector_list: list = None
    ):
        # load pyod transform and apply it to X
        transform_pyod, self.fitted_models_pyod = load_transforms_pyod(
            X_train=X,
            outliers_det_configs=outliers_det_configs,
            fitted_detector_list=fitted_detector_list,
            return_fitted_models=True,
        )

        return transform_pyod(X=X)

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
        if outliers_det_configs is not None:
            X = self.__concat_pyod_scores(
                X,
                outliers_det_configs=outliers_det_configs,
                fitted_detector_list=fitted_detector_list,
            )

        # resample data
        if (sampler_names is not None) and (sampler_cfgs is not None):
            X, y = self.__resample_data(
                X=X, y=y, sampler_names=sampler_names, sampler_cfgs=sampler_cfgs
            )

        return (X, y)
