from collections.abc import Iterable
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pyod.models.base import BaseDetector
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from functools import partial
from sklearn.base import TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from category_encoders import (
    BinaryEncoder,
    CountEncoder,
    HashingEncoder,
    BaseNEncoder,
    CatBoostEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.pipeline import Pipeline, FeatureUnion
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from ..detectors import get_detector, instantiate_detector


def load_cat_encoding(cat_encoding_method: str, **kwargs):
    cat_encodings = ["binary", "count", "hashing", "base_n", "catboost"]

    if cat_encoding_method not in cat_encodings:
        raise KeyError(f"cat_encoding_method should be in {cat_encodings}")

    if cat_encoding_method == "binary":
        return BinaryEncoder(
            handle_missing="value", drop_invariant=False, handle_unknown="value"
        )

    elif cat_encoding_method == "count":
        return CountEncoder(
            handle_missing="value", drop_invariant=False, handle_unknown="value"
        )

    elif cat_encoding_method == "hashing":
        return HashingEncoder(return_df=False, drop_invariant=False, **kwargs)

    elif cat_encoding_method == "base_n":
        return BaseNEncoder(
            return_df=False,
            handle_missing="value",
            drop_invariant=False,
            handle_unknown="value",
            **kwargs,
        )
    elif cat_encoding_method == "catboost":
        return CatBoostEncoder(
            return_df=False,
            handle_missing="value",
            drop_invariant=False,
            handle_unknown="value",
            **kwargs,
        )


def feature_selector(
    X_train:np.ndarray,
    y_train:np.ndarray,
    cv:TimeSeriesSplit=TimeSeriesSplit(n_splits=5),
    estimator=DecisionTreeClassifier(max_depth=15,max_features='sqrt',random_state=41),
    name: str = "rfecv",
    step: float = 0.1,
    scoring: str = "f1",
    n_jobs: int = 4,
    verbose: bool = False,
) -> callable:
    
    if name == "rfecv":
        selector = RFECV(
            estimator=estimator,
            step=step,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
        )
    elif name == "sequential":
        selector = SequentialFeatureSelector(
            estimator=estimator,
            n_features_to_select=int(step * X_train.shape[1]),
            direction="forward",
            n_jobs=n_jobs,
            scoring=scoring,
            cv=cv,
        )
    else:
        raise NotImplementedError

    selector.fit(X=X_train, y=y_train)
    return selector.transform(X=X_train), selector


# ------------ pyod detectors
def fit_outliers_detectors(
    detector_list: list[BaseDetector], X_train: np.ndarray
) -> list[BaseDetector] | FeatureUnion:
    if isinstance(detector_list, list):
        model_list = list()
        for model in tqdm(detector_list, desc="fitting-outliers-det-pyod"):
            model.fit(X_train)
            model_list.append(model)
        return model_list
    else:
        chain_pyod = FeatureUnion(detector_list, n_jobs=4)
        chain_pyod.fit(X=X_train, y=None)
        return chain_pyod


def concat_outliers_scores_pyod(
    fitted_detector_list: list[BaseDetector] | FeatureUnion,
    X: np.ndarray,
):
    if isinstance(fitted_detector_list, list):
        for model in tqdm(fitted_detector_list, desc="concat-outliers-scores-pyod"):
            score = model.decision_function(X)
            score = score.reshape((-1, 1))
        X_t = np.hstack([X] + score)
        return X_t

    else:
        raise NotImplementedError


def load_transforms_pyod(
    X_train: np.ndarray,
    outliers_det_configs: OrderedDict,
    fitted_detector_list: list[BaseDetector] = None,
    return_fitted_models: bool = False,
):
    if fitted_detector_list is not None:
        transform_func = partial(
            concat_outliers_scores_pyod,
            fitted_detector_list=fitted_detector_list,
        )
        if return_fitted_models:
            return transform_func, return_fitted_models
        else:
            return transform_func

    assert isinstance(outliers_det_configs, OrderedDict), (
        f"received {type(outliers_det_configs)}"
    )

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
        concat_outliers_scores_pyod,
        fitted_detector_list=model_list,
    )

    if return_fitted_models:
        return transform_func, model_list

    return transform_func


class FeatureEncoding(TransformerMixin):
    def __init__(
        self,
        cat_encoding_method: str,
        add_imputer: bool = False,
        onehot_threshold: int = 9,
        cols_to_drop: list = None,
        n_jobs: int = 4,
        cat_encoding_kwards: dict = None,
    ):
        self.cols_to_drop = cols_to_drop

        self.add_imputer = add_imputer
        self.imputer = KNNImputer(n_neighbors=5)

        self.cat_encoder = load_cat_encoding(
            cat_encoding_method=cat_encoding_method, **cat_encoding_kwards
        )

        self.onehot_encoder = OneHotEncoder(handle_unknown="ignore")
        self.onehot_threshold = onehot_threshold

        self.scaler = Pipeline(
            steps=[("standard", StandardScaler())],
        )

        self.columns_of_transformed_data = None

        self.col_transformer = None

        self.n_jobs = n_jobs

    def fit(self, X: pd.DataFrame):
        df = X.copy()

        # drop columns
        if self.cols_to_drop is not None:
            df.drop(columns=self.cols_to_drop, inplace=True)

        if "TX_FRAUD" in df.columns:
            # self.y = df["TX_FRAUD"].copy()
            df.drop(columns=["TX_FRAUD"], inplace=True)

        # categorical columns
        cols = df.select_dtypes(include=["object", "string"]).columns
        cols_onehot = [col for col in cols if df[col].nunique() < self.onehot_threshold]
        cols_cat_encode = [
            col for col in cols if df[col].nunique() >= self.onehot_threshold
        ]

        # numeric
        numeric_cols = df.select_dtypes(include=["number"]).columns
        transformers = [
            ("onehot", self.onehot_encoder, cols_onehot),
            ("scaled", self.scaler, numeric_cols),
            ("cat_encode", self.cat_encoder, cols_cat_encode),
        ]

        self.col_transformer = ColumnTransformer(
            transformers, remainder="passthrough", n_jobs=self.n_jobs, verbose=False
        )

        self.col_transformer.fit(df)

        self.columns_of_transformed_data = self.col_transformer.get_feature_names_out()

        return self

    def transform(self, X: pd.DataFrame, y=None):
        _X = X.copy()
        y = None
        if "TX_FRAUD" in _X.columns:
            y = _X["TX_FRAUD"].copy()
            _X.drop(columns=["TX_FRAUD"], inplace=True)

        _X = self.col_transformer.transform(X=_X)

        if self.add_imputer:
            _X = self.imputer.fit_transform(_X)

        return _X, y

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params):
        self.fit(X=X)
        return self.transform(X=X)


class FraudFeatureEngineer(TransformerMixin):
    def __init__(
        self,
        windows_size_in_days: list[int] = [1, 7, 30],
        uid_cols: list = None,
        session_gap_minutes: int = 30,
        n_clusters: int = 8,
    ):
        self.customer_stats = None
        self.windows_size_in_days = windows_size_in_days
        self.uid_cols = None
        self.uid_col_name = "CustomerUID"
        self.behavioral_drift_cols = [
            "AccountId",
        ]
        self.session_gap_minutes = session_gap_minutes
        self.n_clusters = n_clusters
        self.customer_cluster_labels = None
        self.cluster_on_feature = "CUSTOMER_ID"  # str
        self.kmeans = None

    def fit(self, df, y=None):
        self.product_fraud_rate = (
            df.groupby("ProductId")["TX_FRAUD"].mean().rename("ProductFraudRate")
        )
        self.provider_fraud_rate = (
            df.groupby("ProviderId")["TX_FRAUD"].mean().rename("ProviderFraudRate")
        )
        self.channel_fraud_rate = (
            df.groupby("ChannelId")["TX_FRAUD"].mean().rename("ChannelIdFraudRate")
        )

        # if self.n_clusters is not None:
        # self._compute_clusters_customers(df)

        return self

    def transform(self, df, y=None, **fit_params):
        df = df.copy()
        df = df.sort_values(by=["AccountId", "TX_DATETIME"])

        if self.uid_cols is not None:
            df = self._create_unique_identifier(df)

        df = self._add_temporal_features(df)
        df = self._add_account_stats(df)
        df = self._add_customer_stats(df)
        df = self._compute_behavioral_drift(df)
        df = self._compute_batch_gap_features(df)
        df = self._compute_avg_txn_features(df)
        df = self._add_categorical_cross_features(df)
        df = self._add_temporal_identity_interactions(df)
        df = self._add_frequency_features(df)
        df = self._add_fraud_rate_features(df)

        # if self.n_clusters is not None:
        #     df = self._add_customer_clusters(df)

        df = self._cleanup(df)

        return df

    def fit_transform(self, df, y=None):
        self.fit(df)
        return self.transform(df)

    # ---------- Private Helper Methods ----------
    def _create_unique_identifier(self, df: pd.DataFrame):
        df[self.uid_col_name] = df[self.uid_cols].apply(
            lambda x: "+".join(x), axis=1, raw=False, result_type="reduce"
        )

        return df

    def _compute_clusters_customers(self, df) -> None:
        # -- Compute clusters
        cluster_data = (
            df.groupby(self.cluster_on_feature)
            .agg(
                {
                    "TX_AMOUNT": "mean",
                    "ChannelId": lambda x: x.mode().iloc[0]
                    if not x.mode().empty
                    else np.nan,
                }
            )
            .fillna(0)
        )

        # Encode channel as numeric for clustering
        cluster_data["ChannelId"] = (
            cluster_data["ChannelId"].astype("category").cat.codes
        )

        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters, random_state=41, max_iter=500, batch_size=1024
        )
        self.kmeans.fit(cluster_data)

        self.customer_cluster_labels = pd.DataFrame(
            {
                self.cluster_on_feature: cluster_data.index,
                "CustomerCluster": self.kmeans.labels_,
            }
        )

    def _add_temporal_features(self, df):
        df["Hour"] = df["TX_DATETIME"].dt.hour
        df["DayOfWeek"] = df["TX_DATETIME"].dt.dayofweek
        df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
        df["IsNight"] = df["Hour"].between(0, 6).astype(int)

        for col in ["AccountId", "CUSTOMER_ID"]:
            df[col + "_TimeSinceLastTxn"] = (
                df.groupby(col)["TX_DATETIME"].diff().dt.total_seconds().fillna(0) / 60
            )

            df[col + "_Txn1hCount"] = (
                df.sort_values(by=["TX_DATETIME"])
                .set_index("TX_DATETIME")
                .groupby(col)["TRANSACTION_ID"]
                .rolling("1h")
                .count()
                .reset_index(level=0, drop=True)
                .reset_index(level=0, drop=True)
            )

            for day in self.windows_size_in_days:
                df[f"{col}_AvgAmount_{day}day"] = (
                    df.sort_values(by=["TX_DATETIME"])
                    .set_index("TX_DATETIME")
                    .groupby(col)["TX_AMOUNT"]
                    .rolling(window=f"{day}d", min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                    .reset_index(level=0, drop=True)
                )

        return df

    def _add_account_stats(self, df):
        account_stats = (
            df.groupby("AccountId")["TX_AMOUNT"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "AccountMeanAmt", "std": "AccountStdAmt"})
        )
        account_stats["AccountStdAmt"] = (
            account_stats["AccountStdAmt"].fillna(1).replace(0, 1)
        )
        account_stats["AccountMeanAmt"].fillna(0, inplace=True)

        df = df.merge(account_stats, on="AccountId", how="left")
        df["AccountAmountZScore"] = (df["TX_AMOUNT"] - df["AccountMeanAmt"]) / df[
            "AccountStdAmt"
        ]
        df["AccountAmountOverAvg"] = df["TX_AMOUNT"]
        return df

    def _add_customer_stats(self, df):
        customer_stats = (
            df.groupby("CUSTOMER_ID")["TX_AMOUNT"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "CustomerMeanAmt", "std": "CustomerStdAmt"})
        )
        customer_stats["CustomerStdAmt"] = (
            customer_stats["CustomerStdAmt"].fillna(1).replace(0, 1)
        )
        customer_stats["CustomerMeanAmt"].fillna(0, inplace=True)

        df = df.merge(customer_stats, on="CUSTOMER_ID", how="left")
        df["CustomerAmountZScore"] = (df["TX_AMOUNT"] - df["CustomerMeanAmt"]) / df[
            "CustomerStdAmt"
        ]
        df["CustomerAmountOverAvg"] = df["TX_AMOUNT"] / df["CustomerMeanAmt"]
        return df

    def _add_categorical_cross_features(self, df):
        df["Channel_ProductCategory"] = (
            df["ChannelId"].astype(str) + "_" + df["ProductCategory"].astype(str)
        )
        # df['ProductCategory_Account'] = df['ProductCategory'].astype(str) + "_" + df['AccountId'].astype(str)
        # df['ProductCategory_Customer'] = df['ProductCategory'].astype(str) + "_" + df['CUSTOMER_ID'].astype(str)
        df["Country_Currency"] = (
            df["CountryCode"].astype(str) + "_" + df["CurrencyCode"].astype(str)
        )
        df["Channel_PricingStrategy"] = (
            df["ChannelId"].astype(str) + "_" + df["PricingStrategy"].astype(str)
        )
        df["Provider_Product"] = (
            df["ProviderId"].astype(str) + "_" + df["ProductId"].astype(str)
        )
        return df

    def _add_temporal_identity_interactions(self, df):
        df["IsNight_Android"] = df["IsNight"].astype(str) + df["ChannelId"].astype(str)
        df["Weekend_Channel"] = df["IsWeekend"].astype(str) + df["ChannelId"].astype(
            str
        )
        df["Hour_Channel"] = df["Hour"].astype(str) + "_" + df["ChannelId"].astype(str)
        # df['Hour_Account'] = df['Hour'].astype(str) + "_" + df['AccountId'].astype(str)
        # df['Hour_Customer'] = df['Hour'].astype(str) + "_" + df['CUSTOMER_ID'].astype(str)
        # df['DayOfWeek_Account'] = df['DayOfWeek'].astype(str) + "_" + df['AccountId'].astype(str)
        # df['DayOfWeek_Customer'] = df['DayOfWeek'].astype(str) + "_" + df['CUSTOMER_ID'].astype(str)
        df["Country_Hour"] = (
            df["CountryCode"].astype(str) + "_" + df["Hour"].astype(str)
        )
        return df

    def _add_frequency_features(self, df):
        df["TxnDate"] = df["TX_DATETIME"].dt.date
        txn_freq = (
            df.groupby(["AccountId", "TxnDate"])["TRANSACTION_ID"]
            .count()
            .rename("DailyAccountTxnCount")
        )
        df = df.merge(txn_freq, on=["AccountId", "TxnDate"])
        return df

    def _add_fraud_rate_features(self, df):
        df = df.merge(self.product_fraud_rate, on="ProductId", how="left")
        df = df.merge(self.provider_fraud_rate, on="ProviderId", how="left")
        df = df.merge(self.channel_fraud_rate, on="ChannelId", how="left")
        # for pred > handle missing values
        for col in ["ProductFraudRate", "ProviderFraudRate", "ChannelIdFraudRate"]:
            _mean = df[col].mean(skipna=True)
            df[col] = df[col].fillna(_mean)

            pass
        return df

    def _compute_behavioral_drift(self, df):
        df.set_index("TX_DATETIME", inplace=True)

        for col in self.behavioral_drift_cols:
            val_7d = df.groupby(col)["TX_AMOUNT"].transform(
                lambda x: x.rolling("7d").mean()
            )
            val_30d = df.groupby(col)["TX_AMOUNT"].transform(
                lambda x: x.rolling("30d").mean()
            )
            df[col + "_RatioTo7dAvg"] = df["TX_AMOUNT"] / val_7d
            df[col + "_RatioTo30dAvg"] = df["TX_AMOUNT"] / val_30d
            df[col + "_ZScore_7d"] = (df["TX_AMOUNT"] - val_7d) / df.groupby(col)[
                "TX_AMOUNT"
            ].transform(lambda x: x.rolling("7d").std()).replace(0, 1).fillna(1)
            df[col + "_ZScore_30d"] = (df["TX_AMOUNT"] - val_30d) / df.groupby(col)[
                "TX_AMOUNT"
            ].transform(lambda x: x.rolling("30d").std()).replace(0, 1).fillna(1)

        df.reset_index(inplace=True)
        return df

    def _compute_avg_txn_features(self, df):
        for col in self.behavioral_drift_cols:
            df[f"{col}_MovingAvg5"] = (
                df.groupby(col)["TX_AMOUNT"]
                .rolling(window=5, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            long_term_avg = (
                df.groupby(col)["TX_AMOUNT"]
                .agg(["mean"])
                .rename(columns={"mean": f"{col}_LongTermAvg"})
            )
            df = df.merge(long_term_avg, on=col, how="left")
            df["PctChangeFromAvg"] = (
                df[f"{col}_MovingAvg5"] - df[f"{col}_LongTermAvg"]
            ) / df[f"{col}_LongTermAvg"]
        return df

    def _compute_batch_gap_features(self, df):
        df = df.sort_values(by=["BatchId", "TX_DATETIME"])
        batch_time = df.groupby("BatchId")["TX_DATETIME"].min().sort_values()
        batch_time_gap = (
            batch_time.diff().dt.total_seconds().rename("TimeBetweenBatches").fillna(0)
        )
        txn_per_batch = (
            df.groupby("BatchId")["TRANSACTION_ID"].count().rename("TxnPerBatch")
        )
        df = df.merge(txn_per_batch, on="BatchId", how="left")
        df = df.merge(batch_time_gap, left_on="BatchId", right_index=True, how="left")
        return df

    def _compute_session_features(self, df):
        df = df.sort_values(by=["AccountId", "TX_DATETIME"])
        df["TimeDiff"] = (
            df.groupby("AccountId")["TX_DATETIME"].diff().dt.total_seconds().div(60)
        )
        df["NewSession"] = (df["TimeDiff"] > self.session_gap_minutes).fillna(True)
        df["SessionId"] = df.groupby("AccountId")["NewSession"].cumsum()
        session_stats = (
            df.groupby(["AccountId", "SessionId"])
            .agg(
                SessionTxnCount=("TRANSACTION_ID", "count"),
                SessionValue=("TX_AMOUNT", "sum"),
                SessionDuration=(
                    "TX_DATETIME",
                    lambda x: (x.max() - x.min()).total_seconds() / 60,
                ),
                SessionChannel=(
                    "ChannelId",
                    lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
                ),
            )
            .fillna(0)
            .reset_index()
        )
        df = df.merge(session_stats, on=["AccountId", "SessionId"], how="left")
        return df

    def _add_customer_clusters(self, df):
        df = df.merge(
            self.customer_cluster_labels, on=self.cluster_on_feature, how="left"
        )
        df["ClusterChannelInteraction"] = (
            df["CustomerCluster"].astype(str) + "_" + df["ChannelId"].astype(str)
        )
        df["ClusterChannelInteraction"] = (
            df["ClusterChannelInteraction"].astype("category").cat.codes
        )
        return df

    def _cleanup(self, df):
        df.drop(
            columns=[
                "AccountMeanAmt",
                "AccountStdAmt",
                "CustomerMeanAmt",
                "CustomerStdAmt",
                "TxnDate",
            ],
            inplace=True,
        )

        df = df.convert_dtypes()

        df.replace([np.inf, -np.inf], 0, inplace=True)

        return df
