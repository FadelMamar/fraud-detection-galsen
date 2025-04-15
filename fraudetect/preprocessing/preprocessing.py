from collections.abc import Iterable
from pyod.models.base import BaseDetector
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Sequence
from numbers import Integral
from collections import OrderedDict
from functools import partial
import sklearn
from sklearn.base import TransformerMixin, BaseEstimator, _fit_context
from sklearn.utils.validation import validate_data
from sklearn.cluster import MiniBatchKMeans
from category_encoders import (
    BinaryEncoder,
    CountEncoder,
    HashingEncoder,
    BaseNEncoder,
    CatBoostEncoder,
)
from sklearn.utils._param_validation import Interval
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import RFECV, SequentialFeatureSelector,SelectKBest,f_classif
from sklearn.pipeline import Pipeline, FeatureUnion
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from ..detectors import get_detector, instantiate_detector

sklearn.set_config(enable_metadata_routing=True)

def load_cat_encoding(cat_encoding_method: str, 
                      cols=None, 
                      hash_n_components=7, 
                      handle_missing="value", 
                      return_df=True,
                      hash_method='md5',
                    drop_invariant=False,
                       handle_unknown="value", 
                       base:int=4):
    cat_encodings = ["binary", "count", "hashing", "base_n", "catboost"]

    if cat_encoding_method not in cat_encodings:
        raise KeyError(f"cat_encoding_method should be in {cat_encodings}")

    if cat_encoding_method == "binary":
        return BinaryEncoder(
            handle_missing=handle_missing,cols=cols,
              drop_invariant=drop_invariant, 
              handle_unknown=handle_unknown
        )

    elif cat_encoding_method == "count":
        return CountEncoder(
            handle_missing=handle_missing,
            cols=cols, 
            drop_invariant=drop_invariant, 
            handle_unknown=handle_unknown
        )

    elif cat_encoding_method == "hashing":
        return HashingEncoder(n_components=hash_n_components,
                              hash_method=hash_method,
                              cols=cols,
                              return_df=return_df, 
                              drop_invariant=drop_invariant)

    elif cat_encoding_method == "base_n":
        return BaseNEncoder(
            return_df=return_df,
            cols=cols,
            handle_missing=handle_missing,
            drop_invariant=drop_invariant,
            handle_unknown=handle_unknown,
            base=base
        )
    elif cat_encoding_method == "catboost":
        return CatBoostEncoder(
            return_df=return_df,
            cols=cols,
            handle_missing=handle_missing,
            drop_invariant=drop_invariant,
            handle_unknown=handle_unknown,
        )


def load_cols_transformer(df:pd.DataFrame,
                          onehot_threshold=9,
                          n_jobs=1,
                          scaler = StandardScaler(),
                          onehot_encoder = OneHotEncoder(handle_unknown='ignore'),
                          cat_encoding_method='binary',
                          **cat_encoding_kwargs):


    # categorical columns
    cols = df.select_dtypes(include=["object", "string"]).columns
    cols_onehot = [col for col in cols if df[col].nunique() < onehot_threshold]
    cols_cat_encode = [
        col for col in cols if df[col].nunique() >= onehot_threshold
    ]
    numeric_cols = df.select_dtypes(include=["number"]).columns

    scaler =  Pipeline(
            steps=[("scaler", scaler)],
        )
    cat_encoder = load_cat_encoding(
            cat_encoding_method=cat_encoding_method, **cat_encoding_kwargs
        )
        
    transformers = [
        ("onehot", onehot_encoder, cols_onehot),
        ("scaled", scaler, numeric_cols),
        ("cat_encode", cat_encoder, cols_cat_encode),
    ]
    col_transformer = ColumnTransformer(
        transformers, remainder="passthrough", n_jobs=n_jobs, verbose=False
    )

    return col_transformer 


def load_feature_selector(
    n_features_to_select=10,
    cv:TimeSeriesSplit=TimeSeriesSplit(n_splits=5,gap=5000),
    estimator=DecisionTreeClassifier(max_depth=15,max_features='sqrt',random_state=41),
    name: str = "selectkbest",
    rfe_step:int=3,
    top_k_best:int=10,
    scoring: str = "f1",
    n_jobs: int = 4,
    verbose: bool = False,
) -> Pipeline:
    
    if name == "rfecv":
        selector = RFECV(
            estimator=estimator,
            step=rfe_step,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
        )
    elif name == "sequential":
        selector = SequentialFeatureSelector(
            estimator=estimator,
            n_features_to_select=n_features_to_select,
            direction="forward",
            n_jobs=n_jobs,
            tol=1e-4,
            scoring=scoring,
            cv=cv,
        )
    
    elif name == "selectkbest":
        selector = SelectKBest(score_func=f_classif,
                               k=top_k_best
                               )
    else:
        raise NotImplementedError

    return Pipeline(steps=[('feature_selector',selector)])


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


class ColumnDropper(TransformerMixin, BaseEstimator):

    _parameter_constraints = {
        "cols_to_drop": [
            "array-like",
            Interval(Integral, 1, None, closed="left"),
        ]
    }

    def __init__(self,cols_to_drop):
        
        self.cols_to_drop = cols_to_drop

    def fit_transform(self, X, y = None, **fit_params):
        self.fit(X,y,**fit_params)
        return self.transform(X=X,y=y)
            
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self,X:pd.DataFrame,y=None,**fit_params):
        
        assert isinstance(X, pd.DataFrame), "provide a DataFrame"
        
        self.feature_names_in_ = [col for col in X.columns if col != 'TX_FRAUD']
        self.n_features_in_ = len(self.feature_names_in_)
        self.is_fitted_ = True
        
        return self

    def transform(self,X:pd.DataFrame,y=None):

        assert isinstance(X, pd.DataFrame), "provide a DataFrame"

        df = X.copy()

        if self.cols_to_drop is not None:
            df.drop(columns=self.cols_to_drop, inplace=True)

        if "TX_FRAUD" in df.columns:
            # y = df["TX_FRAUD"].copy()
            df = df.drop(columns=["TX_FRAUD"])

        return df


class AdvancedFeatures(TransformerMixin, BaseEstimator):

    _parameter_constraints = {
        "feature_selector_name":[str],
        "estimator":[BaseEstimator],
        "n_splits":[int],
        "cv_gap":[int],
        "n_features_to_select":[int],
        "scoring":[str],
        "n_jobs":[int],    
        "rfe_step":[int],
        "top_k_best":[int],         
        "verbose":[bool]
    }

    def __init__(self,
                 feature_selector_name:str='selectkbest',
                 estimator=DecisionTreeClassifier(),
                 n_splits=5,
                 cv_gap=5000,
                 n_features_to_select=3,
                 rfe_step=3,
                 top_k_best=10,
                 scoring='f1',
                 n_jobs=1,             
                 verbose=False
                 ):
        
        self.n_features_to_select=n_features_to_select
        self.rfe_step=rfe_step
        self.top_k_best=top_k_best
        self.feature_selector_name = feature_selector_name
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.n_splits = n_splits
        self.cv_gap = cv_gap
        self.estimator=estimator
        self.verbose=verbose
        
    def fit_transform(self, X, y = None, **fit_params):
        self.fit(X=X,y=y,**fit_params)
        return self.transform(X=X,y=y)
            
    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self,X,y=None,**fit_params):
                
        validate_data(self,X=X,y=y)
        
        cv = TimeSeriesSplit(n_splits=self.n_splits,gap=self.cv_gap)
        self._selector = load_feature_selector(n_features_to_select=self.n_features_to_select,
                                               cv=cv,
                                               estimator=self.estimator,
                                               name = self.feature_selector_name,
                                               top_k_best=self.top_k_best,
                                               rfe_step=self.rfe_step,
                                               scoring = self.scoring,
                                               n_jobs = self.n_jobs,
                                               verbose = self.verbose,
                                               )
        self._selector.fit(X=X,y=y)
        
        self.is_fitted_ = True
        
        return self

    def transform(self,X,y=None):
        
        validate_data(self,X=X,reset=False)
        
        return self._selector.transform(X=X)



class FeatureEncoding(TransformerMixin, BaseEstimator):

    _parameter_constraints = {
        # "cols_to_drop": [
        #     "array-like",
        #     Interval(Integral, 1, None, closed="left"),
        # ],
        "add_imputer":[bool],
        "imputer_n_neighbors":[int],
        "onehot_threshold":[int],
        "cat_encoding_method":[str],
        "n_jobs":[int]
    }

    def __init__(
        self,
        cat_encoding_method: str='binary' ,
        add_imputer: bool = False,
        imputer_n_neighbors:int=5,
        onehot_threshold: int = 9,
        # cols_to_drop: list = None,
        n_jobs: int = 1,
    ):
        # self.cols_to_drop = cols_to_drop

        self.add_imputer = add_imputer
        self.imputer_n_neighbors = imputer_n_neighbors
        # self.imputer = imputer

        self.cat_encoding_method = cat_encoding_method

        # self.onehot_encoder = onehot_encoder
        self.onehot_threshold = onehot_threshold

        self.n_jobs = n_jobs

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pd.DataFrame, y=None,**fit_params):

        assert isinstance(X, pd.DataFrame), "provide a DataFrame"

        self._col_transformer = load_cols_transformer(df=X,
                                                    onehot_threshold=9,
                                                    n_jobs=self.n_jobs,
                                                    scaler = StandardScaler(),
                                                    onehot_encoder = OneHotEncoder(handle_unknown='ignore'),
                                                    cat_encoding_method=self.cat_encoding_method,
                                                    **fit_params)
        self._imputer = None
        if self.add_imputer:
            self._imputer = KNNImputer(n_neighbors=self.imputer_n_neighbors,
                                       missing_values=np.nan,
                                       add_indicator=False,
                                       weights='distance'
                                       )
        
        
        self._col_transformer.fit(X=X,y=y)

        self.is_fitted_ = True

        self._columns_of_transformed_data = self._col_transformer.get_feature_names_out()

        return self

    def transform(self, X:pd.DataFrame, y=None):
        
        assert isinstance(X,pd.DataFrame), "Give a pandas DataFrame."
        
        X = self._col_transformer.transform(X=X)
        
        if self._imputer is not None:
            X = self._imputer.fit_transform(X=X,y=y)
        
        return X

    def fit_transform(self, X, y = None, **fit_params):
        self.fit(X=X,y=y,**fit_params)
        return self.transform(X=X,y=y)


class FraudFeatureEngineer(TransformerMixin, BaseEstimator):
    
    _parameter_constraints = {
        "windows_size_in_days": [
            "array-like",
            Interval(Integral, 1, None, closed="left"),
        ],
        "uid_cols": [
            "array-like",
            Interval(Integral, 1, None, closed="left"),
        ],
        "behavioral_drift_cols": [
            "array-like",
            Interval(Integral, 1, None, closed="left"),
        ],
        "add_imputer":[bool],
        "session_gap_minutes":[int],
        "n_clusters":[int],
        "cat_encoding_method":[str],
        "uid_col_name":[str],
        "cluster_on_feature":[str]
    }
    
    def __init__(
        self,
        windows_size_in_days: list[int] = [1, 7, 30],
        uid_cols: list[str] = [None,],
        session_gap_minutes: int = 30,
        n_clusters: int = 8,
        uid_col_name:str="CustomerUID",
        cluster_on_feature:str="CUSTOMER_ID",
        behavioral_drift_cols:list[str]=["AccountId",]
    ):
        
        self.windows_size_in_days = windows_size_in_days
        self.uid_cols = uid_cols
        self.uid_col_name = uid_col_name
        self.behavioral_drift_cols = behavioral_drift_cols
        self.session_gap_minutes = session_gap_minutes
        self.n_clusters = n_clusters
        self.cluster_on_feature = cluster_on_feature
    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X:pd.DataFrame, y=None, **fit_params):
        
        self._customer_cluster_labels = None
        
        self._product_fraud_rate = (
            X.groupby("ProductId")["TX_FRAUD"].mean().rename("ProductFraudRate")
        )
        self._provider_fraud_rate = (
            X.groupby("ProviderId")["TX_FRAUD"].mean().rename("ProviderFraudRate")
        )
        self._channel_fraud_rate = (
            X.groupby("ChannelId")["TX_FRAUD"].mean().rename("ChannelIdFraudRate")
        )

        # if self.n_clusters is not None:
        # self._compute_clusters_customers(df)

        return self

    def transform(self, X:pd.DataFrame, y=None):
        
        assert isinstance(X, pd.DataFrame), "Please provide a DataFrame"
        
        df = X.copy()
        df = df.sort_values(by=["AccountId", "TX_DATETIME"])

        if all([col is not None for col in self.uid_cols]):
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

    def fit_transform(self, X, y = None, **fit_params):
        self.fit(X=X,y=y,**fit_params)
        return self.transform(X=X,y=y)

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

        self._kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters, random_state=41, max_iter=500, batch_size=1024
        )
        self._kmeans.fit(cluster_data)

        self.customer_cluster_labels = pd.DataFrame(
            {
                self.cluster_on_feature: cluster_data.index,
                "CustomerCluster": self._kmeans.labels_,
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
        df = df.merge(self._product_fraud_rate, on="ProductId", how="left")
        df = df.merge(self._provider_fraud_rate, on="ProviderId", how="left")
        df = df.merge(self._channel_fraud_rate, on="ChannelId", how="left")
        
        for col in ["ProductFraudRate", "ProviderFraudRate", "ChannelIdFraudRate"]:
            _mean = df[col].mean(skipna=True)
            df[col] = df[col].fillna(_mean)

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
            self._customer_cluster_labels, on=self.cluster_on_feature, how="left"
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














