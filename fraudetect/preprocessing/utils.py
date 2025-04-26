from pyod.models.base import BaseDetector
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import partial
from category_encoders import (
    BinaryEncoder,
    CountEncoder,
    HashingEncoder,
    BaseNEncoder,
    CatBoostEncoder,
    TargetEncoder,
    WOEEncoder,
)
from feature_engine.selection import SmartCorrelatedSelection, RecursiveFeatureAddition

from group_lasso import GroupLasso
from sklearn.base import TransformerMixin, BaseEstimator, _fit_context


# import spacy
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import (
    RFECV,
    SequentialFeatureSelector,
    SelectKBest,
    mutual_info_classif,
    SelectorMixin,
)
from sklearn.pipeline import FeatureUnion
from sklearn.tree import DecisionTreeClassifier
from datetime import timedelta
from ..detectors import get_detector, instantiate_detector


def generate_rolling_group_time_splits(
    df, date_col, group_col, val_window_days=30, n_splits=3, min_train_days=None
):
    """
    Generate rolling, group-aware, time-based splits (no overlap of groups). By chatGPT :)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    min_date = df[date_col].min()
    max_date = df[date_col].max()
    total_days = (max_date - min_date).days

    max_shift = total_days - val_window_days
    if max_shift <= 0:
        raise ValueError("Not enough span for the given val_window_days")

    shifts = np.linspace(0, max_shift, n_splits)
    splits = []

    for shift in shifts:
        val_start = min_date + timedelta(days=int(shift))
        val_end = val_start + timedelta(days=val_window_days)

        if min_train_days is not None:
            min_train_date = val_start - timedelta(days=min_train_days)
            train_mask = (df[date_col] < val_start) & (df[date_col] >= min_train_date)
        else:
            train_mask = df[date_col] < val_start
        val_mask = (df[date_col] >= val_start) & (df[date_col] < val_end)

        train_groups = set(df.loc[train_mask, group_col])
        val_groups = set(df.loc[val_mask, group_col])
        overlap = train_groups & val_groups
        if overlap:
            val_mask &= ~df[group_col].isin(overlap)

        train_idx = df[train_mask].index.to_list()
        val_idx = df[val_mask].index.to_list()
        splits.append((train_idx, val_idx))

    return splits


def get_train_val_split(
    train_data: pd.DataFrame, val_window_days=30, id_column="AccountId"
):
    splits = generate_rolling_group_time_splits(
        train_data,
        "TX_DATETIME",
        id_column,
        val_window_days=val_window_days,
        n_splits=3,
        min_train_days=None,
    )

    train_idx, _ = splits[1]

    _, val_idx = splits[2]

    df_train = train_data.iloc[train_idx, :]
    df_val = train_data.iloc[val_idx, :]

    intersect = np.intersect1d(df_train["AccountId"], df_val["AccountId"]).shape

    print("Number of common AccountId between train&val: ", intersect)

    X_train, y_train = df_train.drop(columns=['TX_FRAUD']), df_train['TX_FRAUD']

    X_val, y_val = df_val.drop(columns=['TX_FRAUD']), df_val['TX_FRAUD']

    return X_train, y_train, X_val, y_val


def load_cat_encoding(
    cat_encoding_method: str,
    cols=None,
    hash_n_components=7,
    handle_missing="value",
    return_df=True,
    hash_method="md5",
    drop_invariant=False,
    handle_unknown="value",
    base: int = 4,
    woe_randomized=True,
    woe_sigma=0.05,
    woe_regularization=1.0,
    nlp_model_name: str = "en_core_web_md",
):
    if cat_encoding_method == "binary":
        return BinaryEncoder(
            handle_missing=handle_missing,
            cols=cols,
            drop_invariant=drop_invariant,
            handle_unknown=handle_unknown,
        )

    elif cat_encoding_method == "count":
        return CountEncoder(
            handle_missing=handle_missing,
            cols=cols,
            drop_invariant=drop_invariant,
            handle_unknown=handle_unknown,
        )

    elif cat_encoding_method == "hashing":
        return HashingEncoder(
            n_components=hash_n_components,
            hash_method=hash_method,
            cols=cols,
            return_df=return_df,
            drop_invariant=drop_invariant,
        )

    elif cat_encoding_method == "base_n":
        return BaseNEncoder(
            return_df=return_df,
            cols=cols,
            handle_missing=handle_missing,
            drop_invariant=drop_invariant,
            handle_unknown=handle_unknown,
            base=base,
        )

    elif cat_encoding_method == "catboost":
        return CatBoostEncoder(
            return_df=return_df,
            cols=cols,
            handle_missing=handle_missing,
            drop_invariant=drop_invariant,
            handle_unknown=handle_unknown,
        )

    elif cat_encoding_method == "target_enc":
        return TargetEncoder(
            return_df=return_df,
            cols=cols,
            handle_missing=handle_missing,
            drop_invariant=drop_invariant,
            handle_unknown=handle_unknown,
        )

    elif cat_encoding_method == "woe":
        return WOEEncoder(
            return_df=return_df,
            cols=cols,
            handle_missing=handle_missing,
            drop_invariant=drop_invariant,
            handle_unknown=handle_unknown,
            randomized=woe_randomized,
            sigma=woe_sigma,
            regularization=woe_regularization,
        )

    elif cat_encoding_method == "similarity":
        return SpaCySimilarityEncoder(
            nlp_model_name=nlp_model_name,
            missing_values="ignore",
            variables=cols,
        )

    else:
        raise NotImplementedError(
            f"cat_encoding_method {cat_encoding_method} is not implemented."
        )


def load_cols_transformer(
    df: pd.DataFrame,
    onehot_threshold=9,
    n_jobs=1,
    scaler=StandardScaler(),
    onehot_encoder=HashingEncoder(n_components=5),
    cat_encoding_method="binary",
    **cat_encoding_kwargs,
):
    numeric_cols = df.select_dtypes(include=["number"]).columns
    transformers = [("scaled", scaler, numeric_cols)]

    # categorical columns
    if str(cat_encoding_method) == "None":
        print("Categorical columns will not be encoded by ColumnTransformer.")

    else:
        cols = df.select_dtypes(include=["object", "string"]).columns
        cols_onehot = [col for col in cols if df[col].nunique() < onehot_threshold]
        cols_cat_encode = [col for col in cols if df[col].nunique() >= onehot_threshold]
        cat_encoder = load_cat_encoding(
            cat_encoding_method=cat_encoding_method, **cat_encoding_kwargs
        )
        transformers = transformers + [
            ("onehot", onehot_encoder, cols_onehot),
            ("cat_encode", cat_encoder, cols_cat_encode),
        ]

    col_transformer = ColumnTransformer(
        transformers,
        remainder="passthrough",
        n_jobs=n_jobs,
        verbose=False,
        verbose_feature_names_out=False,
    )

    return col_transformer


def load_feature_selector(
    n_features_to_select=10,
    cv: TimeSeriesSplit = TimeSeriesSplit(n_splits=5, gap=5000),
    estimator=DecisionTreeClassifier(
        max_depth=5, max_features=None, random_state=41, class_weight="balanced"
    ),
    name: str = "selectkbest",
    rfe_step: int = 3,
    top_k_best: int = 10,
    rf_threshold: float = 0.05,
    scoring: str = "f1",
    corr_method: str = "spearman",  # pearson, spearman, kendall
    corr_threshold: float = 0.8,
    k_score_func=mutual_info_classif,
    group_lasso_alpha: float = 1e-2,
    n_jobs: int = 4,
    verbose: bool = False,
) -> SelectorMixin:
    selector = None

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

    elif name == "rfacv":
        selector = RecursiveFeatureAddition(
            estimator=estimator, scoring=scoring, cv=cv, threshold=rf_threshold
        )

    elif name == "selectkbest":
        selector = SelectKBest(score_func=k_score_func, k=top_k_best)

    elif name == "grouplasso":
        selector = GroupLassoFeatureSelector(alpha=group_lasso_alpha)

    elif name == "smartcorrelated":
        selector = SmartCorrelatedSelection(
            method=corr_method,
            threshold=corr_threshold,
            selection_method="variance",
            estimator=estimator,
            cv=cv,
            scoring=scoring,
        )
    elif name == "psi":
        selector = ...  # DropHighPSIFeatures()
        raise NotImplementedError

    else:
        raise NotImplementedError

    return selector


def get_feature_selector(name: str, config: dict) -> dict:
    """
    Get the configuration based on the name.
    """
    if name not in config:
        raise ValueError(f"Model {name} not found in config.")

    cfg = config[name].copy()

    # Remove from the config dictionary
    selector = cfg.pop("selector")

    return selector, cfg


# ------------ pyod detectors
def fit_outliers_detectors(
    detector_list: list[BaseDetector], X_train: np.ndarray
) -> list[BaseDetector] | FeatureUnion:
    model_list = list()
    for model in tqdm(detector_list, desc="fitting-outliers-det-pyod"):
        model.fit(X_train)
        model_list.append(model)
    return model_list


def compute_score(det: BaseDetector, X):
    score = det.decision_function(X)
    return score.reshape(-1, 1)


def fit_detector(det: BaseDetector, X):
    return det.fit(X)


def concat_outliers_scores_pyod(
    fitted_detector_list: list[BaseDetector] | FeatureUnion,
    X: np.ndarray,
):
    if isinstance(fitted_detector_list, list):
        scores = []
        for model in tqdm(fitted_detector_list, desc="concat-outliers-scores-pyod"):
            score = model.decision_function(X)
            score = score.reshape((-1, 1))
            scores.append(score)

        X_t = np.hstack([X] + scores)
        return X_t

    else:
        raise NotImplementedError


def load_transforms_pyod(
    X_train: np.ndarray,
    outliers_det_configs: dict,
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

    # assert isinstance(outliers_det_configs, dict), (
    #     f"received {type(outliers_det_configs)}"
    # )

    model_list = list()

    # instantiate detectors
    names = sorted(outliers_det_configs.keys(), reverse=False)
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

class GroupLassoFeatureSelector(SelectorMixin, BaseEstimator):
    """
    Group Lasso-based feature selector for mixed categorical and numerical data.
    Inherits from SelectorMixin for seamless pipeline integration.

    Parameters
    ----------
    alpha : float, default=0.01
        Regularization strength for group lasso.
    """
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.scaler = StandardScaler()
        
        self.model = None
        self.selected_idx_ = None
        self.groups_ = None
        self.feature_names_in_ = None
        self.feature_names_encoded_ = None

    def fit(self, X, y=None):  # noqa: D102
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame")
        
        X = X.reset_index(drop=True)

        self.feature_names_in_ = list(X.columns)
        
        # Identify categorical columns
        cat_cols = X.select_dtypes(include='category').columns.tolist()

        # Encode categoricals
        self._encoder = HashingEncoder(drop_invariant=True,return_df=True)
        if cat_cols:
            X_cat = self._encoder.fit_transform(X[cat_cols],y=y)
        else:
            raise ValueError("No columns found in the DataFrame with dtype='category'")

        # Numeric features
        X_num = X.drop(columns=cat_cols, errors='ignore')
        # Combine encoded and numeric
        X_encoded = pd.concat([X_num, X_cat], axis=1)
        self.feature_names_encoded_ = list(X_encoded.columns)

        # Build group assignments
        groups = []
        # Numeric columns: no regularization
        for _ in X_num.columns:
            groups.append(-1)
        # Categorical binary columns: each original category gets a group
        for col in X_cat.columns:
            orig = col.split('_')[0]
            # group index is the position of orig in cat_cols + 1
            grp = cat_cols.index(orig) + 1
            groups.append(grp)
        self.groups_ = groups

        # Scale and fit GroupLasso
        X_scaled = self.scaler.fit_transform(X_encoded)
        self.model = GroupLasso(
            groups=self.groups_,
            group_reg=self.alpha,
            l1_reg=0.0,
            scale_reg='none',
            supress_warning=True
        )
        self.model.fit(X_scaled, y)

        # Selected feature indices in encoded space
        coef = np.ravel(self.model.coef_)
        self.selected_idx_ = np.where(~np.isclose(coef,0,atol=1e-3))[0]
        return self

    def _get_support_mask(self):  # noqa: D105
        # Boolean mask for encoded features
        mask = np.zeros(len(self.feature_names_encoded_), dtype=bool)
        mask[self.selected_idx_] = True
        return mask

    def transform(self, X):  

        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame")
        
        X = X.reset_index(drop=True)
        
        # Repeat encoding and scaling
        cat_cols = X.select_dtypes(include='category').columns.tolist()
        if cat_cols:
            X_cat = self._encoder.transform(X[cat_cols])
        else:
            raise ValueError("No columns found in the DataFrame with dtype='category'")
        
        X_num = X.drop(columns=cat_cols, errors='ignore')
        X_encoded = pd.concat([X_num, X_cat], axis=1)
        X_scaled = self.scaler.transform(X_encoded)

        # Apply support mask
        mask = self._get_support_mask()
        return X_scaled[:, mask]

    def get_support(self, indices=False, **kwargs):  # noqa: D105
        mask = self._get_support_mask()
        if indices:
            return np.where(mask)[0]
        return mask

    @property
    def n_features_in_(self):
        return len(self.self.feature_names_in_)

    def feature_names_in(self):
        return self.feature_names_in_
    
    def get_feature_names_out(input_features=None):
        return 
