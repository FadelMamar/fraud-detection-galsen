from pyod.models.base import BaseDetector
from typing import Sequence
import pandas as pd
import numpy as np
from tqdm import tqdm
from numbers import Integral
from functools import partial
from sklearn.base import TransformerMixin, BaseEstimator, _fit_context
from sklearn.utils.validation import validate_data
from sklearn.cluster import MiniBatchKMeans
from category_encoders import (
    BinaryEncoder,
    CountEncoder,
    HashingEncoder,
    BaseNEncoder,
    CatBoostEncoder,
    TargetEncoder,
    WOEEncoder
)
from copy import deepcopy
from typing import Optional, Union, List
from feature_engine.encoding import StringSimilarityEncoder
from feature_engine.selection import DropFeatures,DropConstantFeatures,SmartCorrelatedSelection
from sklearn.utils.validation import check_is_fitted
from feature_engine.dataframe_checks import _check_optional_contains_na
# import spacy
from group_lasso import GroupLasso
from sklearn.utils._param_validation import Interval
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import (
    RFECV,
    VarianceThreshold,
    SequentialFeatureSelector,
    SelectKBest,
    mutual_info_classif,
    SelectorMixin
)
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from scipy.stats import kendalltau
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from scipy.fft import fft
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from ..detectors import get_detector, instantiate_detector

# sklearn.set_config(enable_metadata_routing=True)


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
    nlp_model_name: str = 'en_core_web_md',

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
            regularization=woe_regularization
        )

    elif cat_encoding_method == "similarity":
        return SpaCySimilarityEncoder(nlp_model_name=nlp_model_name,
                                      missing_values='ignore',
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
    onehot_encoder=OneHotEncoder(handle_unknown="ignore"),
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
        transformers, remainder="passthrough", n_jobs=n_jobs, verbose=False,verbose_feature_names_out=False
    )

    return col_transformer


def load_feature_selector(
    n_features_to_select=10,
    cv: TimeSeriesSplit = TimeSeriesSplit(n_splits=5, gap=5000),
    estimator=DecisionTreeClassifier(
        max_depth=15, max_features=None, random_state=41, class_weight="balanced"
    ),
    name: str = "selectkbest",
    rfe_step: int = 3,
    top_k_best: int = 10,
    scoring: str = "f1",
    corr_method: str = "pearson", # pearson, spearman, kendall
    corr_threshold:float=0.8,
    k_score_func=mutual_info_classif,
    group_lasso_alpha:float=1e-2,
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

    elif name == "selectkbest":
        selector = SelectKBest(score_func=k_score_func, k=top_k_best)

    elif name == "grouplasso":
        selector = GroupLassoFeatureSelector(alpha=group_lasso_alpha)
    
    elif name == "smartcorrelated":
        selector = SmartCorrelatedSelection(method=corr_method,
                                            threshold=corr_threshold,
                                            selection_method="variance",
                                            estimator=estimator,
                                            cv=cv,
                                            scoring=scoring
                                        )

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


class SpaCySimilarityEncoder(StringSimilarityEncoder):
    """
    SpaCySimilarityEncoder variant that uses spaCy's vector-based .similarity()
    in place of the default character-matching formula.
    """ 

    def __init__(
        self,
        nlp_model_name = 'en_core_web_md',
        variables: Optional[Union[str, Sequence[str]]] = None,
        **kwargs
    ):
        
        super().__init__(variables=variables,**kwargs)
        import spacy
        self.nlp_model_name = nlp_model_name
        self.nlp = spacy.load(nlp_model_name)
    
    def compute_similarity(self, s1: str, s2: str) -> float:
        """
        Override the default 2*M/T similarity with spaCy's doc.similarity().
        """
        # spaCy returns a float in [0,1] (if vectors are normalized).
        s1 = str(s1).replace("nan", "")
        s2 = str(s2).replace("nan", "")
        doc1 = self.nlp(s1)
        doc2 = self.nlp(s2)
        # Handle missing vectors: similarity=0 if doc2 has no vector
        if (not doc1.has_vector) or (not doc2.has_vector):
            return 0.0
        return float(doc1.similarity(doc2))
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces the categorical variables with the similarity variables.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: pandas dataframe.
            The transformed dataframe. The shape of the dataframe will be different from
            the original as it includes the similarity variables in place of the
            original categorical ones.
        """

        X = X.copy() #.reset_index(drop=True)

        check_is_fitted(self)
        X = self._check_transform_input_and_state(X)
        if self.missing_values == "raise":
            _check_optional_contains_na(X, self.variables_)

        _compute_similarity = np.vectorize(self.compute_similarity)

        new_values = []
        for var in self.variables_:
            if self.missing_values == "impute":
                X[var] = X[var].astype(str).replace("nan", "")
            categories = X[var].dropna().astype(str).unique()
            column_encoder_dict = {
                x: _compute_similarity(x, self.encoder_dict_[var]) for x in categories
            }
            column_encoder_dict["nan"] = [np.nan] * len(self.encoder_dict_[var])
            encoded = np.vstack(X[var].astype(str).map(column_encoder_dict).values)
            if self.missing_values == "ignore":
                encoded[X[var].isna(), :] = np.nan
            new_values.append(encoded)

        try:
            new_features = self._get_new_features_name()
            X.loc[:, new_features] = np.hstack(new_values)
            return X.drop(columns=self.variables_, axis=1)
        except ValueError as exc:
            print(new_features, np.hstack(new_values).shape)
            raise ValueError from exc

        

class ColumnDropper(TransformerMixin, BaseEstimator):
    _parameter_constraints = {
        "cols_to_drop": [
            "array-like",
            Interval(Integral, 1, None, closed="left"),
        ]
    }

    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X=X, y=y)
    
    def check_dataframe(self,X:pd.DataFrame):
        assert isinstance(X, pd.DataFrame), "provide a DataFrame"
        assert "TX_FRAUD" not in X.columns, "Please drop TX_FRAUD column"
        
    def get_feature_names_out(self,input_features=None):
        return self._cols

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        
        self.check_dataframe(X=X)
        
        self.feature_names_in_ = X.columns
        self.n_features_in_ = len(self.feature_names_in_)
        self.is_fitted_ = True

        return self

    def transform(self, X: pd.DataFrame, y=None):
        
        self.check_dataframe(X=X)

        df = X.copy()

        if self.cols_to_drop is not None:
            df.drop(columns=self.cols_to_drop, inplace=True)
        
        self._cols = df.columns.tolist()

        return df


class ToDataframe(TransformerMixin, BaseEstimator):
    _parameter_constraints = {}
    
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X=X, y=y)
    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, **fit_params):
        
        if isinstance(X, pd.DataFrame):
            self._cols = X.columns
        else:
            self._cols = [f"to_df_{i}" for i in range(X.shape[1])]
        
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True

        return self
    
    # def get_feature_names_out(self,input_features=None):
    #     return self._cols

    def transform(self, X, y=None):
        
        df = pd.DataFrame(data=X,columns=self._cols).convert_dtypes()
        for col in df.select_dtypes(include=['object','string']):
            df[col] = df[col].astype('category')
        
        self.get_feature_names_out = df.columns.tolist()
            
        return df


class OutlierDetector(TransformerMixin, BaseEstimator):
    _parameter_constraints = {
        "detector_list": [
            "array-like",
            Interval(Integral, 1, None, closed="left"),
        ],
        "n_jobs": [int],
    }

    def __init__(self, detector_list: list[BaseDetector] = (None,), n_jobs: int = 1):
        self.detector_list = detector_list
        self.n_jobs = n_jobs

        assert all([det is not None for det in self.detector_list]), (
            "Provide detector_list"
        )

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X=X, y=y)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, **fit_params):
        assert all([det is not None for det in self.detector_list]), (
            "Provide detector_list"
        )

        validate_data(self, X=X, y=y)

        [det.fit(X) for det in self.detector_list]

        self.is_fitted_ = True

        return self

    def transform(self, X, y=None):
        validate_data(self, X=X, reset=False)

        scores = [
            det.decision_function(X).reshape((-1, 1)) for det in self.detector_list
        ]

        if len(scores) > 1:
            scores = np.hstack(scores)
        else:
            scores = scores[0]

        return scores


class DimensionReduction(TransformerMixin, BaseEstimator):
    _parameter_constraints = {
        "feature_selector_name": [str],
        # "estimator": [BaseEstimator],
        "n_splits": [int],
        "cv_gap": [int],
        "n_features_to_select": [int],
        "scoring": [str],
        "n_jobs": [int],
        "rfe_step": [int],
        "top_k_best": [int],
        "verbose": [bool],
        "do_pca": [bool],
        "pca_n_components": [int],
        "k_score_func": [callable],
    }

    def __init__(
        self,
        feature_selector_name: str = "selectkbest",
        k_score_func=mutual_info_classif,
        estimator=None,#DecisionTreeClassifier(),
        n_splits=5,
        cv_gap=5000,
        n_features_to_select=3,
        do_pca: bool = False,
        rfe_step=3,
        top_k_best=10,
        scoring="f1",
        pca_n_components: int = 20,
        n_jobs=1,
        verbose=False,
    ):
        self.n_features_to_select = n_features_to_select
        self.rfe_step = rfe_step
        self.top_k_best = top_k_best
        self.feature_selector_name = feature_selector_name
        self.k_score_func = k_score_func

        self.scoring = scoring

        self.do_pca = do_pca
        self.pca_n_components = pca_n_components

        self.n_jobs = n_jobs
        self.n_splits = n_splits
        self.cv_gap = cv_gap

        self.estimator = estimator
        self.verbose = verbose

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X=X, y=y, **fit_params)
        return self.transform(X=X, y=y)
              
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, **fit_params):
        assert any([self.do_pca, self.feature_selector_name != "None"]), (
            "Provide do_pca=True or feature_selector_name != 'None'"
        )

        validate_data(self, X=X, y=y)

        self._pca_transform = None
        self._selector = None

        
        if self.do_pca:
            n_components = min(self.pca_n_components, X.shape[1])
            if n_components == X.shape[1]:
                print(f"Clipping pca_n_components to X.shape[1]={X.shape[1]}")

            self._pca_transform = make_pipeline(
                StandardScaler(), PCA(n_components=n_components)
            )
            X = self._pca_transform.fit_transform(X=X, y=y)

        if str(self.feature_selector_name) != "None":
            cv = TimeSeriesSplit(n_splits=self.n_splits, gap=self.cv_gap)
            self._selector = load_feature_selector(
                n_features_to_select=self.n_features_to_select,
                cv=cv,
                estimator=self.estimator,
                name=self.feature_selector_name,
                top_k_best=min(self.top_k_best, X.shape[1]//2),
                k_score_func=self.k_score_func,
                rfe_step=self.rfe_step,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )
            self._selector.fit(X=X, y=y)

        self.is_fitted_ = True

        return self

    def transform(self, X, y=None):
                
        validate_data(self, X=X, reset=False)

        if self._pca_transform is not None:
            X = self._pca_transform.transform(X=X)

        if self._selector is not None:
            X = self._selector.transform(X=X)
        
        return X


class FeatureEncoding(TransformerMixin, BaseEstimator):
    _parameter_constraints = {
        "add_imputer": [bool],
        "imputer_n_neighbors": [int],
        "onehot_threshold": [int],
        "cat_encoding_method": [str],
        "n_jobs": [int],
        "cat_encoding_kwargs": [dict],
    }

    def __init__(
        self,
        cat_encoding_method: str = "binary",
        add_imputer: bool = False,
        imputer_n_neighbors: int = 5,
        onehot_threshold: int = 9,
        n_jobs: int = 1,
        cat_encoding_kwargs={},
    ):
        self.add_imputer = add_imputer
        self.imputer_n_neighbors = imputer_n_neighbors

        self.cat_encoding_method = cat_encoding_method
        self.cat_encoding_kwargs = cat_encoding_kwargs

        self.onehot_threshold = onehot_threshold

        self.n_jobs = n_jobs

        self.get_feature_names_out = None

    def check_dataframe(self, X: pd.DataFrame):
        assert isinstance(X, pd.DataFrame), "Please provide a DataFrame"
        assert X.isna().sum().sum() < 1, "Found NaN values"
        assert "TX_FRAUD" not in X.columns, "Please drop TX_FRAUD column"

    def load_cols_transformer(self, df: pd.DataFrame):
        # scalers
        scaler = StandardScaler()
        onehot_encoder = OneHotEncoder(handle_unknown="ignore")

        # numeric columns
        # numeric_cols =  df.select_dtypes(include=["number"]).columns
        # numeric_cols = [col for col in numeric_cols]
        numeric_cols = make_column_selector(dtype_include=['number'])
        transformers = [("scaled_numeric", scaler, numeric_cols)]

        # categorical columns
        if str(self.cat_encoding_method) == "None":
            print("Categorical columns will  pass-through.")

        else:
            cols = df.select_dtypes(include=["object", "string", "category"]).columns
            cols_onehot = [
                col for col in cols if df[col].nunique() < self.onehot_threshold
            ]
            cols_cat_encode = [
                col for col in cols if df[col].nunique() >= self.onehot_threshold
            ]
            cat_encoder = load_cat_encoding(
                cat_encoding_method=self.cat_encoding_method, **self.cat_encoding_kwargs
            )
            transformers = transformers + [
                ("onehot", onehot_encoder, cols_onehot),
                ("cat_encode", cat_encoder, cols_cat_encode),
            ]

        col_transformer = ColumnTransformer(
            transformers, remainder="passthrough", n_jobs=self.n_jobs, verbose=False,verbose_feature_names_out=False
        )

        return col_transformer

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        self.check_dataframe(X)

        X = X.copy()
        
        y = np.array(y)

        self._col_transformer = self.load_cols_transformer(df=X)
        self._imputer = None
        if self.add_imputer:
            self._imputer = KNNImputer(
                n_neighbors=self.imputer_n_neighbors,
                missing_values=np.nan,
                add_indicator=False,
                weights="distance",
            )
            self._imputer.fit(X=X,y=y)

        self._col_transformer.fit(X=X, y=y)

        self.feature_names_in_ = [col for col in X.columns]
        self.n_features_in_ = len(self.feature_names_in_)
        self.is_fitted_ = True

        self.get_feature_names_out = list(self._col_transformer.get_feature_names_out())

        return self

    def transform(self, X: pd.DataFrame, y=None):
        self.check_dataframe(X)

        X = X.copy()

        X = self._col_transformer.transform(X=X)

        if self._imputer is not None:
            y = np.array(y)
            X = self._imputer.transform(X=X)

        X = pd.DataFrame(data=X, columns=self.get_feature_names_out).convert_dtypes()

        cat_cols = X.select_dtypes(include=["object", "string"]).columns
        for col in cat_cols:
            X[col] = X[col].astype("category")

        return X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X=X, y=y, **fit_params)
        return self.transform(X=X, y=y)


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
        "add_imputer": [bool],
        "add_fraud_rate_features": [bool],
        "session_gap_minutes": [int],
        "n_clusters": [int],
        "cat_encoding_method": [str],
        "uid_col_name": [str],
        "cluster_on_feature": [str],
        "use_spline": [bool],
        "spline_degree": [int],
        "spline_n_knots": [int],
        "use_sincos": [bool],
        "add_seasonal_features": [bool],
        "add_fft": [bool],
    }

    def __init__(
        self,
        windows_size_in_days: list[int] = [1, 7, 30],
        uid_cols: list[str] = [
            None,
        ],
        session_gap_minutes: int = 30,
        n_clusters: int = 0,
        uid_col_name: str = "CustomerUID",
        add_fraud_rate_features: bool = True,
        cluster_on_feature: str = "CUSTOMER_ID",
        use_spline=False,
        spline_degree=3,
        spline_n_knots=6,
        use_sincos=False,
        add_seasonal_features=False,
        add_fft=False,
        behavioral_drift_cols: list[str] = [
            "AccountId",
        ],
        reorder_by=['TX_DATETIME'],
    ):
        self.windows_size_in_days = windows_size_in_days
        self.uid_cols = uid_cols
        self.uid_col_name = uid_col_name
        self.behavioral_drift_cols = behavioral_drift_cols
        self.add_fraud_rate_features = add_fraud_rate_features
        self.session_gap_minutes = session_gap_minutes
        self.n_clusters = n_clusters
        self.cluster_on_feature = cluster_on_feature

        self.add_seasonal_features = add_seasonal_features
        self.add_fft = add_fft

        self.use_spline = use_spline
        self.spline_degree = spline_degree
        self.spline_n_knots = spline_n_knots

        self.use_sincos = use_sincos

        self.reorder_by=reorder_by # passed to df.sort_values, str or list[str]


    def check_dataframe(self, X: pd.DataFrame):
        assert isinstance(X, pd.DataFrame), "Please provide a DataFrame"
        assert X.isna().sum().sum() < 1, "Found NaN values"
        assert "TX_FRAUD" not in X.columns, "Please drop TX_FRAUD column"

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pd.DataFrame, y, **fit_params):
        self.check_dataframe(X=X)
        

        # reset index > causes issues when doing cross-val
        df = X.copy().reset_index(drop=True)
        df['TX_FRAUD'] = pd.Series(y)
        
        if self.add_fraud_rate_features:
            self._product_fraud_rate = (
                df.groupby("ProductId")["TX_FRAUD"].mean().rename("ProductFraudRate")
            )
            self._provider_fraud_rate = (
                df.groupby("ProviderId")["TX_FRAUD"].mean().rename("ProviderFraudRate")
            )
            self._channel_fraud_rate = (
                df.groupby("ChannelId")["TX_FRAUD"].mean().rename("ChannelIdFraudRate")
            )

        df_cyclical = self._get_cyclical_data(df=X)
        self._cyclical_features = list(df_cyclical.columns)

        if self.use_spline:
            self._spline_transformers = {}
            for feature in self._cyclical_features:
                transformer = SplineTransformer(
                    degree=self.spline_degree,
                    n_knots=self.spline_n_knots,
                    extrapolation="periodic",
                )
                self._spline_transformers[feature] = transformer.fit(
                    df_cyclical[[feature]]
                )

        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = len(self.feature_names_in_)
        self.is_fitted_ = True

        if self.n_clusters > 0:
            self._compute_clusters_customers(df)

        return self

    def transform(self, X: pd.DataFrame, y=None):
        self.check_dataframe(X=X)

        # reset index > causes issues when doing cross-val
        df = X.copy().reset_index(drop=True)

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

        if self.use_sincos or self.use_spline:
            df = self._add_cyclical_features_transform(df)

        if self.add_fraud_rate_features:
            df = self._add_fraud_rate_features(df)

        if self.add_seasonal_features:
            df = self._add_grouped_seasonal_decomposition(df=df, freq="D", period=7)

        if self.add_fft:
            df = self._add_grouped_fft_features(df=df, freq="D", top_n_freqs=5)

        if self.n_clusters > 0:
            df = self._add_customer_clusters(df)

        df = self._cleanup(df)
        
        self.get_feature_names_out = df.columns.tolist()

        self.check_dataframe(df)

        # reorder
        if self.reorder_by is not None:
            df = df.sort_values(by=self.reorder_by).reset_index(drop=True)

        return df

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X=X, y=y, **fit_params)
        return self.transform(X=X, y=y)
    
    
    # ---------- Private Helper Methods ----------
    def _create_unique_identifier(self, df: pd.DataFrame):
        df[self.uid_col_name] = df[list(self.uid_cols)].apply(
            lambda x: "+".join(x), axis=1, raw=False, result_type="reduce"
        )

        return df

    def _get_cyclical_data(self, df: pd.DataFrame):
        X = df[
            [
                "TX_DATETIME",
            ]
        ].copy()

        X["Hour"] = X["TX_DATETIME"].dt.hour
        X["DayOfWeek"] = X["TX_DATETIME"].dt.dayofweek

        X.drop(columns=["TX_DATETIME"], inplace=True)

        return X

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
            df.sort_values(by=[col, "TX_DATETIME"], inplace=True)

            df[col + "_TimeSinceLastTxn"] = (
                df.groupby(col)["TX_DATETIME"].diff().dt.total_seconds().fillna(0) / 60
            )

            df[col + "_Txn1hCount"] = (
                df
                .set_index("TX_DATETIME")
                .groupby(col)["TRANSACTION_ID"]
                .transform(lambda x: x.rolling('1h').count())
                .reset_index(drop=True)
            )

            for day in self.windows_size_in_days:
                df[f"{col}_AvgAmount_{day}day"] = (
                    df
                    .set_index("TX_DATETIME")
                    .groupby(col)["TX_AMOUNT"]
                    .transform(lambda x: x.rolling(window=f"{day}d", min_periods=1).mean())
                    .reset_index(drop=True)
                )

        return df

    def _add_account_stats(self, df: pd.DataFrame):
        account_stats = (
            df.groupby("AccountId")["TX_AMOUNT"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "AccountMeanAmt", "std": "AccountStdAmt"})
        )
        account_stats["AccountStdAmt"] = (
            account_stats["AccountStdAmt"].fillna(1).replace(0, 1)
        )
        account_stats.fillna({"AccountMeanAmt": 0}, inplace=True)

        df = df.merge(account_stats, on="AccountId", how="left")
        df["AccountAmountZScore"] = (df["TX_AMOUNT"] - df["AccountMeanAmt"]) / df[
            "AccountStdAmt"
        ]
        df["AccountAmountOverAvg"] = df["TX_AMOUNT"] / df["AccountMeanAmt"]

        # df.fillna({"AccountAmountOverAvg":0,
        #            "AccountAmountZScore":0},inplace=True)

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
        customer_stats.fillna({"CustomerMeanAmt": 0}, inplace=True)

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
        df['ProductCategory_Account'] = df['ProductCategory'].astype(str) + "_" + df['AccountId'].astype(str)
        # df['ProductCategory_Customer'] = df['ProductCategory'].astype(str) + "_" + df['CUSTOMER_ID'].astype(str)
        # df["Country_Currency"] = (
        #     df["CountryCode"].astype(str) + "_" + df["CurrencyCode"].astype(str)
        # )
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
        df['Hour_Account'] = df['Hour'].astype(str) + "_" + df['AccountId'].astype(str)
        # df['Hour_Customer'] = df['Hour'].astype(str) + "_" + df['CUSTOMER_ID'].astype(str)
        df['DayOfWeek_Account'] = df['DayOfWeek'].astype(str) + "_" + df['AccountId'].astype(str)
        # df['DayOfWeek_Customer'] = df['DayOfWeek'].astype(str) + "_" + df['CUSTOMER_ID'].astype(str)
        # df["Country_Hour"] = (
        #     df["CountryCode"].astype(str) + "_" + df["Hour"].astype(str)
        # )
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
        df = df.sort_values(by=["TX_DATETIME"]).set_index("TX_DATETIME")

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
        df = df.sort_values(by=["TX_DATETIME"])

        for col in self.behavioral_drift_cols:
            df[f"{col}_MovingAvg5"] = (
                df.groupby(col)["TX_AMOUNT"]
                .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
                # .rolling(window=5, min_periods=1)
                # .mean()
                .reset_index(drop=True)
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

    def _add_cyclical_features_transform(self, df):
        X = df

        if self.use_spline:
            for feature in self._cyclical_features:
                transformed = self._spline_transformers[feature].transform(X[[feature]])
                spline_cols = [
                    f"{feature}_spline_{i}" for i in range(transformed.shape[1])
                ]
                X[spline_cols] = transformed

        if self.use_sincos:
            max_vals = {'hour':24,
                        'dayofweek':6
                        }
            for feature in self._cyclical_features:
                max_val = max_vals[str(feature).lower()]
                X[f"{feature}_sin"] = np.sin(2 * np.pi * X[feature] / max_val)
                X[f"{feature}_cos"] = np.cos(2 * np.pi * X[feature] / max_val)

        return df

    def _add_grouped_seasonal_decomposition(self, df, freq="D", period=7):
        features = []

        group_key = "AccountId"
        value_col = "TX_AMOUNT"
        time_col = "TX_DATETIME"

        for group_val, group_df in df.groupby(group_key):
            ts = group_df.set_index(time_col)[value_col].resample(freq).sum().fillna(0)

            if len(ts) < period * 2:
                continue

            try:
                decomposition = seasonal_decompose(ts, model="additive", period=period)
                trend = decomposition.trend.dropna()
                seasonal = decomposition.seasonal.dropna()
                resid = decomposition.resid.dropna()

                # Trend slope
                X_t = np.arange(len(trend)).reshape(-1, 1)
                y_t = trend.values
                model = LinearRegression().fit(X_t, y_t)
                trend_slope = model.coef_[0]

                # Seasonal amplitude
                seasonal_amp = seasonal.max() - seasonal.min()

                # Residual variance
                resid_var = np.var(resid)

                features.append(
                    {
                        group_key: group_val,
                        "trend_slope": trend_slope,
                        "seasonal_amplitude": seasonal_amp,
                        "residual_variance": resid_var,
                    }
                )

            except Exception:
                continue

        seasonal_df = pd.DataFrame(features)
        df = df.merge(seasonal_df, on=group_key, how="left")
        nan_mask = df[list(seasonal_df.columns)].isna()
        df[nan_mask] = 0

        return df

    def _add_grouped_fft_features(self, df, freq="D", top_n_freqs=5):
        features = []
        group_key = "AccountId"
        value_col = "TX_AMOUNT"
        time_col = "TX_DATETIME"

        for group_val, group_df in df.groupby(group_key):
            ts = group_df.set_index(time_col)[value_col].resample(freq).sum().fillna(0)

            if len(ts) < top_n_freqs * 2:
                continue

            fft_vals = np.abs(fft(ts.values))
            fft_vals = fft_vals[1 : top_n_freqs + 1]  # Exclude DC component

            fft_dict = {group_key: group_val}
            for i, val in enumerate(fft_vals, start=1):
                fft_dict[f"fft_freq_{i}"] = val

            features.append(fft_dict)

        fft_df = pd.DataFrame(features)
        df = df.merge(fft_df, on=group_key, how="left")
        nan_mask = df[list(fft_df.columns)].isna()
        df[nan_mask] = 0

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

# TODO: debug
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
        self._encoder = BinaryEncoder(drop_invariant=True,return_df=True)
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
        self.selected_idx_ = np.where(coef != 0)[0]
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


class PolyInteractions(TransformerMixin, BaseEstimator):

    _parameter_constraints = {
        "cat_cols": [
            "array-like",
            Interval(Integral, 1, None, closed="left"),
        ],
        "degree": [int],
        "cat_encoder":[TransformerMixin]
    }

    def __init__(self, cat_cols:list=None, cat_encoder=None, degree=2):
        self.cat_cols = cat_cols
        self.cat_encoder = cat_encoder        
        self.degree = degree
        
    def check_dataframe(self, X: pd.DataFrame):
        assert isinstance(X, pd.DataFrame), "Please provide a DataFrame"
        assert X.isna().sum().sum() < 1, "Found NaN values"
        assert "TX_FRAUD" not in X.columns, "Please drop TX_FRAUD column"
        for col in self.cat_cols:
            assert col in X.columns, f'{col} is not in the columns. Review the pipeline'

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X=X, y=y, **fit_params)
        return self.transform(X=X, y=y)

    def get_feature_names_out(self,input_features=None):
        return self._feature_name_out
    
    def to_dataframe(self,X,name="col"):

        if not isinstance(X,pd.DataFrame):
            X = pd.DataFrame(X,columns=[f"{name}_{i}" for i in range(X.shape[1])])
        
        return X

    def fit(self, X, y=None):

        X = X.copy()

        self.check_dataframe(X=X)
        X = X.reset_index(drop=True)

        self._numeric_cols = X.select_dtypes(include='number').columns.tolist()

        X_encoded = self.cat_encoder.fit_transform(X[self.cat_cols],y=y)

        X_encoded = self.to_dataframe(X_encoded)
        
        _cat_cols = X_encoded.columns.tolist()

        X_encoded = pd.concat([X_encoded,X[self._numeric_cols]],axis=1)
        
        _poly = PolynomialFeatures(degree=self.degree, interaction_only=True, include_bias=False)
        self._interactor = ColumnTransformer(transformers=[('poly_interact',_poly, _cat_cols + self._numeric_cols)],
                                                            remainder='passthrough',
                                                            verbose_feature_names_out=False)
        
        self._interactor.fit(X_encoded)
        
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        self.is_fitted_ = True

        return self

    def transform(self, X,y=None):

        X = X.copy()

        self.check_dataframe(X=X)
        X = X.reset_index(drop=True)

        try:
            X_encoded = self.cat_encoder.transform(X=X[self.cat_cols],y=y)
        except:
            X_encoded = self.cat_encoder.transform(X=X[self.cat_cols])

        X_encoded = self.to_dataframe(X_encoded)

        X_encoded = pd.concat([X_encoded,X[self._numeric_cols]],axis=1)
        
        poly_out = self._interactor.transform(X=X_encoded)
        poly_out = self.to_dataframe(poly_out,name='poly_inter')

        self._feature_name_out = poly_out.columns.tolist()

        return  poly_out


    

def load_workflow(
        classifier=None,
    cols_to_drop=None,
    pca_n_components=20,
    detector_list=None,
    n_splits=5,
    cv_gap=5000,
    scoring="f1",
    onehot_threshold=9,
    session_gap_minutes=60 * 3,
    uid_cols=[
        None,
    ],
    uid_col_name="CustomerUID",
    add_fraud_rate_features: bool = True,
    reorder_by=['TX_DATETIME',],
    behavioral_drift_cols=[
        "AccountId",
    ],
    feature_select_estimator=DecisionTreeClassifier(),
    feature_selector_name: str | None = "selectkbest",
    top_k_best=10,
    seq_n_features_to_select=3,
    corr_method="spearman", # pearson, spearman, kendall
    corr_threshold: float = 0.8,
    windows_size_in_days=[1, 7, 30],
    cat_encoding_method: str | None = "binary",
    cat_similarity_encode:list=None,
    nlp_model_name='en_core_web_md',
    cat_encoding_kwargs={},
    cluster_on_feature="AccountId",
    add_poly_interactions=False,
    interaction_cat_cols=None,
    poly_degree=2,
    poly_cat_encoder_name="catboost",
    imputer_n_neighbors=9,
    add_fft=False,
    add_seasonal_features=False,
    use_nystrom=False,
    use_sincos=False,
    use_spline=False,
    spline_degree=3,
    spline_n_knots=6,
    nystroem_kernel="poly",
    nystroem_components=50,
    add_imputer=False,
    rfe_step=3,
    n_clusters=0,
    k_score_func=mutual_info_classif,
    do_pca=True,
    verbose=False,
    n_jobs=2,
):
    # preliminary feature expansion
    feature_engineer = FraudFeatureEngineer(
        windows_size_in_days=windows_size_in_days,
        uid_cols=uid_cols,
        reorder_by=list(reorder_by) if isinstance(reorder_by,Sequence) else reorder_by,
        add_fraud_rate_features=add_fraud_rate_features,
        session_gap_minutes=session_gap_minutes,
        n_clusters=n_clusters,
        add_fft=add_fft,
        add_seasonal_features=add_seasonal_features,
        use_sincos=use_sincos,
        use_spline=use_spline,
        spline_degree=spline_degree,
        spline_n_knots=spline_n_knots,
        behavioral_drift_cols=behavioral_drift_cols,
        cluster_on_feature=cluster_on_feature,
        uid_col_name=uid_col_name,  # name given to uid cols created from interactions of uid_cols
    )
    
    workflow_steps = [('feature_engineer', feature_engineer)]

    # drop constant features and cols_to_drop
    workflow_steps.append(('dropper',DropFeatures(cols_to_drop)))
    workflow_steps.append(('drop_constant',DropConstantFeatures(tol=1.0,missing_values='ignore')))

    # similarity based encoding 
    if cat_similarity_encode is not None:
        assert len(set(cat_similarity_encode) & set(interaction_cat_cols)) == 0, "cat_similarity_encode columns must not be in interaction_cat_cols"
        assert len(set(cat_similarity_encode) & set(cols_to_drop)) == 0, "cat_similarity_encode cols must not be in cols_to_drop"
        assert isinstance(cat_similarity_encode,Sequence), "It should be a list of columns to encode"
        similarity_encoder = load_cat_encoding(cat_encoding_method='similarity',
                                                cols=cat_similarity_encode,
                                                nlp_model_name=nlp_model_name,
                                                )
        workflow_steps.append(('similarity_encoder',similarity_encoder))        
        
    # encodes features and impute missing values
    encoder = FeatureEncoding(
        add_imputer=add_imputer,
        cat_encoding_method=str(cat_encoding_method),
        cat_encoding_kwargs=cat_encoding_kwargs,
        imputer_n_neighbors=imputer_n_neighbors,
        n_jobs=n_jobs,
        onehot_threshold=onehot_threshold,
    )
    
    # create interaction features
    if add_poly_interactions :
        interactor = PolyInteractions(
            cat_cols=interaction_cat_cols,
            cat_encoder=load_cat_encoding(cat_encoding_method=poly_cat_encoder_name,
                                          hash_n_components=12,
                                          base=10,
                                          return_df=True,
                                          drop_invariant=True),
            degree=poly_degree,
        )            
        
        encoder_interactor = FeatureUnion(
                        transformer_list=[('poly_interact',interactor),
                                          ('encoder',encoder),
                                        ], 
                        n_jobs=n_jobs
                    )
        workflow_steps.append(('encode-and-interact', encoder_interactor)) 

    else:
        workflow_steps.append(('encoder',encoder))
    
    # add pca
    if do_pca:
        pca = PCA(n_components=pca_n_components, random_state=42)
        workflow_steps.append(("dim_reduce", pca))
    
    if str(feature_selector_name) != 'None':
        
        select_features = load_feature_selector(
                                        n_features_to_select=10,
                                        cv = TimeSeriesSplit(n_splits=n_splits, gap=cv_gap),
                                        estimator=feature_select_estimator,
                                        seq_n_features_to_select=seq_n_features_to_select,
                                        name = str(feature_selector_name),
                                        rfe_step = rfe_step,
                                        top_k_best = top_k_best,
                                        scoring = scoring,
                                        corr_method = corr_method, # pearson, spearman, kendall
                                        corr_threshold=corr_threshold,
                                        k_score_func=k_score_func,
                                        group_lasso_alpha=1e-2,
                                        n_jobs = n_jobs,
                                        verbose = verbose>0,
                                    )

        # numeric_cols = make_column_selector(dtype_include=['number'])
        # select_features = ColumnTransformer(transformers=[('select_features',select_features,numeric_cols)],
        #                                remainder='passthrough', # for categorical variables
        #                                force_int_remainder_cols=False,
        #                                verbose_feature_names_out=False
        #                             )
        workflow_steps.append(('to_df_0',ToDataframe()))
        workflow_steps.append(("feature_selector", select_features))

    # concat outlier detection and/or nystroem
    advanced_features = []
    if detector_list is not None:
        pyod_det = OutlierDetector(detector_list=detector_list)
        advanced_features.append(("outlier_scores", pyod_det))
    
    if use_nystrom:
        nystrom = Nystroem(
            kernel=nystroem_kernel,
            degree=2,
            n_components=nystroem_components,
        )
        advanced_features.append(('nystrom',nystrom))

    if len(advanced_features)>0:
        advanced_features = FeatureUnion(
            transformer_list=advanced_features, n_jobs=n_jobs
        )
          
    
    if isinstance(advanced_features, TransformerMixin):
        workflow_steps.append(('to_df_1',ToDataframe()))
        workflow_steps.append(('advanced_features',advanced_features))
    
    
    
    # Final Pipeline
    if classifier is not None:
        to_df = ToDataframe()
        workflow_steps = workflow_steps + [('to_df_2', to_df), ('model', classifier)]
    
    workflow = Pipeline(steps=workflow_steps)

    return workflow


# AdvancedFeatures = DimensionReduction