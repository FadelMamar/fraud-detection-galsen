from pyod.models.base import BaseDetector
from typing import Sequence
import pandas as pd
import numpy as np
from tqdm import tqdm
from numbers import Integral
from functools import partial
from sklearn.base import TransformerMixin, BaseEstimator, _fit_context
from sklearn.utils.validation import validate_data
from sklearn.cluster import MiniBatchKMeans, Birch
from sklearn.metrics import silhouette_score
from category_encoders import (
    BinaryEncoder,
    CountEncoder,
    HashingEncoder,
    BaseNEncoder,
    CatBoostEncoder,
    TargetEncoder,
    WOEEncoder
)
from astropy.timeseries import LombScargle
from copy import deepcopy
from typing import Optional, Union, List
from feature_engine.encoding import StringSimilarityEncoder
from feature_engine.selection import (DropFeatures,DropConstantFeatures,
                                      SmartCorrelatedSelection,
                                      DropDuplicateFeatures,
                                      RecursiveFeatureAddition,
                                      RecursiveFeatureElimination,
                                      DropHighPSIFeatures
                                      )
from sklearn.utils.validation import check_is_fitted
from feature_engine.dataframe_checks import _check_optional_contains_na
# import spacy
from group_lasso import GroupLasso
from itertools import product
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
        max_depth=5, max_features=None, random_state=41, class_weight="balanced"
    ),
    name: str = "selectkbest",
    rfe_step: int = 3,
    top_k_best: int = 10,
    rf_threshold:float=0.05,
    scoring: str = "f1",
    corr_method: str = "spearman", # pearson, spearman, kendall
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
        
    elif name == 'rfacv':
       selector = RecursiveFeatureAddition(estimator=estimator,
                                           scoring=scoring,
                                           cv=cv,
                                           threshold=rf_threshold
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
    elif name == 'psi':
        selector = ... # DropHighPSIFeatures()
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
                cat_encoding_method=self.cat_encoding_method,**self.cat_encoding_kwargs
            )
            transformers = transformers + [
                ("onehot", onehot_encoder, cols_onehot),
                ("cat_encode", cat_encoder, cols_cat_encode),
            ]

        col_transformer = ColumnTransformer(
            transformers, remainder="passthrough", n_jobs=self.n_jobs, verbose=False,verbose_feature_names_out=True
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
            "AccountId","CUSTOMER_ID"
        ],
        session_gap_minutes: int = 30,
        uid_col_name: str = "CustomerUID",
        add_fraud_rate_features: bool = True,
        use_spline=False,
        spline_degree=3,
        spline_n_knots=6,
        use_sincos=False,
        add_seasonal_features=False,
        add_fft=False,
        behavioral_drift_cols: list[str] = [
            "AccountId","CustomerUID"
        ],
        reorder_by=['TX_DATETIME'],
    ):
        self.windows_size_in_days = windows_size_in_days
        self.uid_cols = uid_cols
        self.uid_col_name = uid_col_name
        self.behavioral_drift_cols = behavioral_drift_cols
        self.add_fraud_rate_features = add_fraud_rate_features
        self.session_gap_minutes = session_gap_minutes

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


        return self

    def transform(self, X: pd.DataFrame, y=None):
        self.check_dataframe(X=X)

        # reset index > causes issues when doing cross-val
        df = X.copy().reset_index(drop=True)

        if all([col is not None for col in self.uid_cols]):
            df = self._create_unique_identifier(df)
            df = self._add_unique_client_stats(df)

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

    def _add_temporal_features(self, df):
        df["Hour"] = df["TX_DATETIME"].dt.hour
        df["DayOfWeek"] = df["TX_DATETIME"].dt.dayofweek
        df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
        df["IsNight"] = df["Hour"].between(0, 6).astype(int)

        id_columns = ["AccountId", "CUSTOMER_ID"]
        if self.uid_col_name in df.columns:
            id_columns.append(self.uid_col_name)

        for col in set(id_columns):
            df.sort_values(by=[col, "TX_DATETIME"], inplace=True)

            df[f"{col}_TimeSinceLastTxn"] = (
                df.groupby(col)["TX_DATETIME"].diff().dt.total_seconds().fillna(0) / 60
            )

            df[f"{col}_Txn6hCount"] = (
                df
                .set_index("TX_DATETIME")
                .groupby(col)["TRANSACTION_ID"]
                .transform(lambda x: x.rolling('6h').count())
                .reset_index(drop=True)
            )
            
            df[f"{col}_Txn1dayCount"] = (
                df
                .set_index("TX_DATETIME")
                .groupby(col)["TRANSACTION_ID"]
                .transform(lambda x: x.rolling('1D').count())
                .reset_index(drop=True)
            )
            
            df[f"{col}_Txn3dayCount"] = (
                df
                .set_index("TX_DATETIME")
                .groupby(col)["TRANSACTION_ID"]
                .transform(lambda x: x.rolling('3D').count())
                .reset_index(drop=True)
            )
            
            df[f"{col}_Txn7dayCount"] = (
                df
                .set_index("TX_DATETIME")
                .groupby(col)["TRANSACTION_ID"]
                .transform(lambda x: x.rolling('7D').count())
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
    
    def _add_unique_client_stats(self, df):

        for value_col in ["TX_AMOUNT","Value"]:
            customer_stats = (
                df.groupby(self.uid_col_name)[value_col]
                .agg(["mean", "std"])
                .rename(columns={"mean": f"ClientMean{value_col}", "std": f"ClientStd{value_col}"})
            )
            customer_stats[f"ClientStd{value_col}"] = (
                customer_stats[f"ClientStd{value_col}"].fillna(1).replace(0, 1)
            )
            customer_stats.fillna({f"ClientMean{value_col}": 0}, inplace=True)

            df = df.merge(customer_stats, on=self.uid_col_name, how="left")
            df[f"Client{value_col}ZScore"] = (df[value_col] - df[f"ClientMean{value_col}"]) / df[
                f"ClientStd{value_col}"
            ]
            df[f"Client{value_col}OverAvg"] = df[value_col] / df[f"ClientMean{value_col}"]

        return df

    def _add_categorical_cross_features(self, df):
        df["Channel_ProductCategory"] = (
            df["ChannelId"].astype(str) + "_" + df["ProductCategory"].astype(str)
        )
        df['ProductCategory_Account'] = df['ProductCategory'].astype(str) + "_" + df['AccountId'].astype(str)
        
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
        df['DayOfWeek_Account'] = df['DayOfWeek'].astype(str) + "_" + df['AccountId'].astype(str)

        return df

    def _add_frequency_features(self, df):

        # daily number of transactions per account
        df["TxnDate"] = df["TX_DATETIME"].dt.date
        txn_freq = (
            df.groupby(["AccountId", "TxnDate"])["TRANSACTION_ID"]
            .count()
            .rename("DailyAccountTxnCount")
        )
        df = df.merge(txn_freq, on=["AccountId", "TxnDate"])

        ## daily number of transactions for each unique client
        if self.uid_col_name in df.columns:
            txn_freq = (
                df.groupby([self.uid_col_name, "TxnDate"])["TRANSACTION_ID"]
                .count()
                .rename("DailyClientTxnCount")
            )
            df = df.merge(txn_freq, on=[self.uid_col_name, "TxnDate"])

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
        
        _behavioral_drift_cols = list(self.behavioral_drift_cols) 
        if self.uid_col_name in df.columns:
            _behavioral_drift_cols.append(self.uid_col_name)
        
        _behavioral_drift_cols = set(_behavioral_drift_cols)
        
        for value_col in ['Value','TX_AMOUNT']:
            for col in _behavioral_drift_cols:
                mean_7d = df.groupby(col)[value_col].transform(
                    lambda x: x.rolling("7d").mean()
                )
                std_7d = df.groupby(col)[value_col].transform(
                    lambda x: x.rolling("7d").std()
                ).replace(0, 1).fillna(1)
                
                mean_30d = df.groupby(col)[value_col].transform(
                    lambda x: x.rolling("30d").mean()
                )
                std_30d = df.groupby(col)[value_col].transform(lambda x: x.rolling("30d").std()
                                                               ).replace(0, 1).fillna(1)

                df[col + f"_{value_col}_RatioTo7dAvg"] = df[value_col] / mean_7d
                df[col +  f"_{value_col}_RatioTo30dAvg"] = df[value_col] / mean_30d
                df[col +  f"_{value_col}_ZScore_7d"] = (df[value_col] - mean_7d) / std_7d
                df[col + f"_{value_col}_ZScore_30d"] = (df[value_col] - mean_30d) / std_30d

        
        df.reset_index(inplace=True)

        return df

    def _compute_avg_txn_features(self, df):
        df = df.sort_values(by=["TX_DATETIME"])

        for col in self.behavioral_drift_cols:
            df[f"{col}_MovingAvg5"] = (
                df.groupby(col)["TX_AMOUNT"]
                .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
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
            batch_time.diff().dt.total_seconds().rename("TimeBetweenBatches").fillna(0) / 60
        )
        txn_per_batch = (
            df.groupby("BatchId")["TRANSACTION_ID"].count().rename("TxnPerBatch")
        )
        df = df.merge(txn_per_batch, on="BatchId", how="left")
        df = df.merge(batch_time_gap, left_on="BatchId", right_index=True, how="left")
        return df

    def _compute_session_features(self, df):
        id_col = self.uid_col_name if (self.uid_col_name in df.columns) else "AccountId"
        df = df.sort_values(by=[id_col, "TX_DATETIME"])
        timeDiff = (
            df.groupby(id_col)["TX_DATETIME"].diff().dt.total_seconds().fillna(0) /60
        )
        df["NewSession"] = (timeDiff > self.session_gap_minutes).fillna(True)
        df["SessionId"] = df.groupby(id_col)["NewSession"].cumsum()
        session_stats = (
            df.groupby([id_col, "SessionId"])
            .agg(
                SessionTxnCount=("TRANSACTION_ID", "count"),
                SessionAmt=("TX_AMOUNT", "sum"),
                SessionAmtMean=("TX_AMOUNT", "mean"),
                SessionAmtStd=("TX_AMOUNT", "std"),
                SessionValue=("Value", "sum"),
                SessionValueMean=("Value","mean"),
                SessionValueStd=("Value","std"),
                SessionDuration=(
                    "TX_DATETIME",
                    lambda x: (x.max() - x.min()).total_seconds() / 60,
                ),
                SessionChannel=(
                    "ChannelId",
                    lambda x: x.mode().sort_values().iloc[0] if not x.mode().empty else np.nan,
                ),
                SessionMainProductCategory=(
                    "ProductCategory",
                    lambda x: x.mode().sort_values().iloc[0] if not x.mode().empty else np.nan,
                ),
                SessionMainProductId=(
                    "ProductId",
                    lambda x: x.mode().sort_values().iloc[0] if not x.mode().empty else np.nan,
                ),
                SessionMainProvider=(
                    "ProviderId",
                    lambda x: x.mode().sort_values().iloc[0] if not x.mode().empty else np.nan,
                ),
                SessionMainPricingStrat=(
                    "PricingStrategy",
                    lambda x: x.mode().sort_values().iloc[0] if not x.mode().empty else np.nan,
                ),
            )
            .reset_index()
        )

        for col in session_stats.columns:
            session_stats[col] = session_stats[col].fillna(session_stats[col].mode())

        df = df.merge(session_stats, on=[id_col, "SessionId"], how="left")

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

        group_key = self.uid_col_name if self.uid_col_name in df.columns else "AccountId"
        value_col = "TX_AMOUNT"
        time_col = "TX_DATETIME"

        for group_val, group_df in df.groupby(group_key):
            ts = group_df.set_index(time_col)[value_col].resample(freq).sum().interpolate(method='nearest') #.fillna(0)

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
    # TODO: Debug
    def _add_grouped_fft_features(self, df, freq="D", top_n_freqs=5):
        features = []
        group_key = self.uid_col_name if self.uid_col_name in df.columns else "AccountId"

        for value_col in ["TX_AMOUNT","Value"]:

            for group_val, group_df in df.groupby(group_key):
                ts = group_df.set_index("TX_DATETIME")[value_col].resample(freq).sum().interpolate(method='nearest') #.fillna(0)

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


class AdvancedFeatureEngineer(TransformerMixin, BaseEstimator):
    _parameter_constraints = {}

    def __init__(self, 
                 min_period=24, # 1 day
                 max_period = 24 * 7, # 1 week
                 col_uid_name:str='CustomerUID',
                 uid_cols: list[str] = [
                     "AccountId","CUSTOMER_ID"
                 ],
                 n_clusters:int=0,
                 add_cum_features:bool=True,
                 add_lombscargle_features:bool=False,
                 add_amount_value_ratio:bool=True,
                #  lombscarge_value_cols=['TX_AMOUNT','Value'],
                 n_freqs=10):
        # Choose range: daily to weekly periodicity (assuming t is in hours)

        self.min_period=min_period
        self.max_period = max_period
        # self.lombscarge_value_cols = lombscarge_value_cols
        self.col_uid_name = col_uid_name
        self.uid_cols = uid_cols
        self.n_freqs=n_freqs
        self.add_lombscargle_features=add_lombscargle_features
        self.add_cum_features=add_cum_features
        self.add_amount_value_ratio=add_amount_value_ratio
        self.n_clusters = n_clusters

    def check_dataframe(self, X: pd.DataFrame):
        assert isinstance(X, pd.DataFrame), "Please provide a DataFrame"
        assert X.isna().sum().sum() < 1, "Found NaN values"
        assert "TX_FRAUD" not in X.columns, "Please drop TX_FRAUD column"
    
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X=X, y=y)
    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, **fit_params):

        df = X.reset_index(drop=True)
        self.check_dataframe(df)
        
        # create_unique_identifier
        if self.col_uid_name not in df.columns:
            df[self.col_uid_name] = df[list(self.uid_cols)].apply(
                    lambda x: "+".join(x), axis=1, raw=False, result_type="reduce"
                )

        if self.n_clusters > 0:
            self._cat_encoder = load_cat_encoding(cat_encoding_method='binary',
                                                  return_df=True,
                                                  drop_invariant=False)
            self._compute_clusters_customers(df)
        
        self.n_features_in_ = df.shape[1]
        self.feature_names_in_ = df.columns.tolist()
                
        self.is_fitted_ = True        

        return self
    
    def transform(self, X, y=None):
        
        self.check_dataframe(X)

        df = X.reset_index(drop=True)
        df.sort_values(by=["TX_DATETIME"], inplace=True)

        # -- is debit
        df['is_debit'] = (df['TX_AMOUNT'] < 0)*1

        # -- add TX_AMOUNT/Value ratio
        if self.add_amount_value_ratio:
            df['TX_AMOUNT_Value_ratio'] = df['TX_AMOUNT'] / df['Value']
        
        if self.add_cum_features:
            df = self._add_cumulative_features(df)
        
        df = self._add_deltas(df)
        df = self._add_categorical_temporal_features(df)

        if self.n_clusters > 0:
            df = self._add_customer_clusters(df)
        #-- TODO: debug behavior first
        # df = self._compute_client_value_counts(df, self.col_uid_name, ["ChannelId","ProductCategory",
        #                                                                "ProviderId","PricingStrategy"]
        #                                        )
        # add lombScargle features
        if self.add_lombscargle_features:
            for value_col in ['TX_AMOUNT','Value']:
                df = self._add_lombscargle_features(df, value_col=value_col)
        
        self.get_feature_names_out = df.columns.tolist()
        
        self.check_dataframe(X)

        return df
    
    # -------- advanced features
    def _add_categorical_temporal_features(self, df):

        comb = product(['IsNight','IsWeekend'],['ChannelId','ProductCategory',
                                                     'ProviderId','PricingStrategy',
                                                     'ProductId'])

        for col1,col2 in comb:
            df[f"{col1}_{col2}"] = df[col1].astype(str) + '_' + df[col2].astype(str)
            df[f"{col1}_{col2}"] = df[f"{col1}_{col2}"].astype('category')
        
        return df
            
    def _add_cumulative_features(self, df:pd.DataFrame):        

        id_cols = ["AccountId", "CUSTOMER_ID"]
        if self.col_uid_name in df.columns:
            id_cols.append(self.col_uid_name)
        
        id_cols = set(id_cols)
        
        for col in ['TX_AMOUNT','Value']:
            for group_key in id_cols:
                df[f"{group_key}_TimeSinceLastTxn_Cumsum"] = df.groupby(group_key)[f"{group_key}_TimeSinceLastTxn"].cumsum()  / 60
                df[f"{col}_Cumsum_{group_key}"] = df.groupby(group_key)[col].cumsum()
        
        return df
    
    def _add_lombscargle_features(self,df:pd.DataFrame,value_col:str='TX_AMOUNT'):

        features = []
        freqs = np.linspace(1 / self.max_period, 1 / self.min_period, self.n_freqs)

        for _, group in df.groupby(self.col_uid_name):
            t = group['TX_DATETIME'].diff().dt.total_seconds().fillna(0) / (60 * 60)  # convert to hours
            y = group[value_col].values

            if len(t) < 3:  # not enough data
                features.append(np.zeros(len(freqs)))
                continue

            # Normalize time to start at 0
            t = t - t.min()
            ls = LombScargle(t, y)
            power = ls.power(freqs)

            features.append(power)
        
        features = np.array(features)
        feature_names = [f"{value_col}_LombScargle_{i}" for i in range(len(freqs))]
        features_df = pd.DataFrame(features, columns=feature_names)

        return pd.concat([df, features_df], axis=1)

    def _add_deltas(self, df:pd.DataFrame):

        id_cols = ["AccountId", self.col_uid_name]

        for group_key in set(id_cols):
            for col in ['TX_AMOUNT','Value']:
                df[col + "_diff_to_last_transaction"] = (
                        df.groupby(group_key)[col].diff().fillna(0)
                    )
        
        return df

    def _add_customer_clusters(self, df):
        
        # transform categorical values
        cluster_data = self._get_cluster_data(df)
        cluster_data = self._cat_encoder.transform(cluster_data)
        cluster_data.reset_index()
        
        # get clusster label
        labels_ = self._birch.predict(cluster_data)
        
        cluster_data['CustomerCluster'] = labels_
        
        df = df.merge(cluster_data, on=self.col_uid_name, how="left")

        return df
    
    def _compute_client_value_counts(self,
        df: pd.DataFrame,
        group_column: str,
        columns_to_analyze: list[str]
    ) -> pd.DataFrame:
        """Compute value counts for multiple columns per client group in one operation.
        
        Uses a single groupby+melt operation instead of iterative groupbys for efficiency.

        Args:
            df: Input DataFrame containing client data.
            group_column: Column name to group by (e.g., client ID).
            columns_to_analyze: List of columns to compute value counts for.

        Returns:
            DataFrame with tidy format containing:
            - Group column
            - Original column name
            - Observed value
            - Count of occurrences

        Example:
            >>> df = pd.DataFrame({
            ...     'ClientId': [1,1,2,2],
            ...     'ChannelId': ['A','A','B','B'],
            ...     'productcategory': ['X','Y','X','Y']
            ... })
            >>> compute_client_value_counts(df, 'ClientId', ['ChannelId', 'productcategory'])
            ClientId          column       value  counts
            0         1       ChannelId          A       2
            1         1  productcategory         X       1
            2         1  productcategory         Y       1
            3         2       ChannelId          B       2
            4         2  productcategory         X       1
            5         2  productcategory         Y       1
        """
        # Reshape data to long format for batch processing
        melted_df = df.melt(
            id_vars=[group_column],
            value_vars=columns_to_analyze,
            var_name="column",
            value_name="value"
        )
        
        # Single groupby operation for all columns
        melted_df = (melted_df
                        .groupby([group_column, "column", "value"])
                        .size()
                        .reset_index(name="counts")
                        )
        
        # merge results in original df
        wide_format = melted_df.pivot(
            index=self.col_uid_name,
            columns=['column', 'value'],
            values='counts'
        ).fillna(0)
        
        wide_format.columns = [f"{col}_perClient_{val}" for col, val in wide_format.columns]
        df_merged = df.merge(wide_format, on=self.col_uid_name, how='left')
        
        
        return df_merged
     
    def _get_cluster_data(self,df):
        
        # customer profile identificators
        agg = {"TX_AMOUNT": "mean",
                "Value": "mean",
                f"{self.col_uid_name}_TimeSinceLastTxn":'mean',
                'DailyClientTxnCount': 'mean',
                "Hour":lambda x: x.mode().sort_values().iloc[0],
                "DayOfWeek": lambda x: x.mode().sort_values().iloc[0],
                "IsWeekend": 'mean',
                f"{self.col_uid_name}_Txn6hCount":'mean',
                f"{self.col_uid_name}_Txn1dayCount":'mean',
                f"{self.col_uid_name}_Txn3dayCount":'mean',
                f"{self.col_uid_name}_Txn7dayCount":'mean',
                "IsNight":'mean',
                f"{self.col_uid_name}_MovingAvg5": "mean",
                "ChannelId": lambda x: x.mode().sort_values().iloc[0]
                if not x.mode().empty
                else np.nan,
                "ProductId": lambda x: x.mode().sort_values().iloc[0]
                if not x.mode().empty
                else np.nan,
                "ProductCategory": lambda x: x.mode().sort_values().iloc[0]
                if not x.mode().empty
                else np.nan,
                "ProviderId": lambda x: x.mode().sort_values().iloc[0]
                if not x.mode().empty
                else np.nan,
                "PricingStrategy": lambda x: x.mode().sort_values().iloc[0]
                if not x.mode().empty
                else np.nan,
            }
        
        for value_col in ['TX_AMOUNT','Value']:
            agg[self.col_uid_name + f"_{value_col}_RatioTo7dAvg"] = 'mean'
            agg[self.col_uid_name +  f"_{value_col}_RatioTo30dAvg"] = 'mean'
            agg[self.col_uid_name +  f"_{value_col}_ZScore_7d"] = 'mean'
            agg[self.col_uid_name + f"_{value_col}_ZScore_30d"] = 'mean'

        # -- compute aggregations
        cluster_data = (
            df.groupby(self.col_uid_name)
            .agg(agg)
        )

        cluster_data.columns = [f"cluster_{col}" if not col.startswith('cluster_') else col 
                          for col in cluster_data.columns]
                
        return cluster_data.convert_dtypes()
               
    def _compute_clusters_customers(self, df) -> None:
        
        df = df.sort_values(['TX_DATETIME'])
        
        # encode categorical data
        cluster_data = self._get_cluster_data(df)
        cluster_data = self._cat_encoder.fit_transform(X=cluster_data,y=None)
        
        # compute clusters
        self._birch = Birch(n_clusters=self.n_clusters,
                            threshold=0.5,
                            branching_factor=50)
        self._birch.fit(cluster_data)

        return None

    
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
                                                            verbose_feature_names_out=True)
        
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
    onehot_threshold=6,
    session_gap_minutes=60 * 24,
    uid_cols=[
        None,
    ],
    uid_col_name="CustomerUID",
    add_fraud_rate_features: bool = False,
    reorder_by=['TX_DATETIME','AccountId'],
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
    add_poly_interactions=False,
    interaction_cat_cols=None,
    add_cum_features=True,
    poly_degree=2,
    poly_cat_encoder_name="binary",
    imputer_n_neighbors=9,
    add_fft=False,
    add_seasonal_features=False,
    use_nystrom=False,
    use_sincos=True,
    use_spline=False,
    spline_degree=3,
    spline_n_knots=6,
    nystroem_kernel="poly",
    nystroem_components=50,
    add_imputer=False,
    rfe_step=3,
    n_clusters=0,
    k_score_func=mutual_info_classif,
    do_pca=False,
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
        add_fft=add_fft,
        add_seasonal_features=add_seasonal_features,
        use_sincos=use_sincos,
        use_spline=use_spline,
        spline_degree=spline_degree,
        spline_n_knots=spline_n_knots,
        behavioral_drift_cols=behavioral_drift_cols,
        uid_col_name=uid_col_name,  # name given to uid cols created from interactions of uid_cols
    )

    workflow_steps = [('feature_engineer', feature_engineer)]

    # advanced feature engineer
    if add_cum_features or n_clusters>0 :
        feature_engineer_2 = AdvancedFeatureEngineer(min_period=6, # 6 hours
                                                    max_period = 24*3, # 1 day
                                                    col_uid_name=uid_col_name,
                                                    uid_cols=uid_cols,
                                                    add_cum_features=add_cum_features,
                                                    n_clusters=n_clusters,
                                                    add_lombscargle_features=False,
                                                    add_amount_value_ratio=True,
                                                    n_freqs=5
                                                )   
        workflow_steps.append(('feature_engineer_2', feature_engineer_2))

    # drop constant features and cols_to_drop
    workflow_steps.append(('dropper',DropFeatures(cols_to_drop)))
    workflow_steps.append(('drop_constant',DropConstantFeatures(tol=1.0,missing_values='ignore')))
    workflow_steps.append(('drop_dupplicates',DropDuplicateFeatures()))

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

    
    # --------------------- DImension Reduction ---------------------
    
    # add pca
    if do_pca:
        pca = PCA(n_components=pca_n_components, random_state=42)
        workflow_steps.append(("dim_reduce", pca))
    
    # select features
    if str(feature_selector_name) != 'None':
        select_features = load_feature_selector(
                                        n_features_to_select=seq_n_features_to_select,
                                        cv = TimeSeriesSplit(n_splits=n_splits, gap=cv_gap),
                                        estimator=feature_select_estimator,
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

        workflow_steps.append(('to_df_0',ToDataframe()))
        workflow_steps.append(("feature_selector", select_features))
    
    # reduce dimension using Nystrom
    if use_nystrom:
        nystrom = Nystroem(
            kernel=nystroem_kernel,
            degree=2,
            n_components=nystroem_components,
        )
        workflow_steps.append(('nystrom',nystrom))
    
    # ------------------------------ Add outlier scores
    
    if detector_list is not None:
        pyod_det = OutlierDetector(detector_list=detector_list)
        outlier_scores = [("outlier_scores", pyod_det)]
        
        outlier_scores = FeatureUnion(
            transformer_list=outlier_scores, 
        )
    
        workflow_steps.append(('to_df_1',ToDataframe()))
        workflow_steps.append(('outlier_scores',outlier_scores))    
    
    # ------------------------ Add classifer
    if classifier is not None:
        to_df = ToDataframe()
        workflow_steps = workflow_steps + [('to_df_2', to_df), ('model', classifier)]
    
    return Pipeline(steps=workflow_steps)


# AdvancedFeatures = DimensionReduction