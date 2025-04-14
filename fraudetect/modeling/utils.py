from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
)
from copy import deepcopy
from itertools import combinations
from sklearn.model_selection._search import BaseSearchCV
from hpsklearn import HyperoptEstimator
from hyperopt import tpe
import json, os, joblib, traceback
import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from collections import OrderedDict
import random
from collections.abc import Iterable
import optuna
from optuna.samplers import TPESampler
from fraudetect import import_from_path
from ..config import Arguments
from ..dataset import MyDatamodule
from ..preprocessing import FeatureEncoding, FraudFeatureEngineer
from ..preprocessing import feature_selector


def evaluate(classifier, X, y):
    metrics = ["accuracy", "f1", "average_precision", "precision", "recall"]
    for metric in metrics:
        scorer = get_scorer(metric)
        score = scorer(classifier, X, y)
        print(f"{metric}: {score:.4f}")


def get_model(model_name: str, config: dict) -> dict:
    """
    Get the model configuration based on the model name.
    """
    if model_name not in config:
        raise ValueError(f"Model {model_name} not found in config.")

    model_cfg = config[model_name].copy()

    # Remove the model from the config dictionary
    model = model_cfg.pop("model")

    return model, model_cfg


def sample_model_cfg(model_cfg: dict) -> dict:
    """
    Sample a model configuration from the given model configuration dictionary.
    """
    sampled_cfg = {}
    for key, value in model_cfg.items():
        if isinstance(value, Iterable):
            sampled_cfg[key] = random.choice(value)
        else:
            sampled_cfg[key] = value
    return sampled_cfg


def instantiate_model(model, **kwargs):
    """
    Instantiate the model with the given parameters.
    """
    return model(**kwargs)


def hyperparameter_tuning(
    cv,
    params_config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model,
    scoring: str = "f1",
    n_iter: int = 50,
    n_jobs: int = 8,
    verbose: int = 0,
    method: str = "optuna",
) -> BaseSearchCV:
    if verbose > 0:
        print(json.dumps(params_config, indent=4))

    if method == "gridsearch":
        search_engine = GridSearchCV(
            model,
            param_grid=params_config,
            scoring=scoring,
            cv=cv,
            refit=True,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    elif method == "random":
        search_engine = RandomizedSearchCV(
            model,
            param_distributions=params_config,
            scoring=scoring,
            cv=cv,
            refit=True,
            n_jobs=n_jobs,
            n_iter=n_iter,
            random_state=41,
            verbose=verbose,
        )

    elif method == "optuna":
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(multivariate=True, group=True),
            load_if_exists=True,
        )
        param_dist = dict()
        for k, v in params_config.items():
            if isinstance(v, Iterable):
                param_dist[k] = optuna.distributions.CategoricalDistribution(v)
            else:
                param_dist[k] = optuna.distributions.CategoricalDistribution(
                    [
                        v,
                    ]
                )

        search_engine = optuna.integration.OptunaSearchCV(
            model,
            param_distributions=param_dist,
            cv=cv,
            refit=True,
            n_jobs=n_jobs,
            study=study,
            scoring=scoring,
            # error_score='raise',
            max_iter=300,
            timeout=60 * 3,
            n_trials=n_iter,
            random_state=41,
            verbose=verbose,
        )

    elif method == "hyperopt":
        loss_fn = lambda y_true, y_pred: 1.0 - f1_score(
            y_true=y_true,
            y_pred=y_pred,
            pos_label=1,
            zero_division=1,
        )
        search_engine = HyperoptEstimator(
            classifier=model,
            algo=tpe.suggest,
            max_evals=n_iter,
            loss_fn=loss_fn,
            n_jobs=n_jobs,
            trial_timeout=60 * 2,
        )
    else:
        raise ValueError(
            "Invalid method. Choose either 'gridsearch',hyperopt, 'optuna' or 'random'."
        )

    search_engine.fit(X_train, y_train)

    return search_engine


# show result
def results_from_search(search_engine, performance_metrics_list=["score"]):
    performances_df = pd.DataFrame()
    expe_type = "val"
    performance_metrics_list_grid = performance_metrics_list

    for i in range(len(performance_metrics_list_grid)):
        performances_df[performance_metrics_list[i] + " " + expe_type] = (
            search_engine.cv_results_["mean_test_" + performance_metrics_list_grid[i]]
        )
        performances_df[performance_metrics_list[i] + " " + expe_type + " Std"] = (
            search_engine.cv_results_["std_test_" + performance_metrics_list_grid[i]]
        )
    performances_df["Execution time"] = search_engine.cv_results_["mean_fit_time"]
    performances_df["Parameters"] = list(search_engine.cv_results_["params"])

    return performances_df


def _tune_models_hyp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    models_config: dict,
    n_splits: int = 5,
    gap: int = 1051 * 5,
    n_iter: int = 100,
    scoring: str = "f1",
    verbose: int = 0,
    n_jobs: int = 8,
    method: str = "random",
) -> dict[str, BaseSearchCV]:
    # 5 days, 1051 = average number of transactions per day

    assert isinstance(scoring, str), "It should be a string."

    cv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    model_name = list(models_config.keys())[0]
    print(f"\nHyperparameter tuning for model: {model_name}")

    model, params_config = get_model(model_name, models_config)

    search_engine = hyperparameter_tuning(
        cv=cv,
        params_config=params_config,
        X_train=X_train,
        y_train=y_train,
        model=model(),
        scoring=scoring,
        method=method,  # other, gridsearch
        verbose=verbose,
        n_iter=n_iter,
        n_jobs=n_jobs,
    )

    if verbose:
        score = search_engine.best_score_
        print(f"mean {scoring} score for best_estimator: {score:.4f}")
        print("best params: ", search_engine.best_params_)

    return {model_name: search_engine}  #


def _run(
    args: Arguments,
    models_config: dict,
    X_train,
    y_train,
    save_path: str = None,
    verbose=0,
) -> dict[str, BaseSearchCV]:
    if np.any(np.isnan(X_train)):
        raise ValueError("There are NaN values in X_train.")

    assert len(models_config) == 1, "Provide only one model in models_config"

    # tune models
    best_results = _tune_models_hyp(
        X_train,
        y_train,
        models_config=models_config,
        n_splits=args.n_splits,
        gap=args.cv_gap,
        n_iter=args.cv_n_iter,
        scoring=args.scoring,
        verbose=verbose,
        n_jobs=args.n_jobs,
        method=args.cv_method,
    )

    # save results
    if save_path:
        joblib.dump(best_results, save_path)

    return best_results


class Tuner(object):
    def __init__(
        self, args: Arguments, 
        verbose: int = 0, 
        cat_encoding_kwards: dict = {},
        feature_selector_kwargs:dict={}
    ):
        self.HYP_CONFIGS = None

        self.args = deepcopy(args)
        self.pyod_detectors = deepcopy(sorted(self.args.pyod_detectors))
        self.pyod_choices = [
            json.dumps(list(k)) for k in combinations(self.pyod_detectors, 4)
        ]
        self.model_names = deepcopy(args.model_names)
        self.disable_pyod = deepcopy(args.disable_pyod_outliers)
        self.disable_samplers = deepcopy(args.disable_samplers)
        self.verbose = verbose
        self.count_iter = 0
        self.feature_selector_kwargs= feature_selector_kwargs
        self.selector = None

        self.datamodule = MyDatamodule()
        feature_engineer = FraudFeatureEngineer(
            windows_size_in_days=args.windows_size_in_days,
            uid_cols=args.concat_features,
            session_gap_minutes=args.session_gap_minutes,
            n_clusters=None,
        )

        encoder = FeatureEncoding(
            cat_encoding_method=args.cat_encoding_method,
            add_imputer=args.add_imputer,
            onehot_threshold=args.onehot_threshold,
            cols_to_drop=args.cols_to_drop,
            n_jobs=args.n_jobs,
            cat_encoding_kwards=cat_encoding_kwards,
        )

        self.datamodule.setup(encoder=encoder, feature_engineer=feature_engineer)

        self.X_train, self.y_train = self.datamodule.get_train_dataset(args.data_path)

        self.best_score = 0.0
        self.transform_pipeline = None
        self.ckpt_filename = os.path.join(
            args.work_dir, args.study_name + "_best-run.joblib"
        )

    def sample_cfg_optuna(self, trial, name: str, config: dict):
        _cfg = deepcopy(config)

        for k in _cfg.keys():
            if not isinstance(_cfg[k], Iterable):
                continue
            _cfg[k] = trial.suggest_categorical(name + "__" + k, _cfg[k])

        return _cfg

    def save_checkpoint(self, score: float, results: dict):
        if score >= self.best_score:
            self.best_score = score
            vals = [results, self.transform_pipeline, self.datamodule, self.selector]
            joblib.dump(vals, self.ckpt_filename)

    def load_hyp_conf(self, path_conf: str):
        try:
            self.HYP_CONFIGS = import_from_path("hyp_search_conf", path_conf)
        except:
            self.HYP_CONFIGS = import_from_path("hyp_search_conf", path_conf)

    def __call__(self, trial):
        self.count_iter += 1

        X = self.X_train.copy()
        y = self.y_train.copy()

        self.transform_pipeline = None
        self.selector = None
        pipe = []

        # select model
        model_name = trial.suggest_categorical(
            "classifier",
            self.model_names,
        )
        models_config = {model_name: self.HYP_CONFIGS.models[model_name]}

        # PCA
        do_pca = trial.suggest_categorical("pca", [False, self.args.do_pca])
        if do_pca:
            n_components = trial.suggest_int(
                "n_components", min(5, self.X_train.shape[1]), self.X_train.shape[1], 3
            )
            pca_transform = PCA(n_components=n_components)
            std_scaler = StandardScaler()
            pipe = pipe + [("scaler", std_scaler), ("pca", pca_transform)]

        if len(pipe) > 0:
            self.transform_pipeline = Pipeline(steps=pipe)
            X = self.transform_pipeline.fit_transform(X=X)

        # select outlier detector for data aug
        disable_pyod = trial.suggest_categorical(
            "disable_pyod", [True, self.disable_pyod]
        )
        if disable_pyod:
            outliers_det_configs = None
        else:
            _cfgs = list()
            pyod_choices = trial.suggest_categorical(
                "pyod_choices",
                self.pyod_choices,  # range(1,self.pyod_detectors+1)
            )
            pyod_choices = json.loads(pyod_choices)
            for name in self.pyod_detectors:
                _cfg = self.sample_cfg_optuna(
                    trial, name, self.HYP_CONFIGS.outliers_detectors[name]
                )
                if name in pyod_choices:
                    _cfgs.append(_cfg)
            outliers_det_configs = OrderedDict(zip(pyod_choices, _cfgs))

        # select samplers
        sampler_cfgs, sampler_names = None, None
        if not trial.suggest_categorical(
            "disable_samplers", [True, self.disable_samplers]
        ):
            conbimed_sampler = trial.suggest_categorical(
                "conbined_sampler", self.HYP_CONFIGS.combinedsamplers
            )
            sampler_names = [
                conbimed_sampler,
            ]

        # augment or resample data on the fly if 
        (X, y), fitted_models_pyod = self.datamodule.augment_resample_dataset(
            X=X,
            y=y,
            outliers_det_configs=outliers_det_configs,
            sampler_names=sampler_names,
            sampler_cfgs=sampler_cfgs,
            fitted_detector_list=None,
        )

        # feature selector:
        if trial.suggest_categorical(
            "select_features", [True, self.args.do_feature_selection]
        ):
            X, self.selector = feature_selector(X_train=X,y_train=y,
                            	cv=TimeSeriesSplit(n_splits=self.args.n_splits,gap=self.args.cv_gap),
                                **self.feature_selector_kwargs              
                        )

        # try:
        results = _run(
            args=self.args,
            models_config=models_config,
            X_train=X,
            y_train=y,
            save_path=None,
            verbose=self.verbose,
        )

        # try to get score
        try:
            score = results[model_name].best_score_
            results["fitted_models_pyod"] = fitted_models_pyod  # log pyod models
            results["samplers"] = (sampler_names, sampler_cfgs)
            self.save_checkpoint(score=score, results=results)

        except ValueError:
            traceback.print_exc()
            # print(results, "\n")
            score = 0

        return score
