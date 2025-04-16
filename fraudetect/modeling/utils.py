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
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier
import random
from collections.abc import Iterable
import optuna
from optuna.samplers import TPESampler
from fraudetect import import_from_path
from ..config import Arguments
from ..dataset import load_data
from ..preprocessing import load_workflow, get_feature_selector
from ..detectors import get_detector, instantiate_detector

# try:
#     import fireducks.pandas as pd
#     # print('importing fireducks.pandas as pd')
# except:
#     import pandas as pd

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

    # elif method == "hyperopt":
    #     loss_fn = lambda y_true, y_pred: 1.0 - f1_score(
    #         y_true=y_true,
    #         y_pred=y_pred,
    #         pos_label=1,
    #         zero_division=1,
    #     )
    #     search_engine = HyperoptEstimator(
    #         classifier=model,
    #         algo=tpe.suggest,
    #         max_evals=n_iter,
    #         loss_fn=loss_fn,
    #         n_jobs=n_jobs,
    #         trial_timeout=60 * 2,
    #     )
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
    classifier,
    params_config,
    n_splits: int = 5,
    gap: int = 1051 * 5,
    n_iter: int = 100,
    scoring: str = "f1",
    verbose: int = 0,
    n_jobs: int = 8,
    method: str = "random",
) -> BaseSearchCV:
    # 5 days, 1051 = average number of transactions per day

    assert isinstance(scoring, str), "It should be a string."

    cv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    search_engine = hyperparameter_tuning(
        cv=cv,
        params_config=params_config,
        X_train=X_train,
        y_train=y_train,
        model=classifier,
        scoring=scoring,
        method=method,  # other, gridsearch
        verbose=verbose,
        n_iter=n_iter,
        n_jobs=n_jobs,
    )

    if verbose:
        score = search_engine.best_score_
        print(f"mean {scoring} score for best_estimator: {score:.4f}")
        # print("best params: ", search_engine.best_params_)

    return search_engine # {model_name: search_engine}  #


def _run(
    args: Arguments,
    classifier,
    params_config,
    X_train,
    y_train,
    save_path: str = None,
    verbose=0,
) -> BaseSearchCV:
    # if np.any(np.isnan(X_train)):
    #     raise ValueError("There are NaN values in X_train.")

    # assert len(models_config) == 1, "Provide only one model in models_config"

    # tune models
    best_results = _tune_models_hyp(
        X_train=X_train,
        y_train=y_train,
        classifier=classifier,
        params_config=params_config,
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
            json.dumps(list(k)) for k in combinations(self.pyod_detectors, min(4,len(args.pyod_detectors)))
        ]
        self.model_names = deepcopy(args.model_names)
        self.disable_pyod = deepcopy(args.disable_pyod_outliers)
        self.disable_samplers = deepcopy(args.disable_samplers)
        self.verbose = verbose
        self.count_iter = 0
        self.feature_selector_kwargs= feature_selector_kwargs
        self.cat_encoding_kwards=cat_encoding_kwards
        self.selector = None
                
        self.raw_data_train = load_data(args.data_path)
        # self.X_train = raw_data_train #.drop(columns=['TX_FRAUD'])
        self.y_train = self.raw_data_train['TX_FRAUD']
               
        self.best_score = 0.0
        self.best_records = dict()
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

    def save_checkpoint(self,model_name:str, score: float, results: BaseSearchCV):
        if score >= self.best_score:
            self.best_score = score
            vals = [results,]
            joblib.dump(vals, self.ckpt_filename)
            self.best_records[model_name]=results

    def load_hyp_conf(self, path_conf: str):
        try:
            self.HYP_CONFIGS = import_from_path("hyp_search_conf", path_conf)
        except:
            self.HYP_CONFIGS = import_from_path("hyp_search_conf", path_conf)

    def __call__(self, trial):
        self.count_iter += 1

        self.transform_pipeline = None
        self.selector = None
    
        # select model
        model_name = trial.suggest_categorical(
            "classifier",
            self.model_names,
        )
        models_config = {model_name: self.HYP_CONFIGS.models[model_name]}
        model, models_config = get_model(model_name, self.HYP_CONFIGS.models)
        model = instantiate_model(model,**sample_model_cfg(models_config)) # instantiate with        
                
        # PCA
        do_pca = trial.suggest_categorical("pca", [False, self.args.do_pca])
        pca_n_components = None
        if do_pca:
            pca_n_components = trial.suggest_int(
                "n_components", 5, 50, 5
            )
                   
        # select outlier detector for data aug
        disable_pyod = trial.suggest_categorical(
            "disable_pyod", [True, self.disable_pyod]
        )
        if disable_pyod:
            # outliers_det_configs = None
            detector_list = None
        else:
            # cfgs = list()
            pyod_choices = trial.suggest_categorical(
                "pyod_choices",
                self.pyod_choices,  # range(1,self.pyod_detectors+1)
            )
            pyod_choices = json.loads(pyod_choices)
            detector_list = []
            for det_name in self.pyod_detectors:
                cfg = self.sample_cfg_optuna(
                    trial, det_name, self.HYP_CONFIGS.outliers_detectors[det_name]
                )
                detector, cfg = get_detector(name=det_name, config={det_name:cfg})
                detector = instantiate_detector(detector, cfg)
                detector_list.append(detector)
        
        

        # feature selector:
        do_feature_selection =  trial.suggest_categorical("select_features", 
                                                          [False, self.args.do_feature_selection])
        selector_cfg = {}
        feature_selector_name = None
        if do_feature_selection:
            feature_selector_name = trial.suggest_categorical(
                "feature_selector_name", self.HYP_CONFIGS.feature_selector.keys())
            selector_cfg = self.sample_cfg_optuna(
                trial, feature_selector_name, self.HYP_CONFIGS.feature_selector[feature_selector_name]
            )
            _, selector_cfg = get_feature_selector(name=feature_selector_name,config={feature_selector_name:selector_cfg})
                      
        
        feature_select_estimator = DecisionTreeClassifier(max_depth=15,
                                           max_features='sqrt',
                                           random_state=41)
        
        data_processor = load_workflow(
                          cols_to_drop=self.args.cols_to_drop,
                          scoring=selector_cfg.get('scoring','f1'),
                          feature_select_estimator=feature_select_estimator,
                          rfe_step=selector_cfg.get('step',3),
                          pca_n_components=pca_n_components or 20,
                          detector_list=detector_list,
                          cv_gap=self.args.cv_gap,
                          n_splits=self.args.n_splits,
                          session_gap_minutes=self.args.session_gap_minutes,
                          uid_cols=self.args.concat_features,
                          feature_selector_name=feature_selector_name,
                          seq_n_features_to_select=selector_cfg.get('n_features_to_select',3),
                          windows_size_in_days=self.args.windows_size_in_days,
                          cat_encoding_method=self.args.cat_encoding_method,
                          imputer_n_neighbors=9,
                          n_clusters=8,
                          top_k_best=selector_cfg.get('k',10),
                          # k_score_func=selector_cfg.get('score_func',10),
                          do_pca=do_pca,
                          verbose=self.verbose,
                          n_jobs=1 #self.args.n_jobs
                          )
        
        if model_name == 'histGradientBoosting':
            model.set_params(categorical_features='from_dtype')
        elif model_name == 'catboost':
            raise NotImplementedError
            # model.set_params(cat_features=[])
            # see https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier#cat_features
        
        classifier = Pipeline(steps=[('data_processor',data_processor),
                               ('model',model)]
                            )
        params_config = {f"model__{k}":v for k,v in models_config.items() if len(v)>1}

        X = self.raw_data_train.copy()
        y = self.y_train.copy()
        results = _run(
            args=self.args,
            classifier=classifier,
            params_config=params_config,
            X_train=X,
            y_train=y,
            save_path=None,
            verbose=self.verbose,
        )

        # try to get score
        try:
            score = results.best_score_
            # results["fitted_models_pyod"] = fitted_models_pyod  # log pyod models
            # results["samplers"] = (sampler_names, sampler_cfgs)
            self.save_checkpoint(model_name=model_name,score=score, results=results)

        except ValueError:
            traceback.print_exc()
            # print(results, "\n")
            score = 0

        return score
