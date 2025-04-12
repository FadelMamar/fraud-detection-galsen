# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 16:51:56 2025

@author: FADELCO
"""

# %% funcs & imports
from itertools import product, combinations
import json
import traceback
from collections import OrderedDict
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
import joblib
import json
import optuna
from hyperopt import tpe, hp, fmin
from datetime import datetime, date
from optuna.samplers import TPESampler
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection._search import BaseSearchCV
from fraudetect.modeling.utils import hyperparameter_tuning, get_model
from fraudetect.dataset import data_loader
from fraudetect.config import (
    COLUMNS_TO_DROP,
    COLUMNS_TO_ONE_HOT_ENCODE,
    COLUMNS_TO_CAT_ENCODE,
    COLUMNS_TO_STD_SCALE,
    COLUMNS_TO_ROBUST_SCALE,
    Arguments,
)
from fraudetect.features import load_transforms_pyod, build_encoder_scalers
from fraudetect.sampling import data_resampling
from fraudetect import import_from_path, sample_cfg


try:
    HYP_CONFIGS = import_from_path(
        "hyp_search_conf", r"D:\fraud-detection-galsen\tools\hyp_search_conf.py"
    )
except:
    HYP_CONFIGS = import_from_path(
        "hyp_search_conf", r"D:\fraud-detection-galsen\tools\hyp_search_conf.py"
    )


def __prepare_data(
    data_path: str,
    kwargs_tranform_data: dict,
    delta_train=40,
    delta_delay=7,
    delta_test=20,
    random_state=41,
):
    # load and transform
    (X_train, y_train), prequential_split_indices, col_transformer = data_loader(
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

    return X_train, y_train, col_transformer


def build_dataset(args: Arguments, verbose=0):
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
    col_transformer = build_encoder_scalers(
        cols_onehot=COLUMNS_TO_ONE_HOT_ENCODE,
        cols_cat_encode=COLUMNS_TO_CAT_ENCODE,
        cols_std=COLUMNS_TO_STD_SCALE,
        cols_robust=COLUMNS_TO_ROBUST_SCALE,
        cat_encoding_method=args.cat_encoding_method,
        add_imputer=args.add_imputer,
        verbose=bool(verbose),
        add_concat_features_transform=args.concat_features is not None,
        n_jobs=args.n_jobs,
        concat_features_encoding_kwargs=args.concat_features_encoding_kwargs,
        **kwargs,
    )
    kwargs_tranform_data = dict(
        col_transformer=col_transformer,
        cols_to_drop=COLUMNS_TO_DROP,
        windows_size_in_days=args.windows_size_in_days,
        train_transform=None,  # some custom transform applied to X_train,y_train
        val_transform=None,  # some custom transform applied to X_val,y_val
        delay_period_accountid=args.delta_delay,
        concat_features=args.concat_features,
    )
    X_train, y_train, col_transformer = __prepare_data(
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
    columns_of_transformed_data = col_transformer.get_feature_names_out()
    df_train_preprocessed = pd.DataFrame(X_train, columns=columns_of_transformed_data)

    print("X_train_preprocessed columns: ", df_train_preprocessed.columns)

    return (X_train, y_train), col_transformer


# functions to augment data
def __resample_data(
    X,
    y,
    sampler_names: list[str],
    sampler_cfgs: list[dict],
):
    assert len(sampler_names) == len(sampler_cfgs), "They should have the same length."

    # Re-sample data
    X, y = data_resampling(
        X=X, y=y, sampler_names=sampler_names, sampler_cfgs=sampler_cfgs
    )
    print("Resampled data shape: ", X.shape, y.shape)

    return X, y


def __concat_pyod_scores(
    X,
    outliers_det_configs: OrderedDict,
):
    # load pyod transform and apply it to X
    transform_pyod, fitted_models_pyod = load_transforms_pyod(
        X_train=X,
        outliers_det_configs=outliers_det_configs,
        fitted_detector_list=None,
        return_fitted_models=True,
    )

    return transform_pyod(X=X), fitted_models_pyod


def augment_resample_dataset(
    X,
    y,
    outliers_det_configs: OrderedDict | None,
    sampler_names: list[str] | None,
    sampler_cfgs: list[dict] | None,
):
    # augment data using outliers scores
    fitted_models_pyod = None
    if outliers_det_configs is not None:
        X, fitted_models_pyod = __concat_pyod_scores(
            X,
            outliers_det_configs=outliers_det_configs,
        )

    # resample data
    if (sampler_names is not None) and (sampler_cfgs is not None):
        X, y = __resample_data(
            X=X, y=y, sampler_names=sampler_names, sampler_cfgs=sampler_cfgs
        )

    return (X, y), fitted_models_pyod


def tune_models_hyp(
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

    # best_results = dict()
    # for model_name in models_config.keys():
    # try:
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

    # except Exception as e:
    #     print(e)
    #     traceback.print_exc()
    #     continue

    return {model_name:search_engine} #best_results


def run(args: Arguments,models_config:dict, X_train, y_train, save_path: str = None, verbose=0)->dict:


    if np.any(np.isnan(X_train)):
        raise ValueError("There are NaN values in X_train.")

    assert len(models_config)==1, "Provide only one model in models_config"
    

    # tune models
    best_results = tune_models_hyp(
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


def demo(args: Arguments, verbose=0):
    def get_samplers_cfgs(sampler_names, configs):
        sampler_cfgs = list()

        for name in sampler_names:
            cfg = configs.samplers[name]
            cfg = sample_cfg(cfg)  # random sampling
            sampler_cfgs.append({name: cfg})

        return sampler_cfgs

    def get_pyod_cfgs(names, configs):
        # sample cfg randomly for debugging
        outliers_det_configs = []
        for name in sorted(names):
            cfg = configs.outliers_detectors[name]
            cfg = sample_cfg(cfg)
            outliers_det_configs.append((name, cfg))

        return OrderedDict(outliers_det_configs)

    # sample cfg randomly for debugging
    if args.sampler_names is not None:
        args.sampler_cfgs = get_samplers_cfgs(args.sampler_names, HYP_CONFIGS)

    if args.disable_pyod_outliers:
        args.outliers_det_configs = None
    else:
        args.outliers_det_configs = get_pyod_cfgs(args.pyod_detectors, HYP_CONFIGS)

    # run hyperparameter search
    (X_train, y_train), col_transformer = build_dataset(args=args, verbose=verbose)
    results = run(
        args=args, X_train=X_train, y_train=y_train, save_path=None, verbose=verbose
    )

    return results


class Objective(object):
    def __init__(self, args: Arguments, verbose: int = 0):
        self.args = deepcopy(args)
        self.pyod_detectors = deepcopy(sorted(self.args.pyod_detectors))
        self.pyod_choices = [json.dumps(list(k)) for k in combinations(["iforest", "cblof", "loda", "knn", "DIF", "abod", "hbos"], 4)]
        self.model_names = deepcopy(args.model_names)
        self.disable_pyod = deepcopy(args.disable_pyod_outliers)
        self.disable_samplers = deepcopy(args.disable_samplers)
        self.verbose = verbose
        self.records = []
        self.count_iter = 0

        (self.X_train, self.y_train), self.col_transformer = build_dataset(
            args=args, verbose=verbose
        )

    def sample_cfg_optuna(self, trial, name: str, config: dict):

        _cfg = deepcopy(config)

        for k in _cfg.keys():
            if not isinstance(_cfg[k], Iterable):
                continue
            _cfg[k] = trial.suggest_categorical(name + "__" + k, _cfg[k])

        return _cfg

    def __call__(self, trial):
        self.count_iter += 1

        # select model
        model_name = trial.suggest_categorical(
            "classifier",
            self.model_names,  # HYP_CONFIGS.models.keys()
        )
        # self.args.model_names = [
        #     model_name,
        # ]
        models_config = {model_name: HYP_CONFIGS.models[model_name]}

        # TODO: sampling outlier detectors does not work!!!
        # select outlier detector for data aug
        disable_pyod = trial.suggest_categorical(
            "disable_pyod", [True, self.disable_pyod]
        )
        if disable_pyod:
            outliers_det_configs = None
        else:
            _cfgs = list()
            pyod_choices = trial.suggest_categorical(
                    "pyod_choices", self.pyod_choices #range(1,self.pyod_detectors+1)
                )
            pyod_choices = json.loads(pyod_choices)
            for name in self.pyod_detectors:
                _cfg = self.sample_cfg_optuna(
                    trial, name, HYP_CONFIGS.outliers_detectors[name]
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
                "conbined_sampler",
                HYP_CONFIGS.combinedsamplers
                + [
                    None,
                ]
                * len(HYP_CONFIGS.combinedsamplers),
            )
            sampler_names = [
                conbimed_sampler,
            ]

            if conbimed_sampler is None:
                oversampler = trial.suggest_categorical(
                    "over_sampler",
                    HYP_CONFIGS.oversamplers
                    + [
                        None,
                    ],  # over_sampling is disabled when None is selected
                )
                undersampler = trial.suggest_categorical(
                    "under_sampler",
                    HYP_CONFIGS.undersamplers,  # always done!
                )
                sampler_names = [oversampler, undersampler]
                sampler_names = [k for k in sampler_names if k is not None]

            # get samplers' config
            sampler_cfgs = []
            for name in sampler_names:
                # if name is not None:
                _cfg = {
                    name: self.sample_cfg_optuna(
                        trial, name, HYP_CONFIGS.samplers[name]
                    )
                }
                sampler_cfgs.append(_cfg)

        # augment and resample data on the fly
        (X_train, y_train), fitted_models_pyod = augment_resample_dataset(
            X=self.X_train.copy(),
            y=self.y_train.copy(),
            outliers_det_configs=outliers_det_configs,
            sampler_names=sampler_names,
            sampler_cfgs=sampler_cfgs,
        )

        # run cv and record
        # try:
        results = run(
            args=self.args,
            models_config=models_config,
            X_train=X_train,
            y_train=y_train,
            save_path=None,
            verbose=self.verbose,
        )
        try:
            score = results[model_name].best_score_
            results["fitted_models_pyod"] = fitted_models_pyod  # log pyod models
            self.records.append(results)
        except Exception as e:
            print("\nError happened in Objective.__call__")
            traceback.print_exc()
            # print(e)
            print(results,"\n")
            score = None

        return score



# %%main
if __name__ == "__main__":
    # Running
    args = Arguments()
    args.data_path = r"D:\fraud-detection-galsen\data\training.csv"

    args.model_names = (
        "decisionTree",
        # "svc",
        # "randomForest",
        "balancedRandomForest",
        # "gradientBoosting",
        # "histGradientBoosting",
        # "xgboost"
    ) 

    current_time = datetime.now().strftime("%H-%M")

    args.study_name = "tree-models"
    args.study_name = args.study_name + f"_{str(date.today())}_{current_time}"

    args.optuna_n_trials = 50

    args.cv_n_iter = 1000
    args.scoring = "f1" # 'f1', precision
    args.cv_method = "optuna"
    args.cv_gap = 1051 * 5
    args.n_splits = 5
    args.n_jobs = 8
    args.delta_train = 50
    args.delta_delay = 7
    args.delta_test = 20

    args.random_state = 41

    args.disable_pyod_outliers = False
    args.pyod_detectors = ["iforest", "cblof", "loda", "knn", "DIF", "abod", "hbos"]
    
    args.sampler_names = None 
    args.sampler_cfgs = None
    args.disable_samplers = True

    args.concat_features = None # ("AccountId", "CUSTOMER_ID")
    args.concat_features_encoding_kwargs = dict(
        cat_encoding_method="hashing", n_components=14
    )

    args.add_imputer = False

    args.cat_encoding_method = "binary"
    args.cat_encoding_hash_n_components = 8  # if cat_encoding_method='hashing'
    args.cat_encoding_base_n = 4  # if cat_encoding_method=base_n
    args.windows_size_in_days = (1, 7, 15, 30)

    # for debugging
    # demo(args=args)

    workdir = Path(r"D:\fraud-detection-galsen\runs-optuna")
    workdir.mkdir(parents=True, exist_ok=True)

    # using optuna
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(multivariate=True, group=True),
        study_name=args.study_name,
        load_if_exists=False,
        storage="sqlite:///hypsearch.sql",
    )

    # save args
    with open(workdir / (args.study_name + ".json"), "w") as file:
        cfg = args.__dict__
        cfg['cols_to_drop'] = COLUMNS_TO_DROP
        cfg['cols_one_hot'] = COLUMNS_TO_ONE_HOT_ENCODE
        cfg['cols_cat_encorde'] = COLUMNS_TO_CAT_ENCODE
        cfg['cols_to_std_Scale'] = COLUMNS_TO_STD_SCALE
        cfg['cols_to_robust_scale'] = COLUMNS_TO_ROBUST_SCALE                
        json.dump(cfg, file, indent=4)

    objective_optuna = Objective(args=args, verbose=0)
    study.optimize(
        objective_optuna,
        n_trials=args.optuna_n_trials,
        n_jobs=8,
        show_progress_bar=True,
        timeout=60 * 60 * 3,
    )
    print(study.best_trial)

    # Save object
    filename = workdir / (args.study_name + ".joblib")
    if filename.exists():
        filename.with_stem(filename.stem + "_1")
    joblib.dump(objective_optuna, filename)
