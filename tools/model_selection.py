# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 16:51:56 2025

@author: FADELCO
"""

# %% funcs & imports
from collections import OrderedDict
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
import joblib
import json
import optuna
from datetime import datetime,date
from optuna.samplers import TPESampler
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection._search import BaseSearchCV
from fraudetect.modeling.utils import hyperparameter_tuning
from fraudetect.dataset import data_loader
from fraudetect.config import (
    COLUMNS_TO_DROP,
    COLUMNS_TO_ONE_HOT_ENCODE,
    COLUMNS_TO_CAT_ENCODE,
    COLUMNS_TO_STD_SCALE,
    COLUMNS_TO_ROBUST_SCALE,
    Arguments
)
from fraudetect.features import load_transforms_pyod,build_encoder_scalers
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
    sampler_names: list[str] = None,
    sampler_cfgs: list[dict] = None,
):
    # load and transform
    (X_train, y_train), prequential_split_indices, col_transformer = data_loader(
        kwargs_tranform_data=kwargs_tranform_data,
        data_path=data_path,
        split_method="prequential",
        delta_train=delta_train,
        delta_delay=delta_delay,
        delta_test=delta_test,
        n_folds=1,# matters if prequential_split_indices are used
        random_state=random_state,
        sampling_ratio=1.0,
    )
    print("Raw data shape: ", X_train.shape, y_train.shape)

    # Re-sample data
    if (sampler_names is not None) and (sampler_cfgs is not None):
        X_train, y_train = data_resampling(
            X=X_train, y=y_train, sampler_names=sampler_names, sampler_cfgs=sampler_cfgs
        )
        print("Resampled data shape: ", X_train.shape, y_train.shape)

    return X_train, y_train, col_transformer

def build_dataset(args: Arguments, verbose=0):
    
    if args.cat_encoding_method == 'hashing':
        kwargs = dict(n_components=args.cat_encoding_hash_n_components,
                      hash_method=args.cat_encoding_hash_method
                      )
    elif args.cat_encoding_method == 'base_n':
        kwargs = dict(base=args.cat_encoding_base_n,
                      )
    else:
        kwargs = dict()

    # load data & do preprocessing
    col_transformer = build_encoder_scalers(cols_onehot=COLUMNS_TO_ONE_HOT_ENCODE,
                                            cols_cat_encode=COLUMNS_TO_CAT_ENCODE,
                                            cols_std=COLUMNS_TO_STD_SCALE,
                                            cols_robust=COLUMNS_TO_ROBUST_SCALE,
                                            cat_encoding_method=args.cat_encoding_method,
                                            add_imputer=args.add_imputer,
                                            verbose=bool(verbose),
                                            add_concat_features_transform=args.concat_features is not None,
                                            n_jobs=args.n_jobs,
                                            concat_features_encoding_kwargs=args.concat_features_encoding_kwargs,
                                            **kwargs
                                            )
    kwargs_tranform_data = dict(
        col_transformer=col_transformer,
        cols_to_drop=COLUMNS_TO_DROP,
        windows_size_in_days=args.windows_size_in_days,
        train_transform=None,  # some custom transform applied to X_train,y_train
        val_transform=None,  # some custom transform applied to X_val,y_val
        delay_period_accountid=args.delta_delay,
        concat_features=args.concat_features
    )
    X_train, y_train, col_transformer = __prepare_data(
                                        data_path=args.data_path,
                                        kwargs_tranform_data=kwargs_tranform_data,
                                        delta_train=args.delta_train,
                                        delta_delay=args.delta_delay,
                                        delta_test=args.delta_test,
                                        random_state=args.random_state,
                                        sampler_names=args.sampler_names,
                                        sampler_cfgs=args.sampler_cfgs,
                                    )
    
    # transformed column names
    # columns_of_transformed_data = list(map(lambda name: name.split('__')[1],
    #                                         list(col_transformer.get_feature_names_out()))
    #                                     )
    columns_of_transformed_data = col_transformer.get_feature_names_out()
    df_train_preprocessed = pd.DataFrame(X_train, columns=columns_of_transformed_data)
    
    print("X_train_preprocessed columns: ", df_train_preprocessed.columns)
    
    # load pyod transform and apply it to X_train
    transform_pyod = None
    fitted_models_pyod = None
    if args.outliers_det_configs is not None:
        transform_pyod, fitted_models_pyod = load_transforms_pyod(
            X_train=X_train,
            outliers_det_configs=args.outliers_det_configs,
            method=args.pyod_predict_proba_method,
            add_confidence=False,
            fitted_detector_list=None,
            return_fitted_models=True
        )
        X_train = transform_pyod(X=X_train)
    
    return (X_train, y_train), col_transformer, fitted_models_pyod

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

    best_results = dict()
    for model_name in models_config.keys():
        try:
            print(f"\nHyperparameter tuning for model: {model_name}")

            search_engine = hyperparameter_tuning(
                cv=cv,
                config=models_config,
                X_train=X_train,
                y_train=y_train,
                model_name=model_name,
                scoring=scoring,
                method=method,  # other, gridsearch
                verbose=verbose,
                n_iter=n_iter,
                n_jobs=n_jobs,
            )
            best_results[model_name] = search_engine

            if verbose:
                score = search_engine.best_score_
                print(f"mean {scoring} score for best_estimator: {score:.4f}")
                print("best params: ", search_engine.best_params_)

        except Exception as e:
            print(e)
            continue

    return best_results

def load_models_cfg(names: list[str]):
    return {name: HYP_CONFIGS.models[name] for name in names}


def run(args: Arguments, X_train, y_train, save_path: str = None, verbose=0):
    
    # tune models
    best_results = tune_models_hyp(
        X_train,
        y_train,
        models_config=load_models_cfg(names=args.model_names),
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


def demo(args: Arguments,verbose=0):
    # sample cfg randomly for debugging
    if args.sampler_names is not None:
        args.sampler_cfgs = get_samplers_cfgs(args.sampler_names, HYP_CONFIGS)

    if args.disable_pyod_outliers:
        args.outliers_det_configs = None
    else:
        args.outliers_det_configs = get_pyod_cfgs(args.pyod_detectors, HYP_CONFIGS)

    # run hyperparameter search
    (X_train, y_train), col_transformer, fitted_models_pyod = build_dataset(args=args,verbose=verbose)
    results = run(args=args,
                  X_train=X_train,
                  y_train=y_train,
                  save_path=None,
                  verbose=verbose)

    return results


class Objective(object):

    def __init__(self,
                 args:Arguments,
                 disable_samplers:bool=True,
                 verbose:int=0
                 ):

        self.args = args
        self.pyod_detectors = deepcopy(args.pyod_detectors)
        self.model_names = deepcopy(args.model_names)
        self.disable_pyod = deepcopy(args.disable_pyod_outliers)
        self.disable_samplers = disable_samplers
        self.verbose=verbose
        self.records = []
        self.count_iter = 0
        
        (self.X_train, self.y_train), self.col_transformer, self.fitted_models_pyod = build_dataset(args=args,verbose=verbose)
    
    def sample_cfg_optuna(self, trial, name: str, cfg: dict):
        for k in cfg.keys():
            if not isinstance(cfg[k], Iterable):
                continue
            cfg[k] = trial.suggest_categorical(name + "__" + k, cfg[k])
        return cfg
          
    def __call__(self, trial):
        
        self.count_iter += 1

        # select model
        model_name = trial.suggest_categorical(
            "classifier",
            self.model_names,  # HYP_CONFIGS.models.keys()
        )
        self.args.model_names = [
            model_name,
        ]

        # select outlier detector for data aug
        self.args.disable_pyod_outliers = trial.suggest_categorical("disable_pyod", [True, self.disable_pyod])
        if self.args.disable_pyod_outliers:
            self.args.outliers_det_configs = None
        else:
            pyod_detector_name = trial.suggest_categorical("pyod_det", self.pyod_detectors)
            self.args.pyod_detectors = [
                pyod_detector_name,
            ]
            self.args.outliers_det_configs = OrderedDict()
            self.args.outliers_det_configs[pyod_detector_name] = self.sample_cfg_optuna(trial,
                pyod_detector_name, HYP_CONFIGS.outliers_detectors[pyod_detector_name]
            )

        # samplers
        self.args.sampler_cfgs = None
        self.args.sampler_names = None
        if not trial.suggest_categorical("disable_samplers", 
                                         [True, self.disable_samplers]):
            conbimed_sampler = trial.suggest_categorical(
                "conbined_sampler", HYP_CONFIGS.combinedsamplers + [None,]*len(HYP_CONFIGS.combinedsamplers)
            )
            sampler_names = [conbimed_sampler]
            if conbimed_sampler is None:
                oversampler = trial.suggest_categorical(
                    "over_sampler", HYP_CONFIGS.oversamplers
                )
                undersampler = trial.suggest_categorical(
                    "under_sampler", HYP_CONFIGS.undersamplers
                )
                sampler_names = [oversampler, undersampler]

            # get sampler config
            self.args.sampler_cfgs = []
            for name in sampler_names:
                # if name is not None:
                _cfg = {name: self.sample_cfg_optuna(trial, name, HYP_CONFIGS.samplers[name])}
                self.args.sampler_cfgs.append(_cfg)

        # run cv and record
        try:
            results = run(args=self.args,
                          X_train=self.X_train,
                          y_train=self.y_train,
                          save_path=None,
                          verbose=self.verbose)
            score = results[model_name].best_score_
            self.records.append(results)
        except Exception as e:
            print(e)
            print(results,'\n\n')
            score = 0.

        return score


#%%main
if __name__ == "__main__":
    
    #Running
    args = Arguments()
    args.data_path = r"D:\fraud-detection-galsen\data\training.csv"
    
    args.model_names = ('randomForest',) #'gradientBoosting','randomForest',)
    
    current_time = datetime.now().strftime("%H-%M")
    
    args.study_name = "small-models" 
    
    args.cv_n_iter = 100
    args.scoring = 'f1'
    args.cv_method =  'optuna'
    args.cv_gap = 1051*5
    args.n_splits = 5
    args.n_jobs = 8
    args.delta_train=50
    args.delta_delay=7
    args.delta_test=20
    
    args.random_state = 41
    
    args.disable_pyod_outliers = True
    args.pyod_detectors = ('iforest', 'cblof', 'loda', 'knn')
    args.pyod_predict_proba_method = 'linear'
    
    args.sampler_names = None # ['SMOTE','nearmiss']
    args.sampler_cfgs = None
    
    args.concat_features=('AccountId', 'CUSTOMER_ID')
    args.concat_features_encoding_kwargs=dict(cat_encoding_method='hashing',
                                         n_components=14
                                         )
    
    args.add_imputer = False
    
    args.cat_encoding_method='binary'
    args.cat_encoding_hash_n_components=8 # if cat_encoding_method='hashing'
    args.cat_encoding_base_n=4 # if cat_encoding_method=base_n
    args.windows_size_in_days = (1, 7, 30)
     
    # for debugging
    # demo(args=args)
    
    workdir = Path(r"D:\fraud-detection-galsen\runs-optuna")
    workdir.mkdir(parents=True,exist_ok=True)   

    # using optuna
    study_name = args.study_name + f"_{str(date.today())}_{current_time}" 
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(multivariate=True, group=True),
        study_name=study_name,
        load_if_exists=True,
        storage="sqlite:///hypsearch.sql",
    )
    
    objective_optuna = Objective(args=args,
                                 disable_samplers=True,
                                 verbose=0
                                 )
    study.optimize(objective_optuna,
                   n_trials=100,
                   n_jobs=8,
                   show_progress_bar=True,
                   timeout=60*60*3)
    print(study.best_trial)
    
    # save args
    with open(workdir/(study_name+'.json'),'w') as file:
        cfg = args.__dict__
        json.dump(cfg, file,indent=4)
    
    # Save object
    filename = workdir / study_name
    if filename.exists():
        filename.with_stem(filename.stem + '_1')
    joblib.dump(objective_optuna, filename)
