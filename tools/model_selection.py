# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 16:51:56 2025

@author: FADELCO
"""

# %% funcs & imports
import json
from collections import OrderedDict
from pathlib import Path
import os
import joblib
import json
import optuna
from datetime import datetime, date
from optuna.samplers import TPESampler
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection._search import BaseSearchCV
from fraudetect.modeling.utils import hyperparameter_tuning, get_model
from fraudetect.config import (
    COLUMNS_TO_DROP,
    COLUMNS_TO_ONE_HOT_ENCODE,
    COLUMNS_TO_CAT_ENCODE,
    COLUMNS_TO_STD_SCALE,
    COLUMNS_TO_ROBUST_SCALE,
    Arguments,
)
# from fraudetect.features import load_transforms_pyod, build_encoder_scalers
# from fraudetect.sampling import data_resampling
from fraudetect import import_from_path, sample_cfg
# from fraudetect.preprocessing import FraudFeatureEngineer, FeatureEncoding
# from fraudetect.dataset import load_data, MyDatamodule
# from sklearn.decomposition import PCA, KernelPCA
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.pipeline import Pipeline
from fraudetect.modeling.utils import Tuner

try:
    HYP_CONFIGS = import_from_path(
        "hyp_search_conf", r"D:\fraud-detection-galsen\tools\hyp_search_conf.py"
    )
except:
    HYP_CONFIGS = import_from_path(
        "hyp_search_conf", r"D:\fraud-detection-galsen\tools\hyp_search_conf.py"
    )


def __tune_models_hyp(
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

    return {model_name: search_engine}  # best_results


def run(
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
    best_results = __tune_models_hyp(
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


# %%main
if __name__ == "__main__":
    # Running
    args = Arguments()
    args.data_path = r"D:\fraud-detection-galsen\data\training.csv"

    args.model_names = (
        # "mlp",
        "decisionTree",
        "logisticReg",
        "svc",
        # "sgdClassifier",
        # "randomForest",
        # "balancedRandomForest",
        # "gradientBoosting",
        # "histGradientBoosting",
        # "xgboost",
    )

    current_time = datetime.now().strftime("%H-%M")

    args.study_name = "debug"
    args.study_name = args.study_name + f"_{str(date.today())}_{current_time}"

    args.optuna_n_trials = 20

    args.cv_n_iter = 500
    args.scoring = "f1"  # 'f1', precision
    args.cv_method = "random" # optuna random
    args.cv_gap = 1051 * 5
    args.n_splits = 3 #
    args.n_jobs = 1
    args.delta_train = 50
    args.delta_delay = 7
    args.delta_test = 20

    args.random_state = 41  # for data prep

    args.cols_to_drop = COLUMNS_TO_DROP

    args.session_gap_minutes=60*3 
    args.onehot_threshold=9

    args.do_pca = True  # try pca
    args.do_poly_expansion = False
    args.do_feature_selection = True

    args.disable_pyod_outliers = True
    args.pyod_detectors = [
        "abod",
        "cblof",
        "hbos",
        "iforest",
        "knn",
        "loda",
        "mcd",
        "mo_gaal",
    ]

    args.sampler_names = None
    args.sampler_cfgs = None
    args.disable_samplers = True

    args.concat_features = (
        None,  # ("AccountId", "CUSTOMER_ID") # ("AccountId", "CUSTOMER_ID") or None
    )
    args.concat_features_encoding_kwargs = dict(
        cat_encoding_method="hashing", n_components=14
    )

    args.add_imputer = False  # handle missing values at prediction time

    args.cat_encoding_method = "binary"  # to handle unknown values effectively, 'catboost', 'binary', 'hashing'
    args.cat_encoding_hash_n_components = 7  # if cat_encoding_method='hashing'
    args.cat_encoding_base_n = 4  # if cat_encoding_method=base_n
    args.windows_size_in_days = (1, 7, 30)
    cat_encoding_kwards = dict(
        n_components=args.cat_encoding_hash_n_components
    )  # used for 'catboost', 'hashing'

    # for debugging
    # demo(args=args)

    args.work_dir = Path(r"D:\fraud-detection-galsen\runs-optuna")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir = str(args.work_dir)  # for json serialization

    # using optuna
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(multivariate=True, group=True),
        study_name=args.study_name,
        load_if_exists=False,
        storage="sqlite:///hypsearch.sql",
    )

    # save args
    with open(os.path.join(args.work_dir, args.study_name + ".json"), "w") as file:
        # cols_preprocessed = dict()
        # cols_preprocessed["cols_to_drop"] = COLUMNS_TO_DROP
        # cols_preprocessed["cols_one_hot"] = COLUMNS_TO_ONE_HOT_ENCODE
        # cols_preprocessed["cols_cat_encorde"] = COLUMNS_TO_CAT_ENCODE
        # cols_preprocessed["cols_to_std_Scale"] = COLUMNS_TO_STD_SCALE
        # cols_preprocessed["cols_to_robust_scale"] = COLUMNS_TO_ROBUST_SCALE

        json.dump({"args": args.__dict__}, file, indent=4)

    objective_optuna = Tuner(
        args=args, verbose=0, cat_encoding_kwards=cat_encoding_kwards
    )

    objective_optuna.load_hyp_conf(r"D:\fraud-detection-galsen\tools\hyp_search_conf.py")

    study.optimize(
        objective_optuna,
        n_trials=args.optuna_n_trials,
        n_jobs=1,
        show_progress_bar=True,
        timeout=60 * 60 * 3,
    )
    # print(study.best_trial)

    # Save study.best_trial
    filename = os.path.join(args.work_dir, args.study_name + ".joblib")
    joblib.dump(study.best_trial, filename)
