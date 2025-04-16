# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 16:51:56 2025

@author: FADELCO
"""

# %% funcs & imports
import json
from pathlib import Path
import os
import joblib
import json
import optuna
from datetime import datetime, date
from optuna.samplers import TPESampler
from fraudetect.config import (
    COLUMNS_TO_DROP,
    Arguments,
)

from fraudetect.modeling.utils import Tuner


# %%main
if __name__ == "__main__":
    # Running
    args = Arguments()
    args.data_path = "../data/training.csv"

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
    args.do_feature_selection = False

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

    args.work_dir = Path("./runs-optuna")
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

    objective_optuna.load_hyp_conf("./hyp_search_conf.py")

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
