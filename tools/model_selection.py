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
import mlflow
from datetime import datetime, date
from optuna.samplers import TPESampler
from fraudetect.config import (
    COLUMNS_TO_DROP,
    Arguments,
)
from optuna.integration.mlflow import MLflowCallback

from fraudetect.modeling.utils import Tuner


# %%main
if __name__ == "__main__":
    # Running
    args = Arguments()
    CURDIR  = Path(__file__).parent
    args.data_path = str(CURDIR / "../data/training.csv")

    args.model_names = (
        # "mlp",
        "decisionTree",
        # "clusterElastic",
        # "logisticReg",
        # "svc",
        # "randomForest",
        # "balancedRandomForest",
        # "gradientBoosting",
        # "histGradientBoosting",
        # "catboost",
        # 'lgbm',
        # "xgboost",
    )

    current_time = datetime.now().strftime("%H-%M")

    args.study_name = "cat-models"
    args.study_name = args.study_name + f"_{str(date.today())}_{current_time}"

    args.optuna_n_trials = 30

    args.cv_n_iter = 200
    args.scoring = "f1"  # 'f1', precision
    args.cv_method = "optuna"  # optuna random
    args.cv_gap = 1051 * 5
    args.n_splits = 5  #
    args.n_jobs = 4 # for hyp tuning
    args.delta_train = 50
    args.delta_delay = 7
    args.delta_test = 20

    args.random_state = 41  # for data prep

    args.cols_to_drop = COLUMNS_TO_DROP

    args.session_gap_minutes = 60 * 3

    args.do_pca = False  # try pca
    args.do_feature_selection = True
    args.add_fft=True
    args.add_seasonal_features=True
    args.use_nystrom=True
    args.use_sincos=False
    args.use_spline=True

    args.reorder_by = ('TX_DATETIME','AccountId')

    args.disable_pyod_outliers = True
    args.pyod_detectors = [
        "abod",
        "cblof",
        "hbos",
        "iforest",
        "knn",
        "loda",
        "mcd",
        # "mo_gaal",
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

    args.cat_encoding_method = "binary"  # to handle unknown values effectively, 'catboost', 'binary', 'hashing', 'None'
    args.cat_encoding_hash_n_components = 7  # if cat_encoding_method='hashing'
    args.cat_encoding_base_n = 4  # if cat_encoding_method=base_n
    args.onehot_threshold = 9
    args.windows_size_in_days = (1, 7, 30)
    cat_encoding_kwards = dict(
        hash_n_components=args.cat_encoding_hash_n_components,
        handle_missing="value",
        return_df=True,
        hash_method="md5",
        drop_invariant=False,
        handle_unknown="value",
        base=args.cat_encoding_base_n,
    ) 

    # for debugging
    # demo(args=args)

    args.work_dir = CURDIR / "runs-optuna"
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir = str(args.work_dir)  # for json serialization

    # using optuna
    
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(multivariate=True, group=True),
        study_name=args.study_name,
        load_if_exists=True,
        storage="sqlite:///hypsearch.sql",
    )

    # save args
    with open(os.path.join(args.work_dir, args.study_name + ".json"), "w") as file:
        json.dump({"args": args.__dict__}, file, indent=4)

    objective_optuna = Tuner(
        args=args, 
        verbose=0, 
        cat_encoding_kwards=cat_encoding_kwards,
        tune_threshold=True
    )
    objective_optuna.load_hyp_conf(CURDIR / "hyp_search_conf.py")

    mlflow.set_tracking_uri(uri="http://localhost:5000")
    exp_name = "".join(args.study_name.split('_')[:-2])
    
    try:
        exp_id = mlflow.get_experiment_by_name(exp_name).experiment_id
    except:
        exp_id = mlflow.create_experiment(name=exp_name)

    mlflow.set_experiment(experiment_id=exp_id)
    mlflc = MLflowCallback(metric_name=args.scoring,
                           create_experiment=False,
                           )
    study.optimize(
        objective_optuna,
        n_trials=args.optuna_n_trials,
        n_jobs=1,
        show_progress_bar=True,
        timeout=60 * 60 * 3,
        callbacks=[mlflc],
    )
    # print(study.best_trial)

    # Save study.best_trial
    filename = os.path.join(args.work_dir, args.study_name + ".joblib")
    joblib.dump(study.best_trial, filename)
