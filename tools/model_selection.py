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
    # COLUMNS_TO_DROP,
    Arguments,
)
from optuna.integration.mlflow import MLflowCallback

from fraudetect.modeling.utils import Tuner
from typing import Sequence

# %%main
if __name__ == "__main__":
    # Running
    args = Arguments()
    CURDIR  = Path(__file__).parent
    args.data_path = str(CURDIR / "../data/training.csv")

    args.model_names = (
        # "mlp",
        # "decisionTree",
        # "clusterElastic",
        # "logisticReg",
        # "svc",
        # "randomForest",
        # "balancedRandomForest",
        "gradientBoosting",
        # "histGradientBoosting",
        # "catboost",
        # 'lgbm',
        "xgboost",
    )

    current_time = datetime.now().strftime("%H-%M")

    mlflow_exp_name = "ensemble-trees" 
    args.study_name = mlflow_exp_name
    args.study_name = args.study_name + f"_{str(date.today())}_{current_time}"

    args.optuna_n_trials = 300

    # args.cv_n_iter = 200 # not used
    args.scoring = ["f1", "average_precision"]  # ["f1", "average_precision", "precision", "recall"]
    args.cv_method = "optuna"  # optuna random
    args.cv_gap = 1051 * 5
    args.n_splits = 5  #
    args.n_jobs = 10 # for hyp tuning
    args.delta_train = 50
    args.delta_delay = 7
    args.delta_test = 20 

    args.random_state = 41  # for data prep

    cols_to_drop = ["CurrencyCode",
                    "CountryCode",
                    "SubscriptionId",
                    "BatchId",
                    "CUSTOMER_ID",
                    "AccountId",
                    "TRANSACTION_ID",
                    "TX_DATETIME",
                    "TX_TIME_DAYS",
                    "Hour_Account",
                    "DayOfWeek_Account",
                    "ProductCategory_Account",
                    "CustomerCluster",
                    "CustomerUID",
    ]

    args.interaction_cat_cols= [
                        'ChannelId',
                        'PricingStrategy',
                        'ProductId',
                        'ProductCategory',
                        'ProviderId'
                        ]
    args.add_poly_interactions = True
    args.poly_degree=1
    args.iterate_poly_cat_encoder_name=False
    args.poly_cat_encoder_name="binary" # or woe or catboost or binary -> used if iterate_poly_cat_encoder_name=False

    args.nlp_model_name = 'en_core_web_sm' # en-core-web-sm en_core_web_md
    args.cat_similarity_encode = None #['ProductCategory',] # None ProductCategory

    args.n_clusters = 0

    # cols to concat and form a UID
    args.uid_col_name="CustomerUID"
    args.uid_cols = (
        "AccountId", "CUSTOMER_ID" # or None
    )
    # dropping uid of client if exists
    if all(args.uid_cols):
        cols_to_drop.append(args.uid_col_name)
    
    args.cols_to_drop = cols_to_drop

    args.add_fraud_rate_features = False

    iterate_session_gap = True
    args.session_gap_minutes = 60 * 12

    iterate_cat_method = True # if True, then args.cat_encoding_method is not used
    args.cat_encoding_methods = (
                                'binary',
                                # 'catboost',
                                # 'count',
                                # 'woe', 
                                # 'similarity' only suitable for ProductCategory
                                )

    args.do_pca = False  # try pca
    args.do_feature_selection = True
    args.add_fft=False
    args.add_seasonal_features=False
    args.use_nystrom=False
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

    args.add_imputer = False  # handle missing values at prediction time
    args.imputer_n_neighbors=9
    
    args.cat_encoding_method = "binary"  # to handle unknown values effectively, 'catboost', 'binary', 'hashing', 'None'
    args.cat_encoding_hash_n_components = 12  # if cat_encoding_method='hashing'
    args.cat_encoding_base_n = 4  # if cat_encoding_method=base_n
    args.onehot_threshold = 6
    args.windows_size_in_days = (1, 3, 7, 30)
    cat_encoding_kwards = dict(
        hash_n_components=args.cat_encoding_hash_n_components,
        handle_missing="value",
        return_df=True,
        hash_method="md5",
        drop_invariant=False,
        handle_unknown="value",
        base=args.cat_encoding_base_n,
        woe_randomized=True,
        woe_sigma=0.05,
        woe_regularization=1.0,
    ) 

    # for debugging
    # demo(args=args)

    args.work_dir = CURDIR / "../runs-optuna"
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir = str(args.work_dir)  # for json serialization

    # using optuna
    opt_direction = dict()
    # if isinstance(args.scoring,Sequence):
    #     opt_direction['directions'] = ["maximize"]*len(args.scoring)
    # else:
    opt_direction['direction'] = "maximize"
    study = optuna.create_study(
        sampler=TPESampler(multivariate=True, group=True),
        study_name=args.study_name,
        load_if_exists=True,
        storage="sqlite:///hypsearch.sql",
        **opt_direction
    )

    # save args
    with open(os.path.join(args.work_dir, args.study_name + ".json"), "w") as file:
        json.dump({"args": args.__dict__}, file, indent=4)

    objective_optuna = Tuner(
        args=args, 
        verbose=0, 
        cat_encoding_kwards=cat_encoding_kwards,
        tune_threshold=False,
        iterate_cat_method=iterate_cat_method,
        iterate_session_gap=iterate_session_gap,
    )
    objective_optuna.load_hyp_conf(CURDIR / "hyp_search_conf.py")

    mlflow.set_tracking_uri(uri="http://localhost:5000")    
    try:
        exp_id = mlflow.get_experiment_by_name(mlflow_exp_name).experiment_id
    except:
        exp_id = mlflow.create_experiment(name=mlflow_exp_name)

    mlflow.set_experiment(experiment_id=exp_id)
    mlflow_metric_name = args.scoring if isinstance(args.scoring, str) else "fitness"
    mlflc = MLflowCallback(metric_name=mlflow_metric_name,
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
    if isinstance(args.scoring,str):
        joblib.dump(study.best_trial, filename)
    else:
        joblib.dump(study.best_trials, filename)
