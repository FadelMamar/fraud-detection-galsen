# %% init
# from fraudetect.dataset import load_data
# from fraudetect.features import perform_feature_engineering
# from fraudetect.config import (
#     COLUMNS_TO_DROP,
#     COLUMNS_TO_ONE_HOT_ENCODE,
#     COLUMNS_TO_SCALE,
# )
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# import numpy as np
# from fraudetect import import_from_path, sample_cfg
# from collections import OrderedDict


# df_data = load_data(r"D:\fraud-detection-galsen\data\training.csv")

# onehot_encoder = OneHotEncoder(
#     sparse_output=False, handle_unknown="ignore", drop=None, dtype=np.float64
# )
# scaler = StandardScaler()

# X_train, y_train = perform_feature_engineering(
#     transactions_df=df_data,
#     columns_to_drop=COLUMNS_TO_DROP,
#     columns_to_onehot_encode=COLUMNS_TO_ONE_HOT_ENCODE,
#     columns_to_scale=COLUMNS_TO_SCALE,
#     onehot_encoder=onehot_encoder,
#     scaler=scaler,
#     mode="train",
#     windows_size_in_days=[1, 7, 30],
# )

# %% load configs
# configs = import_from_path(
#     "hyp_search_conf", r"D:\fraud-detection-galsen\tools\hyp_search_conf.py"
# )

# %% Outliers detectors
# from fraudetect.features import (
#     fit_outliers_detectors,
#     concat_outliers_probs_pyod,
#     load_transforms_pyod,
# )
# from fraudetect.detectors import get_detector, instantiate_detector

# model_list = list()
# # names = configs.outliers_detectors.keys()
# names = ["cblof", "iforest"]

# outliers_det_configs = []
# for name in sorted(names):
#     detector, cfg = get_detector(name=name, config=configs.outliers_detectors)
#     cfg = sample_cfg(cfg)
#     detector = instantiate_detector(detector, cfg)
#     model_list.append(detector)

#     cfg["detector"] = configs.outliers_detectors[name]["detector"]
#     outliers_det_configs.append((name, cfg))

# model_list = fit_outliers_detectors(model_list, X_train)
# X_t = concat_outliers_probs_pyod(
#     fitted_detector_list=model_list,
#     X=X_train,

# )

# outliers_det_configs = OrderedDict(outliers_det_configs)
# transform = load_transforms_pyod(
#     X_train=X_train,
#     outliers_det_configs=outliers_det_configs,
#     method="linear",
#     add_confidence=False,
# )

# %%  Undersampling


# # under_sampler = TomekLinks(sampling_strategy='majority',n_jobs=4)
# # X_us,  y_us = under_sampler.fit_resample(X_train, y_train)

# # under_sampler_rus = RandomUnderSampler(sampling_strategy='majority',random_state=41,replacement=False)
# # X_rus,  y_rus = under_sampler_rus.fit_resample(X_train, y_train)

# # under_sampler_knn = AllKNN(sampling_strategy='majority',n_neighbors=5,kind_sel='all',allow_minority=False,n_jobs=4)
# # X_knn,  y_knn = under_sampler_knn.fit_resample(X_train, y_train)

# # >> Selected
# # frac = float(2*y_train.sum()/(1-y_train).sum())
# # under_sampler_nm = NearMiss(sampling_strategy=frac,n_neighbors=5,version=1,n_jobs=4)
# # X_nm,  y_nm = under_sampler_nm.fit_resample(X_train, y_train)

# # under_sampler_enn = EditedNearestNeighbours(sampling_strategy='majority',n_neighbors=5,kind_sel='mode',n_jobs=4)
# # X_enn,  y_enn = under_sampler_enn.fit_resample(X_train, y_train)

# # under_sampler_cnn = CondensedNearestNeighbour(sampling_strategy='majority',n_neighbors=5,random_state=41,n_seeds_S=5,n_jobs=4)
# # X_cnn,  y_cnn = under_sampler_cnn.fit_resample(X_train, y_train)

# # under_sampler_oss = OneSidedSelection(sampling_strategy='majority',n_neighbors=5,random_state=41,n_seeds_S=2,n_jobs=4)
# # X_oss,  y_oss = under_sampler_oss.fit_resample(X_train, y_train)

# # Too slow
# # under_sampler_cc = ClusterCentroids(sampling_strategy=frac,
# #                                     estimator=MiniBatchKMeans(max_iter=100,batch_size=2048,tol=1e-6)
# #                                     ,voting='auto',random_state=41)
# # X_cc,  y_cc = under_sampler_cc.fit_resample(X_train, y_train)

# # under_sampler_ncr = NeighbourhoodCleaningRule(sampling_strategy='majority',n_neighbors=5,threshold_cleaning=0.2)
# # X_ncr,  y_ncr = under_sampler_ncr.fit_resample(X_train, y_train)


# %% Oversampling


# # frac = float(2*y_train.sum()/(1-y_train).sum())

# # oversampler_smote = SMOTE(sampling_strategy=frac,random_state=41, k_neighbors=5,)
# # X_ste, y_ste = oversampler_smote.fit_resample(X_train, y_train)
# # print(y_ste.sum()/y_train.sum())

# # oversampler_ada = ADASYN(sampling_strategy=frac,random_state=41, n_neighbors=5,)
# # X_ada, y_ada = oversampler_ada.fit_resample(X_train, y_train)
# # print(y_ada.sum()/y_train.sum())

# # oversampler_bste = BorderlineSMOTE(sampling_strategy=frac,random_state=41,m_neighbors=5,k_neighbors=10,kind='borderline-1')
# # X_bste, y_bste = oversampler_bste.fit_resample(X_train, y_train)
# # print(y_bste.sum()/y_train.sum())

# # oversampler_sste = SVMSMOTE(sampling_strategy=frac,random_state=41,out_step=0.5,m_neighbors=5,k_neighbors=3)
# # X_sste, y_sste = oversampler_sste.fit_resample(X_train, y_train)
# # print(y_sste.sum()/y_train.sum())


# # %% Test of samplers
# from fraudetect.sampling import sample_cfg, data_resampling


# sampler_names = [
#     "nearmiss",
# ]  #'nearmiss','SMOTE']


# def get_samplers_cfgs(sampler_names, configs):
#     sampler_cfgs = list()

#     for name in sampler_names:
#         cfg = configs.sampler[name]
#         cfg = sample_cfg(cfg)  # for test purposes
#         sampler_cfgs.append({name: cfg})
#     return sampler_cfgs


# sampler_cfgs = get_samplers_cfgs(sampler_names, configs)

# X_t, y_t = data_resampling(
#     X=X_train, y=y_train, sampler_names=sampler_names, sampler_cfgs=sampler_cfgs
# )

# %% Dataset

# from fraudetect.preprocessing import FraudFeatureEngineer, FeatureEncoding
# from fraudetect.dataset import load_data, MyDatamodule
# from fraudetect.config import Arguments
# from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

# COLUMNS_TO_DROP = [
#     "CurrencyCode",
#     "CountryCode",
#     "SubscriptionId",
#     "BatchId",
#     "CUSTOMER_ID",
#     "AccountId",
#     "TRANSACTION_ID",
#     "TX_DATETIME",
#     "TX_TIME_DAYS",
# ]


# args = Arguments()

# args.windows_size_in_days = (1, 7, 30)
# args.cat_encoding_method = "hashing"
# args.add_imputer = True
# args.data_path = r"D:\fraud-detection-galsen\data\training.csv"
# encoding_kwargs = dict(n_components=14)


# datamodule = MyDatamodule()

# feature_engineer = FraudFeatureEngineer(
#     windows_size_in_days=args.windows_size_in_days,
#     uid_cols=None,
#     session_gap_minutes=30,
#     n_clusters=8,
# )

# encoder = FeatureEncoding(
#     cat_encoding_method=args.cat_encoding_method,
#     add_imputer=args.add_imputer,
#     onehot_threshold=9,
#     cols_to_drop=COLUMNS_TO_DROP,
#     n_jobs=1,
#     cat_encoding_kwards=encoding_kwargs,
# )

# datamodule.setup(encoder=encoder, feature_engineer=feature_engineer)


# X_train, y = datamodule.get_train_dataset(args.data_path)

# X_pred, _ = datamodule.get_predict_dataset(r"D:\fraud-detection-galsen\data\test.csv")


# train_raw_data = load_data("../data/training.csv")
# pred_raw_data = load_data(r"D:\fraud-detection-galsen\data\test.csv")

# # feature enineering
# df_train = feature_engineer.fit_transform(train_raw_data)
# df_pred = feature_engineer.transform(pred_raw_data)


# # encoding
# encoder = FeatureEncoding(cat_encoding_method='hashing',
#                           add_imputer=False,
#                             onehot_threshold=9,
#                             cols_to_drop=COLUMNS_TO_DROP,
#                             n_jobs=1,
#                             cat_encoding_kwards={'n_components':7}
#                             )
# # encoder.fit(df_train)
# X_train,y_train = encoder.fit_transform(X=df_train)
# X_pred,y_pred= encoder.transform(X=df_pred)


# %% Preprocessing pipeline
from pathlib import Path
from fraudetect.dataset import load_data
from fraudetect.preprocessing import FraudFeatureEngineer, FeatureEncoding,ToDataframe,DimensionReduction
from fraudetect.preprocessing.preprocessing import (
    load_cols_transformer,
    fit_outliers_detectors,
    load_feature_selector,
    ColumnDropper,
    load_workflow,
    load_cat_encoding,
    Pipeline,
    FeatureUnion,
    OutlierDetector,
)
from fraudetect import import_from_path, sample_cfg

from fraudetect.modeling.utils import get_model, sample_model_cfg, instantiate_model
from fraudetect.detectors import get_detector, instantiate_detector
import sklearn
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from shutil import rmtree
from joblib import Memory
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit
from typing import Sequence


CURDIR = Path(__file__).parent
cfg_path = CURDIR / "hyp_search_conf.py"
CONFIGS = import_from_path("hyp_search_conf", cfg_path)

raw_data_train = load_data(CURDIR / "../data/training.csv")

raw_data_pred = load_data(CURDIR / "../data/test.csv")

# sklearn.set_config(enable_metadata_routing=True)

cols_to_drop = ["CurrencyCode",
                    "CountryCode",
                    "SubscriptionId",
                    "BatchId",
                    # "CUSTOMER_ID",
                    # "AccountId",
                    "TRANSACTION_ID",
                    "TX_DATETIME",
                    "TX_TIME_DAYS",
                    "CustomerUID",
    ]

interaction_cat_cols= [
                        'ChannelId',
                        'PricingStrategy',
                        'ProductId',
                        'ProductCategory',
                        'ProviderId',
                        "CustomerCluster"
                        ]


y_train = raw_data_train["TX_FRAUD"]


# model_list = list()
# names = ["cblof", "iforest"]
# names = sorted(names)
# outliers_det_configs = dict()

# for name in names:
#     detector, cfg = get_detector(name=name, config=CONFIGS.outliers_detectors)
#     cfg = sample_cfg(cfg)
#     detector = instantiate_detector(detector, cfg)
#     model_list.append(detector)

#     cfg["detector"] = CONFIGS.outliers_detectors[name]["detector"]
#     outliers_det_configs[name] = cfg


# load model
# "mlp",
# "decisionTree",
# "logisticReg",
# "svc",
# "randomForest",
# "balancedRandomForest",
# "gradientBoosting",
# "histGradientBoosting",
# "catboost",
# 'lgbm',
model_name = 'decisionTree' #
model, model_cfgs = get_model(model_name, CONFIGS.models)
model_cfg = sample_model_cfg(model_cfgs)
model = instantiate_model(model, **model_cfg)



workflow = load_workflow(
    classifier=None,
    cols_to_drop=cols_to_drop,
    pca_n_components=20,
    detector_list=None,
    n_splits=5,
    cv_gap=5000,
    scoring="f1",
    onehot_threshold=6,
    session_gap_minutes=60 * 36,
    uid_cols=['AccountId','CUSTOMER_ID'],
    uid_col_name="CustomerUID",
    add_fraud_rate_features = False,
    reorder_by=['TX_DATETIME','AccountId'],
    behavioral_drift_cols=[
                            "CustomerUID",
                           'AccountId',
                           ],
    feature_selector_name = "smartcorrelated",
    top_k_best=100,
    seq_n_features_to_select=3,
    corr_method="spearman", 
    corr_threshold = 0.8,
    windows_size_in_days=[1, 7, 30],
    cat_encoding_method = "hashing",
    cat_similarity_encode=None,
    nlp_model_name='en_core_web_md',
    cat_encoding_kwargs={'hash_n_components':7},
    add_poly_interactions=True,
    interaction_cat_cols=interaction_cat_cols,
    add_cum_features=True,
    poly_degree=1,
    poly_cat_encoder_name="hashing",
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
    n_clusters=5,
    do_pca=False,
    verbose=False,
    n_jobs=2,
)


y_train = raw_data_train['TX_FRAUD']
X_train = raw_data_train.drop(columns=['TX_FRAUD'])

X_preprocessed = workflow.fit_transform(X=X_train, y=y_train)

# selector = load_feature_selector(name='sequential',
#                                  rfe_step=1,
#                                  n_features_to_select=60,
#                                  scoring='f1')
# selector.set_params(n_jobs=8)

# X_selected = selector.fit_transform(X_preprocessed,y=y_train)

# score = cross_val_score(model,
#                         X=X_selected,y=y_train,
#                         cv=TimeSeriesSplit(n_splits=5,gap=5000),
#                         scoring='f1',
#                         n_jobs=8)
# print("score: ", score)
# workflow.predict(X_train)


# X_pred = data_processor.transform(raw_data_pred)

# print('X_train.shape: ',X_train.shape)
# print('X_pred.shape: ',X_pred.shape)

# print('Num NaN train', (X_train_processed==np.nan).sum())

# model.fit(X_train_processed,y_train)
# score = model.score(X_train_processed,y_train)



# params_config = {f"model__{k}":v for k,v in model_cfgs.items() if isinstance(v, Sequence)}


# search_engine = RandomizedSearchCV(
#     workflow,
#     param_distributions=params_config,
#     scoring='f1',
#     cv=TimeSeriesSplit(n_splits=4,gap=5000),
#     refit=True,
#     n_jobs=12,
#     n_iter=10,
#     error_score='raise',
#     # random_state=41,
#     verbose=True,
# )

# import optuna
# from optuna.samplers import TPESampler

# study = optuna.create_study(
#     direction="maximize",
#     sampler=TPESampler(multivariate=True, group=True),
#     load_if_exists=True,
# )

# cfg = dict()
# for k, v in params_config.items():
#     if isinstance(v, Sequence):
#         cfg[k] = optuna.distributions.CategoricalDistribution(v)

# search_engine = optuna.integration.OptunaSearchCV(
#     workflow,
#     param_distributions=cfg,
#     cv=TimeSeriesSplit(n_splits=5, gap=5000),
#     refit=True,
#     n_jobs=1,
#     study=study,
#     scoring="f1",
#     error_score="raise",
#     max_iter=300,
#     timeout=60 * 3,
#     n_trials=5,
#     # random_state=41,
#     verbose=True,
# )

# search_engine.fit(X=X_train, y=y_train)

# print(search_engine.best_estimator_)
