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

from fraudetect.preprocessing import FraudFeatureEngineer, FeatureEncoding
from fraudetect.dataset import load_data, MyDatamodule
from fraudetect.config import Arguments
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

COLUMNS_TO_DROP = [
    "CurrencyCode",
    "CountryCode",
    "SubscriptionId",
    "BatchId",
    "CUSTOMER_ID",
    "AccountId",
    "TRANSACTION_ID",
    "TX_DATETIME",
    "TX_TIME_DAYS",
]


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


#%% Preprocessing pipeline

from fraudetect.dataset import load_data
from fraudetect.preprocessing import FraudFeatureEngineer, FeatureEncoding
from fraudetect.preprocessing.preprocessing import (load_cols_transformer,
                                                    fit_outliers_detectors,
                                                    ColumnDropper, 
                                                    Pipeline, 
                                                    FeatureUnion,
                                                    OutlierDetector,
                                                    AdvancedFeatures)
from fraudetect import import_from_path, sample_cfg

from fraudetect.modeling.utils import get_model, sample_model_cfg, instantiate_model
from fraudetect.detectors import get_detector, instantiate_detector
import sklearn
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from shutil import rmtree
from joblib import Memory

CONFIGS = import_from_path(
    "hyp_search_conf", r"D:\fraud-detection-galsen\tools\hyp_search_conf.py"
)

raw_data_train = load_data(r"D:\fraud-detection-galsen\data\training.csv")

raw_data_pred = load_data(r"D:\fraud-detection-galsen\data\test.csv")


# sklearn.set_config(enable_metadata_routing=True)

COLUMNS_TO_DROP = [
    "CurrencyCode",
    "CountryCode",
    "SubscriptionId",
    "BatchId",
    "CUSTOMER_ID",
    "AccountId",
    "TRANSACTION_ID",
    "TX_DATETIME",
]

dropper = ColumnDropper(cols_to_drop=COLUMNS_TO_DROP)


y_train = raw_data_train['TX_FRAUD']


# v2
encoder_2 = FeatureEncoding(add_imputer=False,
                            cat_encoding_method='binary',
                            imputer_n_neighbors=9,
                            n_jobs=4
                            )
# X_ = encoder_2.fit_transform(X=raw_data_train)


# v3
# pipe = Pipeline(steps=[('col_dropper',dropper), ('col_encoder',encoder_2)])
# X_ = pipe.fit_transform(X=raw_data_train)

# v4
feature_engineer = FraudFeatureEngineer(windows_size_in_days=[1,7,30],
                                         uid_cols=[None,],
                                         session_gap_minutes=60*3,
                                         n_clusters=8
                                        )
# X_ = feature_engineer.fit_transform(X=raw_data_train)


# v5
pipe2 = Pipeline(steps=[('feature_engineer',feature_engineer),
                       ('col_dropper',dropper),
                       ('col_encoder',encoder_2)
                ]
            )


X_all = pd.concat([raw_data_train,raw_data_pred],axis=0).reset_index(level=0,drop=True)

X_all = pipe2.fit_transform(X=X_all,y=None)
X_train = X_all[:len(raw_data_train),:]

# X_pred = pipe2.transform(X=raw_data_pred)



model_list = list()
# names = CONFIGS.outliers_detectors.keys()
names = ["cblof", "iforest"]
names = sorted(names)
outliers_det_configs = dict()

for name in names:
    detector, cfg = get_detector(name=name, config=CONFIGS.outliers_detectors)
    cfg = sample_cfg(cfg)
    detector = instantiate_detector(detector, cfg)
    model_list.append(detector)

    cfg["detector"] = CONFIGS.outliers_detectors[name]["detector"]
    outliers_det_configs[name] = cfg


# fit_outliers_detectors(detector_list=zip(names,model_list), X_train=X_train)

pyod_det = OutlierDetector(detector_list=model_list)
# det_scores = pyod_det.fit_transform(X=X_train, y=None)


# Feature selection
estimator = DecisionTreeClassifier(max_depth=15,
                                   max_features='sqrt',
                                   random_state=41)

feature_selector = AdvancedFeatures(verbose=True,
                                    estimator=estimator,
                                    do_pca=True,
                                    top_k_best=10,
                                    pca_n_components=20,
                                    feature_selector_name="selectkbest"
                                    )
# X_selected = feature_selector.fit_transform(X=X_train,y=y_train)

concatenator = FeatureUnion(transformer_list=[('pyod_det',pyod_det), ('feature_selector',feature_selector)],
                            n_jobs=2
                            )
X_pyod = concatenator.fit_transform(X=X_train,y=y_train)

# load model
model, model_cfg = get_model('logisticReg', CONFIGS.models)
model_cfg = sample_model_cfg(model_cfg)
model = instantiate_model(model, **model_cfg)

# Create a temporary folder to store the transformers of the pipeline
location = "cachedir"
memory = Memory(location=location, verbose=1)

# Final Pipeline
workflow = Pipeline(steps=[('feature_engineer',feature_engineer),
                       ('col_dropper',dropper),
                       ('col_encoder',encoder_2),
                       # ('feature_selector',feature_selector),
                       ('concat',concatenator),
                       # ('model', model),
                       ],
                    # memory=memory
            )

X_processed = workflow.fit_transform(X=raw_data_train, y=y_train)

# score = workflow.fit(X=raw_data_train, y=y_train).score(X=raw_data_train,y=y_train)

# Delete the temporary cache before exiting
# memory.clear(warn=False)
# rmtree(location)
 

