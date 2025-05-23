import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import (
    RFECV,
    SequentialFeatureSelector,
    SelectKBest, 
    f_classif, 
    mutual_info_classif,
    r_regression
)

# from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    BaggingClassifier,
    VotingClassifier,
    AdaBoostClassifier,
    StackingClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from feature_engine.selection import SmartCorrelatedSelection

from imblearn.under_sampling import NearMiss, EditedNearestNeighbours
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import (BalancedRandomForestClassifier,
                               EasyEnsembleClassifier,
                               BalancedBaggingClassifier,
                               RUSBoostClassifier)

from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.loda import LODA
from pyod.models.knn import KNN
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.vae import VAE
from pyod.models.ae1svm import AE1SVM
from pyod.models.mo_gaal import MO_GAAL
from collections import OrderedDict

from fraudetect.modeling.models import ClusterElasticClassifier


# %% data aug - pyod
outliers_detectors = dict()

_cnt = [0.0001, 0.00046, 0.00077, 0.0021, 0.006, 0.008]

outliers_detectors["iforest"] = dict(
    detector=IForest,
    n_estimators=[20, 50, 100, 200],
    contamination=_cnt,
    random_state=[41],
)

outliers_detectors["DIF"] = dict(
    contamination=_cnt,
    detector=DIF,
    hidden_activation=["tanh", "relu"],
    n_ensemble=[20, 50, 75, 100],
    n_estimators=[3, 6, 9, 12],
    hidden_neurons=[[64, 32], [128, 64, 32], [128, 64, 32, 16]],
    representation_dim=[10, 20, 50, 85],
    max_samples=[256, 512, 1024, 2048],
    skip_connection=[False, True],
    random_state=[41],
)

outliers_detectors["abod"] = dict(
    detector=ABOD,
    n_neighbors=[5, 7, 9],
    contamination=_cnt,
    method=[
        "fast",
    ],
)

outliers_detectors["cblof"] = dict(
    detector=CBLOF,
    n_clusters=[5, 8, 13, 18],
    contamination=_cnt,
    alpha=[0.55, 0.75, 0.85, 0.95],
    beta=[5, 10, 15],
    random_state=[
        41,
    ],
    use_weights=[False, True],
)

outliers_detectors["hbos"] = dict(
    detector=HBOS,
    n_bins=["auto", 10, 15, 20],
    contamination=_cnt,
    alpha=[0.55, 0.75, 0.85, 0.95],
    tol=[0.15, 0.25, 0.5, 0.75, 0.85],
)

outliers_detectors["loda"] = dict(
    detector=LODA,
    n_bins=["auto", 10, 15, 20],
    contamination=_cnt,
    n_random_cuts=[50, 75, 100, 150],
)

outliers_detectors["knn"] = dict(
    detector=KNN,
    n_neighbors=[3, 5, 7, 9],
    contamination=_cnt,
    method=["largest"],
    radius=[1.0],
    algorithm=[
        "auto",
    ],
)

outliers_detectors["mcd"] = dict(
    detector=MCD,
    assume_centered=[False, True],
    contamination=_cnt,
    random_state=[41],
)

outliers_detectors["ocsvm"] = dict(
    detector=OCSVM,
    kernel=["rbf", "poly"],
    tol=[
        1e-4,
    ],
    max_iter=[1000],
    gamma=[
        1e-2,
    ],
    nu=[0.1, 0.25, 0.5],
    contamination=_cnt,
)

outliers_detectors["ae1svm"] = dict(
    detector=AE1SVM,
    learning_rate=[1e-3, 1e-4],
    hidden_neurons=[[64, 32], [128, 64, 32], [128, 64, 32, 16]],
    batch_size=[
        64,
    ],
    weight_decay=[1e-1, 1e-2, 1e-3],
    contamination=_cnt,
)

outliers_detectors["vae"] = dict(
    detector=VAE,
    lr=[1e-3, 1e-4],
    output_activation_name=["relu", "sigmoid"],
    batch_size=[
        64,
    ],
    epoch_num=[
        50,
    ],
    batch_norm=[
        True,
    ],
    latent_dim=[2, 8, 16, 32],
    optimizer_params=[
        {"weight_decay": 1e-1},
        {"weight_decay": 1e-2},
        {"weight_decay": 1e-3},
    ],
    random_state=[
        41,
    ],
    use_compile=[
        False,
    ],
    contamination=_cnt,
)

outliers_detectors["mo_gaal"] = dict(
    detector=MO_GAAL,
    k=[5, 10, 20],
    momentum=[
        0.9,
    ],
    stop_epochs=[
        20,
    ],
    lr_d=[1e-2],
    lr_g=[1e-4],
    contamination=_cnt,
)

outliers_detectors = OrderedDict(
    [
        (k, outliers_detectors[k])
        for k in sorted(outliers_detectors.keys(), reverse=False)
    ]
)

# %% samplers config
samplers = dict()

# under sampling
fracs = (np.linspace(0.05,0.15,10)).tolist()
n_neighbors=[3, 5, 7, 9]
samplers["nearmiss"] = dict(
    sampling_strategy=fracs,
    n_neighbors=n_neighbors,
    version=[
        1,
    ],
    sampler=NearMiss,
)
# oversampling
fracs = (np.arange(2, 5) * 4e-3).tolist()
samplers["SMOTE"] = dict(
    sampling_strategy=fracs, 
    # random_state=[41], 
    k_neighbors=n_neighbors, sampler=SMOTE
)

samplers["adasyn"] = dict(
    sampling_strategy=fracs, 
    # random_state=[41], 
    n_neighbors=n_neighbors, 
    sampler=ADASYN
)

samplers["borderlineSMOTE"] = dict(
    sampling_strategy=fracs,
    # random_state=[41],
    m_neighbors=[3, 5, 7, 9],
    k_neighbors=n_neighbors,
    kind=["borderline-1", "borderline-2"],
    sampler=BorderlineSMOTE,
)

samplers["kmeansSMOTE"] = dict(
    sampling_strategy=fracs,
    # random_state=[41],
    k_neighbors=n_neighbors,
    sampler=KMeansSMOTE,
)

samplers["svmSMOTE"] = dict(
    sampling_strategy=fracs,
    # random_state=[41],
    m_neighbors=[5, 7, 10, 15, 20],
    out_step=[0.1, 0.2, 0.5, 0.7],
    k_neighbors=n_neighbors,
    sampler=SVMSMOTE,
)

# combined sampler
samplers["smoteENN"] = dict(
    sampling_strategy=fracs,
    random_state=[41],
    enn=EditedNearestNeighbours(
        sampling_strategy="majority", n_neighbors=5, kind_sel="all",
    ),
    sampler=SMOTEENN,
)

samplers["smoteTOMEK"] = dict(
    sampling_strategy=fracs, 
    # random_state=[41], 
    sampler=SMOTETomek
)

undersamplers = [
    "nearmiss",
]
oversamplers = [
    "SMOTE",
    "adasyn",
    "borderlineSMOTE",
    "svmSMOTE",
    "kmeansSMOTE"
]
combinedsamplers = [
    "smoteENN",
    "smoteTOMEK",
]


# %% models
models = dict()

learning_rate = [1e-1,1e-2,1e-3] #np.linspace(1e-3,3e-1,10).round(4).tolist()
C = np.logspace(1,4,50).tolist()
n_estimators = np.arange(15, 200,step=2).tolist()
max_depth = np.arange(2, 4).tolist()
criterion=["gini",]
sampling_strategy=np.linspace(5e-2,2e-1,20).tolist()

# own models
models["clusterElastic"] = dict(
    en_l1_ratio=np.linspace(0.1, 0.9, 10).round(3).tolist(),
    random_state=[41],
    n_clusters=np.arange(2,8).tolist(),
    base_estimator=DecisionTreeClassifier(max_depth=7,
                                        class_weight='balanced',
                                        max_features=None),
    model=ClusterElasticClassifier,
)

# sklearn-like models
models["logisticReg"] = dict(
    penalty=["l2"],
    C=C,
    class_weight=["balanced", None],
    solver=["liblinear"],
    max_iter=[int(1e4)],
    random_state=[None],
    tol=[1e-4],
    model=LogisticRegression,
)

models["svc"] = dict(
    C=C,
    kernel=["poly", "rbf", "linear"],
    degree=[2, 3, 5, 7],
    gamma=["auto", "scale"],
    tol=[1e-4],
    class_weight=["balanced", None],
    max_iter=[int(1e4)],
    random_state=[None],
    probability=[False],
    model=SVC,
)

models["linearSVC"] = dict(
    penalty=["l2"],
    loss=["square_hinge", "hinge"],
    C=C,
    class_weight=["balanced", None],
    max_iter=[int(1e4)],
    random_state=[None],
    model=LinearSVC,
    tol=[1e-4],
)

models["sgdClassifier"] = dict(
    loss=[
        "hinge",
        "squared_hinge",
        "modified_huber",
        "log_loss",
    ],
    penalty=["l2", "l1", "elasticnet", None],
    alpha=C,
    l1_ratio=[0.15, 0.5, 0.85],
    class_weight=[
        "balanced",
    ],
    max_iter=[int(1e5)],
    learning_rate=[
        "optimal",
    ],
    random_state=[None],
    shuffle=[False],
    tol=[1e-4],
    eta0=[1e-5],
    early_stopping=[False],
    n_iter_no_change=[10],
    model=SGDClassifier,
)

models["mlp"] = dict(
    hidden_layer_sizes=[
        (100,),
        (50, 25),
    ],
    activation=[
        "relu",
    ],
    solver=[
        "adam",
    ],
    alpha=[1e-1, 1e-2, 1e-3, 1e-4],
    learning_rate_init=[1e-3, 1e-4],
    max_iter=[50, 100, 300, 500],
    batch_size=[
        "auto",
    ],
    shuffle=[
        True,
    ],
    random_state=[
        None,
    ],
    tol=[
        1e-4,
    ],
    n_iter_no_change=[
        10,
    ],
    beta_1=[
        0.9,
    ],
    beta_2=[
        0.999,
    ],
    early_stopping=[
        False,
    ],
    model=MLPClassifier,
)

models["decisionTree"] = dict(
    criterion=criterion,
    splitter=["best"],
    max_depth=list(range(7,15)),
    min_samples_split=[2,],
    min_samples_leaf=[1,],
    class_weight=[
        "balanced",
    ],
    max_features=[None],
    random_state=[None],
    model=DecisionTreeClassifier,
)

models["extraTrees"] = dict(
    criterion=criterion,
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=[2,],
    min_samples_leaf=[1,],
    class_weight=[
        "balanced",
    ],
    max_features=[None,'sqrt'],
    random_state=[None],
    oob_score=f1_score,
    bootstrap=[True,False],
    model=ExtraTreesClassifier,
)

models["randomForest"] = dict(
    n_estimators=n_estimators,
    criterion=criterion,
    max_depth=max_depth,
    min_samples_split=[2,],
    min_samples_leaf=[1,],
    class_weight=["balanced",],
    max_features=[
        "sqrt",
    ],
    random_state=[
        None,
    ],
    model=RandomForestClassifier,
)

models["balancedRandomForest"] = dict(
    n_estimators=n_estimators,
    criterion=criterion,
    max_depth=max_depth,
    min_samples_split=[2,],
    min_samples_leaf=[1,],
    class_weight=["balanced",],
    max_features=["sqrt",],
    random_state=[
        None,
    ],
    sampling_strategy=np.linspace(5e-2,2e-1,20).tolist(),
    model=BalancedRandomForestClassifier,
)

models["easyensemble"] = dict(
    n_estimators=n_estimators,
    warm_start=[False,True],
    sampling_strategy=np.linspace(5e-2,3e-1,20).tolist(),
    model=EasyEnsembleClassifier,
)

models["balancedBagging"] = dict(
    n_estimators=n_estimators,
    estimator=DecisionTreeClassifier(max_depth=3,class_weight='balanced'),
    warm_start=[False,True],
    sampler=SMOTE(),
    max_samples=[0.8,0.85,0.9,],
    max_features=[0.8,0.85,0.9,],
    sampling_strategy=np.linspace(5e-2,0.15,20).tolist(),
    model=BalancedBaggingClassifier,
)

models["rusboostclassifier"] = dict(
    estimator=DecisionTreeClassifier(max_depth=3,class_weight='balanced',max_features='sqrt'),
    n_estimators=n_estimators,
    learning_rate=0.023,
    sampling_strategy=np.linspace(0.05,0.15,10).tolist(),
    model=RUSBoostClassifier,
)

models["gradientBoosting"] = dict(
    loss=["log_loss",],
    n_estimators=list(range(150,800))[::25],
    learning_rate=0.023,
    subsample=[0.95,],
    criterion=["squared_error",],
    max_depth=[3,],
    min_samples_split=2,
    min_samples_leaf=3,
    max_features=['sqrt',],
    random_state=[None],
    tol=1e-4,
    model=GradientBoostingClassifier,
)

models['gaussianProcess'] = dict(model=GaussianProcessClassifier,
                                 n_restarts_optimizer=np.linspace(0,30,10).round().astype(int).tolist(),
                                 max_iter_predict=[50,100,200,300],
                                 warm_start=[True,]
                            )
# -> handles missig values
models["histGradientBoosting"] = dict(
    loss=[
        "log_loss",
    ],
    max_iter=[100, 500, 1000, 10000],
    learning_rate=learning_rate,
    max_depth=max_depth,
    l2_regularization=C,
    categorical_features=[
        "from_dtype",
    ],
    random_state=[
        None,
    ],
    max_bins=[
        255,
    ],
    class_weight=[
        "balanced",
    ],
    n_iter_no_change=[
        10,
    ],
    tol=[1e-7],
    model=HistGradientBoostingClassifier,
)

models["xgboost"] = dict(
    n_estimators=np.arange(10,500,10).tolist(),
    max_depth=2,
    learning_rate=0.1,
    booster=["gbtree",],
    objective=["binary:logistic"],
    tree_method=['hist',],
    scale_pos_weight=1e3,
    subsample=np.linspace(0.5,1,num=10).round(3).tolist(),
    max_bin=[
        255,
    ],
    colsample_bytree=np.linspace(0.1,0.5,num=10).round(3).tolist(),
    gamma=[
        0.0,
    ],
    reg_lambda=1e4,
    reg_alpha=[
        0.0,
    ],
    eval_metric=[
        "aucpr",
    ],
    importance_type=[
        "gain",
    ],
    random_state=[
        None,
    ],
    enable_categorical=[
        True,
    ],
    max_cat_to_onehot=[
        6,
    ],
    device=['cpu',],
    model=XGBClassifier,
)

models["catboost"] = dict(loss_function=['Logloss',],
                          eval_metric=['F1:use_weights=false',],
                          depth=2,
                          learning_rate=0.1,
                          subsample=0.73,
                          rsm=0.377,
                          l2_leaf_reg=1e4,
                          use_best_model=True,
                          early_stopping_rounds=50,
                          scale_pos_weight=1e3,
                          iterations=1000,
                          verbose=[0,],
                          model=CatBoostClassifier
                          )

models["lgbm"] = dict(
    n_estimators=np.arange(10,500,10).tolist(),
    boosting_type=['gbdt','rf',],
    objective=[
        "binary",
    ],
    class_weight=["balanced",],
    learning_rate=0.1,
    colsample_bytree=0.3,
    subsample=0.56,
    num_leaves=[31,],
    max_depth=2,
    verbosity=[-1],
    model=LGBMClassifier,
)

# TODO: For the models below, try only after optimizing the previous models
models["baggingClassifier"] = dict(
    estimator=[
        HistGradientBoostingClassifier(),
    ],
    n_estimators=n_estimators,
    max_samples=np.linspace(0.3, 0.95, num=10).round(3).tolist(),
    max_features=np.linspace(0.3, 0.95, num=10).round(3).tolist(),
    #    bootstrap=[True, False],
    #    bootstrap_features=[True, False],
    model=BaggingClassifier,
)

models["votingClassifier"] = dict(
    estimators=[
        HistGradientBoostingClassifier(),
        CatBoostClassifier(),
        LGBMClassifier(),
    ],
    voting=[
        "soft",
    ],
    model=VotingClassifier,
)

models["adaBoostClassifier"] = dict(
    estimator=DecisionTreeClassifier(max_depth=5,class_weight='balanced'),
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    model=AdaBoostClassifier,
)

models["stackingClassifier"] = dict(
    estimators=[
        [HistGradientBoostingClassifier(), CatBoostClassifier(), LGBMClassifier()],
    ],
    final_estimator=[LogisticRegression(solver='liblinear',C=1e3,class_weight='balanced'),],
    cv=[
        TimeSeriesSplit(n_splits=5, gap=5000),
    ],
    passthrough=[
        False,
    ],
    stack_method=[
        "auto",
    ],
    model=StackingClassifier,
)
# %% feature selector
feature_selector = dict()
# feature_select_estimator=DecisionTreeClassifier(max_depth=5,
#                                                 max_features=None,
#                                                 random_state=41, 
#                                                 class_weight="balanced")
# scoring = [
#     "f1",
# ]
# cv=TimeSeriesSplit(n_splits=3,gap=1000)
# # feature_selector["rfecv"] = dict(selector=RFECV,
# #                                  scoring=scoring,
# #                                  estimator=feature_select_estimator,
# #                                  step=list(range(1,10)),
# #                                  cv=[cv,]
# #                                  )


# feature_selector["sequential"] = dict(selector=SequentialFeatureSelector,
#                                       estimator=feature_select_estimator,
#                                       n_features_to_select=list(range(1,10)),
#                                       scoring=scoring,
#                                       tol=[1e-4],
#                                       cv=[cv,]
# )
feature_selector["selectkbest"] = dict(
    selector=SelectKBest,
    k=np.arange(5,20,step=3).tolist(),
)

# feature_selector["smartcorrelated"] = dict(
#     selector=SmartCorrelatedSelection,
#     method=['spearman',],
#     threshold=0.82,
#     scoring=['f1',],
#     estimator=DecisionTreeClassifier(max_depth=7,random_state=41,class_weight='balanced'),
#     cv=TimeSeriesSplit(n_splits=3, gap=5000),
# )
                                        
selectkbest_score_func=dict(
                            # f_classif=f_classif,
                            mutual_info_classif=mutual_info_classif,
                            # r_regression=r_regression
                            )