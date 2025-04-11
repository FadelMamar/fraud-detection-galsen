import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from xgboost import XGBClassifier

from imblearn.under_sampling import NearMiss, EditedNearestNeighbours
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

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


# %% data aug - pyod
outliers_detectors = dict()

_cnt = np.logspace(-4, -2, 10).tolist()

outliers_detectors["iforest"] = dict(
    detector=IForest,
    n_estimators=[20, 50, 100, 200],
    contamination=_cnt,
    n_jobs=[8],
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
    n_jobs=[8],
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
    n_jobs=[8],
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
fracs = (np.arange(1, 5) * 4e-3).tolist()
samplers["nearmiss"] = dict(
    sampling_strategy=fracs,
    n_neighbors=[3, 5, 7, 9],
    version=[1],
    n_jobs=[8],
    sampler=NearMiss,
)
# oversampling
fracs = (np.arange(1, 3) * 4e-3).tolist()
samplers["SMOTE"] = dict(
    sampling_strategy=fracs, random_state=[41], k_neighbors=[3, 5, 7, 9], sampler=SMOTE
)

samplers["adasyn"] = dict(
    sampling_strategy=fracs, random_state=[41], n_neighbors=[3, 5, 7, 9], sampler=ADASYN
)

samplers["borderlineSMOTE"] = dict(
    sampling_strategy=fracs,
    random_state=[41],
    m_neighbors=[3, 5, 7, 9],
    k_neighbors=[5, 7, 10, 15, 20],
    kind=["borderline-1", "borderline-2"],
    sampler=BorderlineSMOTE,
)

samplers["svmSMOTE"] = dict(
    sampling_strategy=fracs,
    random_state=[41],
    m_neighbors=[5, 7, 10, 15, 20],
    out_step=[0.1, 0.2, 0.5, 0.7],
    k_neighbors=[3, 5, 7, 9],
    kind=["borderline-1", "borderline-2"],
    sampler=SVMSMOTE,
)

# combined sampler
samplers["smoteENN"] = dict(
    sampling_strategy=fracs,
    random_state=[41],
    enn=EditedNearestNeighbours(
        sampling_strategy="majority", n_neighbors=5, kind_sel="all", n_jobs=8
    ),
    n_jobs=[8],
    sampler=SMOTEENN,
)

samplers["smoteTOMEK"] = dict(
    sampling_strategy=fracs, random_state=[41], n_job=[8], sampler=SMOTETomek
)

undersamplers = ["nearmiss",]
oversamplers = ["SMOTE", "adasyn", "borderlineSMOTE", "svmSMOTE",]
combinedsamplers = ["smoteENN", "smoteTOMEK",]


# %% models
models = dict()

models["logisticReg"] = dict(
    penalty=["l2"],
    C=np.logspace(-4, 4, 10).tolist(),
    class_weight=["balanced", None],
    solver=["liblinear"],
    max_iter=[int(1e4)],
    random_state=[None],
    tol=[1e-4],
    model=LogisticRegression,
)

models["svc"] = dict(
    C=np.logspace(-4, 4, 10).tolist(),
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
    C=np.logspace(-4, 4, 10).tolist(),
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
    alpha=np.logspace(-4, 4, 10).tolist(),
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

models["decisionTree"] = dict(
    criterion=["gini", "entropy", "log_loss"],
    splitter=["best"],
    max_depth=[None, 5, 7, 9, 10, 15, 20],
    min_samples_split=[2, 3, 4],
    min_samples_leaf=[1, 2],
    class_weight=[
        "balanced",
    ],
    max_features=["sqrt", "log2", None],
    random_state=[None],
    model=DecisionTreeClassifier,
)

models["randomForest"] = dict(
    n_estimators=[10, 20, 50, 100, 200, 500],
    criterion=["gini", "entropy", "log_loss"],
    max_depth=[None, 5, 7, 9, 10, 15, 20],
    min_samples_split=[2, 3, 4],
    min_samples_leaf=[1, 2],
    class_weight=["balanced", "balanced_subsample"],
    max_features=["sqrt", "log2", None],
    random_state=[None],
    n_jobs=[8],
    model=RandomForestClassifier,
)

models["gradientBoosting"] = dict(
    loss=["log_loss", "exponential"],
    n_estimators=[10, 50, 100, 200, 500],
    learning_rate=np.logspace(-4, -1, 10).tolist(),
    subsample=[0.5, 0.75, 1.0],
    criterion="friedman_mse",
    max_depth=[None, 3, 5, 7, 9, 10, 15, 20],
    min_samples_split=[2, 3, 4],
    min_samples_leaf=[1, 2],
    max_features=["sqrt", "log2", None],
    random_state=[None],
    tol=1e-4,
    model=GradientBoostingClassifier,
)

# -> handles missig values
models["histGradientBoosting"] = dict(
    loss=["log_loss"],
    max_iter=[100, 500, 1000, 10000],
    learning_rate=np.logspace(-4, -1, 10).tolist(),
    max_depth=[None, 3, 5, 7, 9, 10, 15, 20],
    l2_regularization=np.logspace(-4, 4, 10).tolist(),
    categorical_features=[None],
    random_state=[None],
    max_bins=[2**5 - 1, 2**6 - 1, 2**7 - 1, 2**8 - 1],
    class_weight=[
        "balanced",
    ],
    n_iter_no_change=[10],
    tol=[1e-7],
    model=HistGradientBoostingClassifier,
)

models["xgboost"] = dict(
    n_estimators=[10, 20, 50, 75, 100],
    max_depth=[3, 6, 9],
    learning_rate=[1e-1, 1e-2, 1e-3],
    booster=["gbtree", "gblinear", "dart"],
    n_jobs=[8],
    objective=["binary:hinge", "binary:logistic"],
    tree_method=["hist"],
    scale_pos_weight=np.logspace(0, 3, 5).tolist(),
    subsample=[0.5, 0.75, 0.85, 1.0],
    colsample_bytree=[0.5, 0.75, 0.85, 1.0],
    gamma=[0.0],
    reg_lambda=[1.0],
    reg_alpha=[0.0],
    eval_metric=["error"],
    importance_type=["gain"],
    random_state=[41],
    enable_categorical=[
        True,
    ],
    model=XGBClassifier,
)

# #TODO: For the models below, try only after optimizing the previous model
# models['baggingClassifier'] = dict(estimator=[RandomForestClassifier(),
#                                               HistGradientBoostingClassifier(),
#                                               DecisionTreeClassifier(),
#                                               LinearSVC(),
#                                               SGDClassifier()],
#                                    n_estimators=[3, 5, 7, 15],
#                                    max_samples=[0.5, 0.75, 1.0],
#                                    max_features=[0.5, 0.75, 1.0],
#                                    bootstrap=[True, False],
#                                    bootstrap_features=[True, False],
#                                    random_state=[41],
#                                    model=BaggingClassifier,
#                                 )

# models['votingClassifier'] = dict(estimators=[...],
#                                    voting=['soft'],
#                                    random_state=[41],
#                                    model=VotingClassifier,
#                                 )

# models['adaBoostClassifier'] = dict(estimator=[RandomForestClassifier(),
#                                               HistGradientBoostingClassifier(),
#                                               DecisionTreeClassifier(),
#                                               LinearSVC(),
#                                               SGDClassifier()],
#                          n_estimators=[10,50,75,100],
#                          learning_rate=np.logspace(-4, -1, 10).tolist(),
#                          algorithm=['SAMME'],
#                          random_state=[41],
#                          model=AdaBoostClassifier,
#                         )

# models['stackingClassifier'] = dict(estimators=[...],
#                                  final_estimator=[LogisticRegression(),SVC(),SGDClassifier()],
#                                  cv=[None],
#                                  stack_method=['auto'],
#                                  passthrough=[False],
#                                  verbose=[0],
#                                  model=StackingClassifier,
#                               )
