from dataclasses import dataclass
import json
from typing import Sequence
import traceback


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
COLUMNS_TO_ONE_HOT_ENCODE = [
    "PricingStrategy",
    "ChannelId",
    "ProviderId",
]
COLUMNS_TO_CAT_ENCODE = [
    "ProductCategory",
    "ProductId",
]
COLUMNS_TO_STD_SCALE = [
    "TX_DURING_WEEKEND",
    "TX_DURING_NIGHT",
    "Value",
    "TX_AMOUNT",
    "TX_HOUR",
]
COLUMNS_TO_ROBUST_SCALE = []


@dataclass
class Arguments:
    data_path: str = ""

    # optuna
    study_name: str = "demo"

    work_dir: str = None

    run_name: str = "debug"

    reorder_by:Sequence=('TX_DATETIME',)

    # data pre-processing
    delta_train: int = 40
    delta_delay: int = 7
    delta_test: int = 20
    random_state: int = 41
    windows_size_in_days: Sequence = (1, 7, 30)
    sampler_names: Sequence = None
    sampler_cfgs: Sequence = None
    outliers_det_configs = None
    model_names: Sequence = (
        "sgdClassifier",
        "xgboost",
        "randomForest",
        "histGradientBoosting",
    )
    session_gap_minutes: int = 30
    onehot_threshold: int = 9
    pyod_detectors: Sequence = ("iforest", "cblof", "loda", "knn")
    disable_pyod_outliers: bool = False
    disable_samplers: bool = False
    do_pca: bool = False  #  try pca
    do_poly_expansion: bool = False # not used. does do anything
    do_feature_selection: bool = False
    cv_n_iter: int = 20  # for cross validation
    cv_gap: int = 1051 * 5
    cv_method: str = "optuna"
    n_splits: int = 5
    n_jobs: int = 8
    scoring: str = "f1"
    cat_encoding_method: str = "binary"  # count, binary, base_n, hashing, 'None'
    cat_encoding_base_n: int = 4
    cat_encoding_hash_method: str = "md5"
    cat_encoding_hash_n_components: int = 8
    add_imputer: bool = False
    concat_features: Sequence = (
        None,
    )  # ("AccountId", "CUSTOMER_ID")  # or None to disable
    concat_features_encoding_kwargs = dict(
        cat_encoding_method="hashing", n_components=14
    )
    optuna_n_trials: int = 50

    add_fft:bool=False
    add_seasonal_features:bool=False
    use_nystrom:bool=False
    use_sincos:bool=False
    use_spline:bool=False

    cols_to_drop: Sequence[str] = None
    # training parameters
    # max_epochs: int = 50
    # learning_rate: float = 1e-3
    # weightdecay: float = 1e-4

    # data augmentation


def load_args_from_json(path: str):
    args = Arguments()

    with open(path, "r") as file:
        cfg = json.load(file)

    for k in cfg["args"].keys():
        try:
            args.__setattr__(k, cfg["args"][k])
        except KeyError:
            print("\nFailed to load: ", k)
            traceback.print_exc()

    return args, cfg
