from dataclasses import dataclass

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
    "TX_HOUR"
]
COLUMNS_TO_STD_SCALE = [
    "TX_DURING_WEEKEND",
    "TX_DURING_NIGHT",
    "Value",
    "TX_AMOUNT",
]
COLUMNS_TO_ROBUST_SCALE = []


@dataclass
class Arguments:
    data_path: str = ""

    # optuna
    study_name: str = "demo"

    run_name: str = "debug"

    # data pre-processing
    delta_train: int = 40
    delta_delay: int = 7
    delta_test: int = 20
    random_state: int = 41
    windows_size_in_days = (1, 7, 30)
    sampler_names = None
    sampler_cfgs = None
    outliers_det_configs = None
    model_names = ("sgdClassifier", "xgboost", "randomForest", "histGradientBoosting")
    pyod_detectors = ("iforest", "cblof", "loda", "knn")
    disable_pyod_outliers = False
    disable_samplers = False
    cv_n_iter = 20  # for cross validation
    cv_gap = 1051 * 5
    cv_method = "optuna"
    n_splits = 5
    n_jobs = 8
    scoring = "f1"
    cat_encoding_method: str = "binary"  # count, binary, base_n, hashing
    cat_encoding_base_n: int = 4
    cat_encoding_hash_method: str = "md5"
    cat_encoding_hash_n_components: int = 8
    add_imputer: bool = False
    concat_features = ("AccountId", "CUSTOMER_ID")  # or None to disable
    concat_features_encoding_kwargs = dict(
        cat_encoding_method="hashing", n_components=14
    )
    optuna_n_trials: int = 50
    # training parameters
    # max_epochs: int = 50
    # learning_rate: float = 1e-3
    # weightdecay: float = 1e-4

    # data augmentation
