import joblib
from fraudetect.config import load_args_from_json
from fraudetect.preprocessing import FraudFeatureEngineer, FeatureEncoding
from fraudetect.dataset import MyDatamodule
from fraudetect.config import Arguments


if __name__ == "__main__":
    run = joblib.load(
        r"D:\fraud-detection-galsen\runs-optuna\small-models_2025-04-14_19-16_best-run.joblib"
    )
    results, transform_pipe, datamodule = run

    clf = results["decisionTree"].best_estimator_
    args, cfg = load_args_from_json(
        r"D:\fraud-detection-galsen\runs-optuna\small-models_2025-04-14_19-16.json"
    )

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

    # args.data_path = r"D:\fraud-detection-galsen\data\training.csv"

    # datamodule = MyDatamodule()
    # feature_engineer = FraudFeatureEngineer(windows_size_in_days=args.windows_size_in_days,
    #                                         uid_cols=None,
    #                                         session_gap_minutes=30,
    #                                         n_clusters=8
    #                                         )
    # encoding_kwargs = dict(n_components=14)
    # encoder = FeatureEncoding(cat_encoding_method=args.cat_encoding_method,
    #                         add_imputer=args.add_imputer,
    #                             onehot_threshold=9,
    #                             cols_to_drop=COLUMNS_TO_DROP,
    #                             n_jobs=1,
    #                             cat_encoding_kwards=encoding_kwargs
    #                             )
    # datamodule.setup(encoder=encoder, feature_engineer=feature_engineer)

    X_train, y = datamodule.get_train_dataset(args.data_path)

    X_pred, _ = datamodule.get_predict_dataset(
        r"D:\fraud-detection-galsen\data\test.csv"
    )

    y_pred = clf.predict(X_pred)
