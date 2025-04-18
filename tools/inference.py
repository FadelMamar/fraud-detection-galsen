import joblib
from fraudetect.config import load_args_from_json
from fraudetect.dataset import load_data

from pathlib import Path
import pandas as pd
from datetime import datetime, date
from sklearn.model_selection import (TimeSeriesSplit,
                                     TunedThresholdClassifierCV)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
import os
import json


if __name__ == "__main__":
        
    clf_path = r"..\runs-optuna\cat-models_2025-04-17_01-04_best-run.joblib"
    
    run = joblib.load(clf_path)

    args, cfg = load_args_from_json(
    r"D:\fraud-detection-galsen\runs-optuna\cat-models_2025-04-18_16-41.json"
    )   

    clf = run[0].best_estimator_ #[0]

    raw_data_train = load_data("../data/training.csv")

    raw_data_pred = load_data("../data/test.csv")
    
    X = raw_data_train.drop(columns=['TX_FRAUD'])
    y = raw_data_train['TX_FRAUD']

    # tune threshold
    clf_tuned = TunedThresholdClassifierCV(clf,
                                    scoring='f1',
                                    cv=TimeSeriesSplit(n_splits=5,gap=0),
                                )
    clf_tuned.fit(X,y)

    # calibrated model
    clf_calibrated = CalibratedClassifierCV(FrozenEstimator(clf),
                                    method='sigmoid',
                                    n_jobs=2,
                                    ensemble=True,
                                    cv=TimeSeriesSplit(n_splits=5,gap=0),
                                )
    clf_calibrated.fit(X,y)
    
    # Predict
    y_pred_origin = clf.predict(raw_data_pred)

    y_pred_calibrated = clf_calibrated.predict(raw_data_pred)

    y_pred_tuned = clf_tuned.predict(raw_data_pred)

    for pred in [y_pred_origin, y_pred_calibrated, y_pred_tuned]:
        print(pred.sum(), pred.sum()/pred.shape[0])

    submission = pd.read_csv("../data/sample_submission.csv")
    
    submission['FraudResult'] = y_pred_calibrated
    submission['FraudResult'] = submission['FraudResult'].astype(int)

    # print(submission['FraudResult'].sum())

    current_time = datetime.now().strftime("%H-%M")
    filename = Path(clf_path).with_suffix('.csv').name
    filename = os.path.join("../submissions",filename)

    # submission.to_csv(filename,index=False)

