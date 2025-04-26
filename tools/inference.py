import joblib
from fraudetect.config import load_args_from_json
from fraudetect.dataset import load_data

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, date
from sklearn.model_selection import (TimeSeriesSplit,GroupKFold,
                                     TunedThresholdClassifierCV)
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
import os
import json

from sklearn.ensemble import (
    StackingClassifier
)

if __name__ == "__main__":
        
    clf_path = r"D:\fraud-detection-galsen\runs-optuna\ensemble-trees-1_2025-04-21_20-00_best-run.joblib"
    
    run = joblib.load(clf_path)

    # args, cfg = load_args_from_json(
    # r"D:\fraud-detection-galsen\runs-optuna\ensemble-trees-1_2025-04-21_20-00_best-run.joblib"
    # )   

    # clf = run[0].best_estimator_ #[0]
    

    raw_data_train = load_data("../data/training.csv")

    raw_data_pred = load_data("../data/test.csv")
    
    X = raw_data_train.drop(columns=['TX_FRAUD'])
    y = raw_data_train['TX_FRAUD']
    
    cv=TimeSeriesSplit(n_splits=4,gap=5000)

    # tune threshold
    # clf_tuned = TunedThresholdClassifierCV(clf,
    #                                 scoring='f1',
    #                                 cv=cv,
    #                             )
    # clf_tuned.fit(X,y)
    
    # Stacking
    final_estimator=LogisticRegressionCV(Cs=np.logspace(-1,4,10),cv=cv,
                                         scoring='average_precision',
                                         solver='liblinear',
                                         )
    clf_stacking =  StackingClassifier([(str(i),pipe) for i,pipe in enumerate(run)],
                                       final_estimator=final_estimator,
                                          n_jobs=5,
                                          cv='prefit')
    clf_stacking.fit(X,y)
    
    
    # calibrated model
    # clf_calibrated = CalibratedClassifierCV(FrozenEstimator(clf_stacking),
    #                                 method='sigmoid',
    #                                 n_jobs=2,
    #                                 ensemble=True,
    #                                 cv=TimeSeriesSplit(n_splits=5,gap=0),
    #                             )
    # clf_calibrated.fit(X,y)
    
# =============================================================================
#     # Predict
# =============================================================================
    # y_pred_origin = clf.predict(raw_data_pred)

    # y_pred_calibrated = clf_calibrated.predict(raw_data_pred)

    y_pred_stacked = clf_stacking.predict(raw_data_pred)

    for pred in [y_pred_stacked,]:
        print(pred.sum(), pred.sum()/pred.shape[0])

    submission = pd.read_csv("../data/sample_submission.csv")
    
    submission['FraudResult'] = y_pred_stacked
    submission['FraudResult'] = submission['FraudResult'].astype(int)

    print(submission['FraudResult'].sum())

    current_time = datetime.now().strftime("%H-%M")
    filename = Path(clf_path).with_suffix('.csv').name
    filename = os.path.join("../submissions",filename)

    submission.to_csv(filename,index=False)

