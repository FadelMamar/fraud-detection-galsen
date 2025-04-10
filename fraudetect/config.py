from dataclasses import dataclass

COLUMNS_TO_DROP = ['CurrencyCode','CountryCode','SubscriptionId','BatchId','CUSTOMER_ID','AccountId','TRANSACTION_ID','TX_DATETIME','TX_TIME_DAYS']
COLUMNS_TO_ONE_HOT_ENCODE = ['PricingStrategy','ChannelId', 'ProductCategory', 'ProviderId', 'ProductId']
COLUMNS_TO_SCALE = ['TX_DURING_WEEKEND','TX_DURING_NIGHT','TX_AMOUNT','Value'] # or None to select all numeric columns

@dataclass
class Arguments:

    data_path:str=""
    
    # data pre-processing
    delta_train:int=40 
    delta_delay:int=7 
    delta_test:int=20
    random_state:int=41
    windows_size_in_days=(1,7,30)
    sampler_names = None
    sampler_cfgs = None
    pyod_predict_proba_method='unify' # unify or linear
    model_names = ('sgdClassifier', 'xgboost', 'randomForest','histGradientBoosting')
    pyod_detectors = ('iforest', 'cblof', 'loda', 'knn')
    n_iter=20
    cv_gap=1051*5
    cv_method='random'
    n_splits=5
    n_jobs=8
    scoring='f1'    

    # training parameters
    max_epochs:int=50
    learning_rate:float=1e-3
    weightdecay:float=1e-4

    # data augmentation



