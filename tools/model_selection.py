# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 16:51:56 2025

@author: FADELCO
"""
#%% funcs & imports
from fraudetect.modeling.utils import hyperparameter_tuning
from fraudetect.dataset import data_loader
from sklearn.model_selection import TimeSeriesSplit
from fraudetect.config import COLUMNS_TO_DROP, COLUMNS_TO_ONE_HOT_ENCODE, COLUMNS_TO_SCALE, Arguments
import numpy as np
from fraudetect.features import data_resampling, load_transforms_pyod
from fraudetect import import_from_path, sample_cfg
from collections import OrderedDict
import joblib
from functools import partial


def prepare_data(data_path:str, 
                 kwargs_tranform_data:dict,
                 delta_train=40, 
                 delta_delay=7, 
                 delta_test=20,
                 random_state=41,
                 sampler_names:list[str]=None,
                 sampler_cfgs:list[dict]=None,
                 ):
    
    # load and transform
    (X_train, y_train), prequential_split_indices  = data_loader(kwargs_tranform_data=kwargs_tranform_data,
                                                                data_path=data_path, 
                                                                split_method='prequential',
                                                                delta_train=delta_train, 
                                                                delta_delay=delta_delay, 
                                                                delta_test=delta_test,
                                                                n_folds=5,
                                                                random_state=random_state,
                                                                sampling_ratio=1.0
                                                                )
    print('Raw data shape: ', X_train.shape, y_train.shape)
    
    # Re-sample data    
    if (sampler_names is not None) and (sampler_cfgs is not None):
        X_train, y_train = data_resampling(X=X_train,y=y_train,
                                        sampler_names=sampler_names,
                                        sampler_cfgs=sampler_cfgs)
        print('Resampled data shape: ', X_train.shape, y_train.shape)
    
    return X_train, y_train

def tune_models_hyp(X_train:np.ndarray,
                    y_train:np.ndarray,
                    models_config:dict,
                    n_splits:int=5,
                    gap:int=1051*5,
                    n_iter:int=100,
                    scoring:str='f1',
                    verbose:int=0,
                    n_jobs:int=8,
                    method:str='random'):
    
    # 5 days, 1051 = average number of transactions per day
    
    assert isinstance(scoring, str), "It should be a string."
    
    cv= TimeSeriesSplit(n_splits=n_splits, gap=gap) 
    
    best_results = dict()
    for model_name in models_config.keys():
        try:
            print(f"Hyperparameter tuning for model: {model_name}")
            
            search_engine = hyperparameter_tuning(cv=cv, 
                                                config=models_config,
                                                X_train=X_train,
                                                y_train=y_train,
                                                model_name=model_name, 
                                                scoring=scoring,
                                                method=method, # other, gridsearch
                                                verbose=verbose,
                                                n_iter=n_iter,
                                                n_jobs=n_jobs,
                                                )
            best_results[model_name] = [search_engine]
            
            score = search_engine.best_score_
            print(f"mean {scoring} score: {score}")
            
        except Exception as e:
            print(e)
            continue
    
    return best_results

def load_models_cfg(names:list[str]):
    return {name:HYP_CONFIGS.models[name] for name in names}

def run(args:Arguments, outliers_det_configs:dict=None,save_path:str=None,verbose=1):
    
    # args
    data_path = args.data_path
    
    delta_train=args.delta_train
    delta_delay=args.delta_delay
    delta_test=args.delta_test
    random_state=args.random_state
    windows_size_in_days=args.windows_size_in_days
    sampler_names = args.sampler_names
    sampler_cfgs = args.sampler_cfgs
    pyod_predict_proba_method=args.pyod_predict_proba_method
    model_names = args.model_names
    n_iter=args.n_iter
    cv_gap=args.cv_gap
    cv_method=args.cv_method
    n_splits=args.n_splits
    n_jobs=args.n_jobs
    scoring=args.scoring
    
    # load data & do basic preprocessing
    kwargs_tranform_data = dict(columns_to_drop=COLUMNS_TO_DROP,
                                columns_to_onehot_encode=COLUMNS_TO_ONE_HOT_ENCODE,
                                columns_to_scale=COLUMNS_TO_SCALE,
                                windows_size_in_days=windows_size_in_days,
                                train_transform=None, # some custom transform applied to X_train,y_train
                                val_transform=None, # some custom transform applied to X_val,y_val
                                delay_period_accountid=delta_delay,
                                )
    X_train, y_train = prepare_data(data_path=data_path, 
                                    kwargs_tranform_data=kwargs_tranform_data,
                                     delta_train=delta_train, 
                                     delta_delay=delta_delay, 
                                     delta_test=delta_test,
                                     random_state=random_state,
                                     sampler_names=sampler_names,
                                     sampler_cfgs=sampler_cfgs,
                                     )
    
    # load pyod transform and apply it to X_train
    if outliers_det_configs is not None:
        transform = load_transforms_pyod(X_train=X_train,
                                         outliers_det_configs=outliers_det_configs,
                                         method=pyod_predict_proba_method
                                         )    
        X_train = transform(X=X_train)
    
    # tune models
    best_results = tune_models_hyp(X_train,
                                   y_train,
                                    models_config=load_models_cfg(names=model_names),
                                    n_splits=n_splits,
                                    gap=cv_gap,
                                    n_iter=n_iter,
                                    scoring=scoring,
                                    verbose=verbose,
                                    n_jobs=n_jobs,
                                    method=cv_method
                                )
    # save results
    if save_path:
        joblib.dump(best_results,save_path)
        
    return best_results
    
# sample cfg for resamplers
def get_samplers_cfgs(sampler_names, configs):
    
    sampler_cfgs = list()
    
    for name in sampler_names:
        
        if name in configs.under_sampler.keys():
            cfg = configs.under_sampler[name]
            
        elif name in configs.over_sampler.keys():
            cfg = configs.over_sampler[name]
            
        elif name in configs.combined_sampler.keys():
            cfg = configs.combined_sampler[name]
        
        cfg = sample_cfg(cfg) # for test purposes
        sampler_cfgs.append({name:cfg})
        
    return sampler_cfgs
#%% Run
if __name__ == "__main__":
    
    
    try:
        HYP_CONFIGS = import_from_path('hyp_search_conf',
                                   r'D:\fraud-detection-galsen\tools\hyp_search_conf.py')
    except:
        HYP_CONFIGS = import_from_path('hyp_search_conf',
                                   r'D:\fraud-detection-galsen\tools\hyp_search_conf.py')
    
    args = Arguments()
    args.data_path = r"D:\fraud-detection-galsen\data\training.csv"
    args.model_names = ('logisticReg', 'xgboost', 'randomForest','histGradientBoosting')
    args.pyod_detectors = ('iforest', 'cblof', 'loda', 'knn')
    args.n_iter = 100
    disable_pyod_outliers = True
    
    # sample cfg randomly for debugging
    outliers_det_configs = []
    names = args.pyod_detectors
    for name in sorted(names):
        cfg = HYP_CONFIGS.outliers_detectors[name]
        cfg = sample_cfg(cfg)
        outliers_det_configs.append((name,cfg))

    outliers_det_configs = OrderedDict(outliers_det_configs)
    
    if disable_pyod_outliers:
        outliers_det_configs = None
    
    

    args.sampler_cfgs = get_samplers_cfgs(args.sampler_names, HYP_CONFIGS)
    
    # run hyperparameter search
    results = run(args=args, 
                  outliers_det_configs=outliers_det_configs,
                  save_path=None,
                  verbose=0)
    
    
    
    
    
    
    
    
    
  
    