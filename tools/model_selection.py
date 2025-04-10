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
from fraudetect.features import load_transforms_pyod
from fraudetect.sampling import data_resampling
from fraudetect import import_from_path, sample_cfg
from collections import OrderedDict
import joblib
from functools import partial
import optuna
from optuna.samplers import TPESampler
from collections.abc import Iterable

try:
    HYP_CONFIGS = import_from_path('hyp_search_conf',
                               r'D:\fraud-detection-galsen\tools\hyp_search_conf.py')
except:
    HYP_CONFIGS = import_from_path('hyp_search_conf',
                               r'D:\fraud-detection-galsen\tools\hyp_search_conf.py')
    
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
            print(f"mean {scoring} score for best_estimator: {score:.4f}")
            print("best params: ",search_engine.best_params_ )
            
        except Exception as e:
            print(e)
            continue
    
    return best_results

def load_models_cfg(names:list[str]):
    return {name:HYP_CONFIGS.models[name] for name in names}

def run(args:Arguments, save_path:str=None,verbose=1):
    
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
    if args.outliers_det_configs is not None:
        transform = load_transforms_pyod(X_train=X_train,
                                         outliers_det_configs=args.outliers_det_configs,
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
    

#%% Run

# helpers for debugging
def get_samplers_cfgs(sampler_names, configs):
    
    sampler_cfgs = list()
    
    for name in sampler_names:
        cfg = configs.samplers[name]
        cfg = sample_cfg(cfg) # random sampling
        sampler_cfgs.append({name:cfg})
        
    return sampler_cfgs

def get_pyod_cfgs(names,configs):
    
    # sample cfg randomly for debugging
    outliers_det_configs = []
    names = args.pyod_detectors
    for name in sorted(names):
        cfg = configs.outliers_detectors[name]
        cfg = sample_cfg(cfg)
        outliers_det_configs.append((name,cfg))

    return OrderedDict(outliers_det_configs)
    

def objective_optuna(trial):
    
    def sample_cfg_optuna(name:str,cfg:dict):
        for k in cfg.keys():
            if not isinstance(cfg[k], Iterable):
                cfg[k] = [k,]
            cfg[k] = trial.suggest_categorical(name+'__'+k, cfg[k])
        return cfg
    
    args = Arguments()
    args.data_path = r"D:\fraud-detection-galsen\data\training.csv"
    pyod_detectors = ('iforest', 'cblof', 'loda', 'knn')
    model_names = ('logisticReg','decisionTree','linearSVC','svc')
    args.n_iter = 100
    args.disable_pyod_outliers = False
    
    # select model
    model_name = trial.suggest_categorical("classifier", model_names # HYP_CONFIGS.models.keys()
                                                  )
    args.model_names = [model_name,]
    
    # select outlier detector for data aug
    args.disable_pyod_outliers = trial.suggest_categorical("disable_pyod",[False,False])
    if args.disable_pyod_outliers:
        args.outliers_det_configs = None
    else:
         pyod_detector_name =  trial.suggest_categorical("pyod_det", pyod_detectors)
         args.pyod_detectors = [pyod_detector_name,]
         args.outliers_det_configs = OrderedDict()
         args.outliers_det_configs[pyod_detector_name] = sample_cfg_optuna(pyod_detector_name, 
                                                                           HYP_CONFIGS.outliers_detectors[pyod_detector_name]
                                                                           )
                        
        
    # samplers
    args.sampler_cfgs = None
    args.sampler_names = None
    if trial.suggest_categorical("disable_samplers",[True,False]):
        conbimed_sampler = trial.suggest_categorical("conbined_sampler",HYP_CONFIGS.combinedsamplers
                                                          )
        sampler_names = [conbimed_sampler]
        if conbimed_sampler is None:
            oversampler = trial.suggest_categorical("over_sampler",HYP_CONFIGS.oversamplers
                                                              )
            undersampler = trial.suggest_categorical("under_sampler",HYP_CONFIGS.undersamplers
                                                              )
            sampler_names = [oversampler, undersampler]       
        
        # get sampler config
        args.sampler_cfgs = []
        for name in sampler_names:
            _cfg = {name:sample_cfg_optuna(name, HYP_CONFIGS.samplers[name])}
            args.sampler_cfgs.append(_cfg)                                                                     
    
    # run cv
    results = run(args=args,save_path=None,verbose=0)
    score = results[model_name].best_score_
    
    return score

def demo(args:Arguments):
        
    # sample cfg randomly for debugging
    if args.sampler_names is not None:
        args.sampler_cfgs = get_samplers_cfgs(args.sampler_names, HYP_CONFIGS)
    
    if args.disable_pyod_outliers:
        args.outliers_det_configs = None
    else:
        args.outliers_det_configs = get_pyod_cfgs(args.pyod_detectors,HYP_CONFIGS)

    # run hyperparameter search
    results = run(args=args,
                  save_path=None,
                  verbose=0)
    
    return results

if __name__ == "__main__":
    
    # Debugging
    # args = Arguments()
    # args.data_path = r"D:\fraud-detection-galsen\data\training.csv"
    # args.model_names = ('logisticReg','decisionTree','linearSVC','svc') #'gradientBoosting','randomForest',)
    # args.pyod_detectors = ('iforest', 'cblof', 'loda', 'knn')
    # args.sampler_names = ['SMOTE','nearmiss']
    # args.n_iter = 100
    # args.disable_pyod_outliers = True
    # demo(args=args)
    
    
    # using optuna
    study = optuna.create_study(direction='maximize',
                                sampler=TPESampler(multivariate=True,group=True),
                                study_name='demo',
                                load_if_exists=True,
                                storage="sqlite:///hypsearch.sql"
                                )
    study.optimize(objective_optuna, n_trials=100)
    print(study.best_trial)
    
    
    
    
    
    
    
    
    
  
    