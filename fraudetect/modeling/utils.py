from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
)
from pathlib import Path
from datetime import datetime, date

from copy import deepcopy
from itertools import combinations
from sklearn.model_selection._search import BaseSearchCV
import json, os, joblib, traceback
import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (TimeSeriesSplit,
                                     TunedThresholdClassifierCV,
                                     cross_val_score,
                                     cross_validate)
from sklearn.tree import DecisionTreeClassifier
import random
from collections.abc import Iterable
from typing import Sequence
import optuna
from optuna.samplers import TPESampler
from fraudetect import import_from_path
from catboost import Pool, EShapCalcType, EFeaturesSelectionAlgorithm

from ..preprocessing import get_train_val_split
from ..config import Arguments
from ..dataset import load_data
from ..preprocessing import load_workflow, get_feature_selector
from ..detectors import get_detector, instantiate_detector
from ..sampling import get_sampler, instantiate_sampler

# try:
#     import fireducks.pandas as pd
#     # print('importing fireducks.pandas as pd')
# except:
#     import pandas as pd


def evaluate(classifier, 
             X, 
             y, 
             metrics = ["f1", "average_precision", "precision", "recall"]):
    out = {metric:get_scorer(metric)(classifier, X, y) for metric in metrics}
    return out


def get_model(model_name: str, config: dict) -> dict:
    """
    Get the model configuration based on the model name.
    """
    if model_name not in config:
        raise ValueError(f"Model {model_name} not found in config.")

    model_cfg = config[model_name].copy()

    # Remove the model from the config dictionary
    model = model_cfg.pop("model")

    return model, model_cfg


def sample_model_cfg(model_cfg: dict) -> dict:
    """
    Sample a model configuration from the given model configuration dictionary.
    """
    sampled_cfg = {}
    for key, value in model_cfg.items():
        if isinstance(value, Sequence):
            sampled_cfg[key] = random.choice(value)
        else:
            sampled_cfg[key] = value
    return sampled_cfg


def instantiate_model(model, **kwargs):
    """
    Instantiate the model with the given parameters.
    """
    return model(**kwargs)


def hyperparameter_tuning(
    cv,
    params_config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model,
    scoring: str = "f1",
    n_iter: int = 50,
    n_jobs: int = 8,
    verbose: int = 0,
    method: str = "optuna",
) -> BaseSearchCV:
    if verbose > 0:
        print(json.dumps(params_config, indent=4))

    if method == "gridsearch":
        search_engine = GridSearchCV(
            model,
            param_grid=params_config,
            scoring=scoring,
            cv=cv,
            refit=True,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    elif method == "random":
        search_engine = RandomizedSearchCV(
            model,
            param_distributions=params_config,
            scoring=scoring,
            cv=cv,
            refit=True,
            n_jobs=n_jobs,
            n_iter=n_iter,
            verbose=verbose,
        )

    elif method == "optuna":
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(multivariate=True, group=True),
            load_if_exists=True,
        )
        param_dist = dict()
        for k, v in params_config.items():
            if isinstance(v, Sequence):
                param_dist[k] = optuna.distributions.CategoricalDistribution(v)
        
        search_engine = optuna.integration.OptunaSearchCV(
            model,
            param_distributions=param_dist,
            cv=cv,
            refit=True,
            n_jobs=n_jobs,
            study=study,
            scoring=scoring,
            error_score='raise',
            max_iter=300,
            timeout=60 * 3,
            n_trials=n_iter,
            # random_state=41,
            verbose=verbose,
        )

    else:
        raise ValueError(
            "Invalid method. Choose either 'gridsearch',hyperopt, 'optuna' or 'random'."
        )

    search_engine.fit(X_train, y_train)

    return search_engine


# show result
def results_from_search(search_engine, performance_metrics_list=["score"]):
    performances_df = pd.DataFrame()
    expe_type = "val"
    performance_metrics_list_grid = performance_metrics_list

    for i in range(len(performance_metrics_list_grid)):
        performances_df[performance_metrics_list[i] + " " + expe_type] = (
            search_engine.cv_results_["mean_test_" + performance_metrics_list_grid[i]]
        )
        performances_df[performance_metrics_list[i] + " " + expe_type + " Std"] = (
            search_engine.cv_results_["std_test_" + performance_metrics_list_grid[i]]
        )
    performances_df["Execution time"] = search_engine.cv_results_["mean_fit_time"]
    performances_df["Parameters"] = list(search_engine.cv_results_["params"])

    return performances_df


def _tune_models_hyp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    classifier,
    params_config,
    n_splits: int = 5,
    gap: int = 1051 * 5,
    n_iter: int = 100,
    scoring: str = "f1",
    verbose: int = 0,
    n_jobs: int = 8,
    method: str = "random",
) -> BaseSearchCV:
    # 5 days, 1051 = average number of transactions per day

    assert isinstance(scoring, str), "It should be a string."

    cv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    search_engine = hyperparameter_tuning(
        cv=cv,
        params_config=params_config,
        X_train=X_train,
        y_train=y_train,
        model=classifier,
        scoring=scoring,
        method=method,  # other, gridsearch
        verbose=verbose,
        n_iter=n_iter,
        n_jobs=n_jobs,
    )

    if verbose:
        score = search_engine.best_score_
        print(f"mean {scoring} score for best_estimator: {score:.4f}")
        # print("best params: ", search_engine.best_params_)

    return search_engine  # {model_name: search_engine}  #



class Tuner(object):
    def __init__(
        self,
        args: Arguments,
        verbose: int = 0,
        val_window_days=30,
        split_id_column="AccountId",
        cat_encoding_kwards: dict = {},
        feature_selector_kwargs: dict = {},
        tune_threshold:bool=False,
        iterate_cat_method:bool=False,
        iterate_session_gap:bool=False,
    ):
        self.HYP_CONFIGS = None

        self.args = args
        self.pyod_detectors = sorted(self.args.pyod_detectors)
        self.pyod_choices = [
            json.dumps(list(k))
            for k in combinations(self.pyod_detectors, min(4, len(args.pyod_detectors)))
        ]
        self.model_names = deepcopy(args.model_names)
        self.disable_pyod = deepcopy(args.disable_pyod_outliers)
        self.disable_samplers = deepcopy(args.disable_samplers)
        self.verbose = verbose
        self.count_iter = 0
        self.feature_selector_kwargs = feature_selector_kwargs
        self.cat_encoding_kwards = cat_encoding_kwards
        self.selector = None
        self.iterate_cat_method=iterate_cat_method
        self.iterate_session_gap=iterate_session_gap

        raw_data_train = load_data(args.data_path)

        if self.args.do_train_val_split:
            self.X_train, self.y_train, self.X_val, self.y_val = get_train_val_split(train_data=raw_data_train,
                                                                 val_window_days=val_window_days,
                                                                 id_column=split_id_column
                                                                )
            
        else:
            self.X_train = raw_data_train.drop(columns=['TX_FRAUD'])
            self.y_train = raw_data_train["TX_FRAUD"]
            self.X_val, self.y_val = None, None

        self.tune_threshold = tune_threshold

        self.best_score = 0.0
        self.best_records = dict()
        self.ckpt_filename = os.path.join(
            args.work_dir, args.study_name + "_best-run.joblib"
        )
        
        # assert isinstance(self.args.scoring, str), "It should be a string."

    def sample_cfg_optuna(self, trial, name: str, config: dict):
        
        cfg = dict()

        for k,v in config.items():
            if isinstance(v, Sequence):
                cfg[k] = trial.suggest_categorical(f"{name}__{k}", v)
            else:
                cfg[k] = v

        return cfg

    def save_checkpoint(self, model_name: str, score: float, results):
        if score > self.best_score:
            self.best_score = score
            joblib.dump(results, self.ckpt_filename)
            # self.best_records[model_name] = results

    def load_hyp_conf(self, path_conf: str):
        try:
            self.HYP_CONFIGS = import_from_path("hyp_search_conf", path_conf)
        except:
            self.HYP_CONFIGS = import_from_path("hyp_search_conf", path_conf)
    
    def _run(
        self,
        classifier,
        params_config: dict,
        X_train,
        y_train,
        save_path: str = None,
        verbose=0,
    ) -> BaseSearchCV:
        
        
        cv = TimeSeriesSplit(n_splits=self.args.n_splits, gap=self.args.cv_gap)

        search_engine = hyperparameter_tuning(
            cv=cv,
            params_config=params_config,
            X_train=X_train,
            y_train=y_train,
            model=classifier,
            scoring=self.args.scoring,
            method=self.args.cv_method,
            verbose=self.verbose,
            n_iter=self.args.cv_n_iter,
            n_jobs=self.args.n_jobs,
        )

        if self.verbose:
            score = search_engine.best_score_
            print(f"mean {self.args.scoring} score for best_estimator: {score:.4f}")

        # save results
        if save_path:
            joblib.dump(search_engine, save_path)

        return search_engine

    def __call__(self, trial):
        self.count_iter += 1

        self.transform_pipeline = None
        self.selector = None

        # select model
        model_name = trial.suggest_categorical(
            "classifier",
            self.model_names,
        )
        models_config = {model_name: self.HYP_CONFIGS.models[model_name]}
        model, models_config = get_model(model_name, self.HYP_CONFIGS.models)
        model = instantiate_model(
            model, **sample_model_cfg(models_config)
        )

        # Iterate on categorical encoding method
        if self.iterate_cat_method:
            cat_encoding_method = trial.suggest_categorical(
                    "cat_encoding_method", self.args.cat_encoding_methods 
                )
        else:
            #set handle categorical variables
            cat_encoding_method = self.args.cat_encoding_method

            if model_name == "histGradientBoosting":
                model.set_params(categorical_features="from_dtype")
            elif model_name == "catboost":
                cat_encoding_method = 'catboost'
            elif model_name == 'xgboost':
                model.set_params(enable_categorical=True)

            if str(self.args.cat_encoding_method) == 'None':
                if model_name not in ['catboost','xgboost','histGradientBoosting','lgbm']:
                    # raise ValueError("The provided model does not support un-encoded categorical variables")
                    cat_encoding_method = 'woe'
                    print(f"The provided model does not support un-encoded categorical variables. Using {cat_encoding_method} categorical variable encoder.")
        
        # PCA
        # do_pca = trial.suggest_categorical("pca", [False, self.args.do_pca])
        do_pca = self.args.do_pca
        pca_n_components = None
        if do_pca:
            pca_n_components = trial.suggest_int("n_components", 35, 50, 5)

        # select outlier detector for data aug
        disable_pyod = trial.suggest_categorical(
            "disable_pyod", [True, self.disable_pyod]
        )
        if disable_pyod:
            # outliers_det_configs = None
            detector_list = None
        else:
            # cfgs = list()
            pyod_choices = trial.suggest_categorical(
                "pyod_choices",
                self.pyod_choices,  # range(1,self.pyod_detectors+1)
            )
            pyod_choices = json.loads(pyod_choices)
            detector_list = []
            for det_name in self.pyod_detectors:
                cfg = self.sample_cfg_optuna(
                    trial, det_name, self.HYP_CONFIGS.outliers_detectors[det_name]
                )
                detector, cfg = get_detector(name=det_name, config={det_name: cfg})
                detector = instantiate_detector(detector, cfg)
                detector_list.append(detector)


        # select outlier detector for data aug
        disable_samplers = trial.suggest_categorical(
            "disable_sampling", [True, self.args.disable_samplers]
        )
        resampler = None
        if not disable_samplers:
            # cfgs = list()
            sampler = trial.suggest_categorical(
                "samplers",self.HYP_CONFIGS.undersamplers + self.HYP_CONFIGS.oversamplers
                # self.HYP_CONFIGS.oversamplers + self.HYP_CONFIGS.combinedsamplers, 
            )
            cfg = self.sample_cfg_optuna(
                trial, sampler, self.HYP_CONFIGS.samplers[sampler]
            )
            resampler, cfg = get_sampler(name=sampler, config={sampler: cfg})
            resampler = instantiate_sampler(resampler, cfg)


        # feature selector:
        # do_feature_selection = self.args.do_feature_selection
        do_feature_selection = trial.suggest_categorical(
            "select_features", [False, self.args.do_feature_selection]
        )
        selector_cfg = {}
        feature_selector_name = None
        k_score_func = None
        
        if do_feature_selection:
            feature_selector_name = trial.suggest_categorical(
                "feature_selector_name", self.HYP_CONFIGS.feature_selector.keys()
            )
            selector_cfg = self.sample_cfg_optuna(
                trial,
                feature_selector_name,
                self.HYP_CONFIGS.feature_selector[feature_selector_name],
            )

            if feature_selector_name == 'selectkbest':
                k_score_func = trial.suggest_categorical('selectkbest_score_func', self.HYP_CONFIGS.selectkbest_score_func.keys())
                k_score_func = self.HYP_CONFIGS.selectkbest_score_func[k_score_func]
                    
        # advanced features cfg
        advanced_transformation = dict(add_fft=self.args.add_fft,
                add_seasonal_features=self.args.add_seasonal_features,
                use_nystrom=self.args.use_nystrom,
                use_sincos=self.args.use_sincos,
                use_spline=self.args.use_spline,
                add_fraud_rate_features=self.args.add_fraud_rate_features,
                add_cum_features = self.args.add_cum_features,
        )
        # for k,v in advanced_transformation.items():
        #     advanced_transformation[k] = trial.suggest_categorical(
        #                                 k, [False, v],
        #                             )
        if advanced_transformation['use_nystrom']:
            advanced_transformation['nystroem_components'] = trial.suggest_int("nystroem_components", 35, 72, 5)
        
        if do_pca:
            advanced_transformation['pca_n_components']= pca_n_components or 20

        # feature selector cfg
        if k_score_func is not None:
            advanced_transformation['k_score_func'] = k_score_func
        advanced_transformation['top_k_best']= selector_cfg.get("k",50)
        advanced_transformation['corr_method']=selector_cfg.get('method',"pearson")
        advanced_transformation['corr_threshold']=selector_cfg.get('threshold',0.82)
        advanced_transformation['scoring']=selector_cfg.get("scoring", "f1")
        advanced_transformation['rfe_step']=selector_cfg.get("step", 3) 
        advanced_transformation['seq_n_features_to_select']= selector_cfg.get("n_features_to_select", 3) 
        advanced_transformation['feature_select_estimator']= DecisionTreeClassifier(max_depth=7, class_weight='balanced',
                                                                                    max_features=None, random_state=41)        
        
        # session gap length
        session_gap_minutes = self.args.session_gap_minutes
        if self.iterate_session_gap:
            session_gap_minutes = trial.suggest_categorical(
                "session_gap_minutes", self.args.session_gap_minutes
            )
        else:
            assert isinstance(self.args.session_gap_minutes, int), "session_gap_minutes should be an integer."
            session_gap_minutes = self.args.session_gap_minutes
        
        # categorical-numerical interactions
        do_poly_interact = trial.suggest_categorical("do_poly_interact", [False, self.args.add_poly_interactions])
        if do_poly_interact:
            poly_degree = trial.suggest_categorical('poly_degree_interact',[1,self.args.poly_degree])
            if self.args.iterate_poly_cat_encoder_name:
                poly_cat_encoder_name = trial.suggest_categorical('poly_cat_encoder_name',
                                                                  self.args.poly_iterate_cat_encoders)
            else:
                poly_cat_encoder_name = self.args.poly_cat_encoder_name
            data_interaction_args = dict(
                                        cat_similarity_encode=self.args.cat_similarity_encode,
                                        nlp_model_name=self.args.nlp_model_name,
                                        add_poly_interactions=self.args.add_poly_interactions,
                                        interaction_cat_cols=self.args.interaction_cat_cols,
                                        poly_degree=poly_degree or self.args.poly_degree,
                                        poly_cat_encoder_name=poly_cat_encoder_name
                            )
            advanced_transformation.update(data_interaction_args)
        
        # number of clusters
        n_clusters = self.args.n_clusters
        if self.args.n_clusters>0:
            range_ = list(range(0,n_clusters+1))
            range_.remove(1)
            n_clusters = trial.suggest_categorical('n_clusters', range_)
                                                

        #-- load workflow
        classifier = load_workflow(
            classifier=model,
            cols_to_drop=self.args.cols_to_drop,
            detector_list=detector_list,
            cv_gap=self.args.cv_gap,
            reorder_by=self.args.reorder_by,
            n_splits=self.args.n_splits,
            behavioral_drift_cols=list(self.args.behavioral_drift_cols),
            add_imputer=self.args.add_imputer,
            session_gap_minutes=session_gap_minutes,
            uid_cols=self.args.uid_cols,
            feature_selector_name=feature_selector_name,
            windows_size_in_days=self.args.windows_size_in_days,
            cat_encoding_method=cat_encoding_method,
            cat_encoding_kwargs=self.cat_encoding_kwards,
            imputer_n_neighbors=self.args.imputer_n_neighbors,
            n_clusters=n_clusters,
            do_pca=do_pca,
            verbose=self.verbose,
            n_jobs=self.args.n_jobs,
            **advanced_transformation
        )        

        # get workflow parameters
        params_config = {
            f"model__{k}": v
            for k, v in models_config.items()
            if isinstance(v, Sequence)
        }

        sampled_cfg = dict()
        for k, v in params_config.items():
            if isinstance(v, Sequence):
                sampled_cfg[k] = trial.suggest_categorical(f"{model_name}_{k}",v)
            else:
                sampled_cfg[k] = k
        # update params
        classifier.set_params(**sampled_cfg)

        # tuned threshold version
        if trial.suggest_categorical('tune_threshold',[False,self.tune_threshold]):
            classifier = TunedThresholdClassifierCV(classifier,
                                               scoring='f1',
                                               cv=TimeSeriesSplit(n_splits=3,gap=1000),
                                            )

        X = self.X_train.copy()
        y = self.y_train.copy()

        if self.args.do_train_val_split:
            X_val = self.X_val.copy()
            y_val = self.y_val.copy()
            

            if model_name == "catboost":

                # assert not do_feature_selection, "Disable feature selection"

                # classifier.fit(X,y)

                preprocessor = classifier[:-1]
                model = classifier[-1]

                X = preprocessor.fit_transform(X,y)
                train_pool = Pool(X,y,
                                  timestamp=self.X_train['TX_DATETIME'].diff().dt.total_seconds().fillna(0).astype(float)/60,
                                #   weight=y*1e3,
                                  )
                
                X_val = preprocessor.transform(X_val)
                val_pool = Pool(X_val,
                                y_val,
                                timestamp=self.X_val['TX_DATETIME'].diff().dt.total_seconds().fillna(0).astype(float)/60
                            )

                model.fit(train_pool,
                            use_best_model=True,
                            eval_set=val_pool
                        )

                # summary = model.select_features(
                #                     train_pool,
                #                     eval_set=val_pool,
                #                     features_for_select=list(range(X.shape[1])),
                #                     num_features_to_select=50,
                #                     steps=3,
                #                     algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
                #                     shap_calc_type=EShapCalcType.Regular,
                #                     train_final_model=True,
                #                     logging_level='Silent',
                #                     plot=False
                #                 )

                # selected_features_indices = summary['selected_features']

                ## get score
                scores = list()
                for i in range(max(model.tree_count_-1, 1)):
                    y_pred_val = model.predict( val_pool,
                                                prediction_type='Class',
                                                ntree_start=i,
                                                ntree_end=model.get_best_iteration()
                                            )
                    score = f1_score(y_true=y_val,
                                y_pred=y_pred_val
                                )
                    scores.append(score)
                fitness = scores[np.argmax(scores)]   

                self.save_checkpoint(model_name=model_name, score=fitness, results=classifier)
                return fitness

            else:
                classifier.fit(X,y)

            if isinstance(self.args.scoring,Sequence):
                scores = [get_scorer(scoring=metric)(classifier,X_val,y_val) for metric in self.args.scoring]
            else:
                scores = get_scorer(scoring=self.args.scoring)(classifier,X_val,y_val)

            results = {'estimator':classifier}

        else:
            # get results
            results = cross_validate(estimator=classifier,
                            X=X,
                            y=y,
                            return_estimator=True,
                            cv=TimeSeriesSplit(n_splits=self.args.n_splits,
                                            gap=self.args.cv_gap),
                            scoring=self.args.scoring, #evaluate
                            error_score='raise',
                            n_jobs=self.args.n_jobs,
                            pre_dispatch=self.args.n_jobs,
                        )
                
            scores = []
            if isinstance(self.args.scoring,Sequence):
                scores = [np.mean(results[f"test_{metric}"]) for metric in self.args.scoring]
            else:
                scores = np.mean(results["test_score"])
        

        #- save checkpoint
        estimator = results['estimator']
        fitness = np.mean(scores)
        self.save_checkpoint(model_name=model_name, score=fitness, results=estimator)

        return fitness

