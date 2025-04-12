from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.model_selection._search import BaseSearchCV
from hpsklearn import HyperoptEstimator
from hyperopt import tpe
import json
import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer, f1_score
import random
from collections.abc import Iterable
import optuna
from optuna.samplers import TPESampler


def evaluate(classifier, X, y):
    metrics = ["accuracy", "f1", "average_precision", "precision", "recall"]
    for metric in metrics:
        scorer = get_scorer(metric)
        score = scorer(classifier, X, y)
        print(f"{metric}: {score:.4f}")


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
        if isinstance(value, Iterable):
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
            random_state=41,
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
            if isinstance(v, Iterable):
                param_dist[k] = optuna.distributions.CategoricalDistribution(v)
            else:
                param_dist[k] = optuna.distributions.CategoricalDistribution(
                    [
                        v,
                    ]
                )

        search_engine = optuna.integration.OptunaSearchCV(
            model,
            param_distributions=param_dist,
            cv=cv,
            refit=True,
            n_jobs=n_jobs,
            study=study,
            scoring=scoring,
            # error_score='raise',
            max_iter=10000,
            timeout=60 * 3,
            n_trials=n_iter,
            random_state=41,
            verbose=verbose,
        )

    elif method == "hyperopt":
        loss_fn = lambda y_true, y_pred: 1.0 - f1_score(
            y_true=y_true,
            y_pred=y_pred,
            pos_label=1,
            zero_division=1,
        )
        search_engine = HyperoptEstimator(
            classifier=model,
            algo=tpe.suggest,
            max_evals=n_iter,
            loss_fn=loss_fn,
            n_jobs=n_jobs,
            trial_timeout=60 * 2,
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
