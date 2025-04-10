# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 12:51:16 2025

@author: FADELCO
"""

# import numpy as np
from pyod.models.base import BaseDetector
import random


def get_detector(name: str, config: dict) -> tuple[BaseDetector, dict]:
    """
    Get the model configuration based on the model name.
    """
    if name not in config:
        raise ValueError(f"Detector {name} not found in config.")

    cfg = config[name].copy()

    # Remove the model from the config dictionary
    detector = cfg.pop("detector")

    return detector, cfg


def sample_cfg(cfg: dict) -> dict:
    """
    Sample a configuration from the given configuration dictionary.
    """
    sampled_cfg = {}
    for key, value in cfg.items():
        if isinstance(value, list):
            sampled_cfg[key] = random.choice(value)
        else:
            sampled_cfg[key] = value
    return sampled_cfg


def instantiate_detector(detector: BaseDetector, kwargs: dict):
    """
    Instantiate the detector with the given parameters.
    """
    return detector(**kwargs)
