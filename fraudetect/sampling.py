# import numpy as np
from imblearn.pipeline import Pipeline
import random

def get_sampler(name:str, config:dict) -> dict:
    """
    Get the configuration based on the name.
    """
    if name not in config:
        raise ValueError(f"Sampler {name} not found in config.")
    
    cfg = config[name].copy()
    
    # Remove the model from the config dictionary
    sampler = cfg.pop('sampler')
    
    return sampler, cfg

def sample_cfg(cfg:dict) -> dict:
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

def instantiate_sampler(sampler, kwargs:dict):
    """
    Instantiate the sampler with the given parameters.
    """
    return sampler(**kwargs)

def build_samplers_pipeline(sampler_list:list)->Pipeline:
    
    list_ = [(f'sampler{i}',sampler) for i,sampler in enumerate(sampler_list)]
    
    return Pipeline(list_)

