def import_from_path(module_name: str, file_path: str):
    """https://docs.python.org/3/library/importlib.html#importing-programmatically


    Parameters
    ----------
    module_name : str
        name to be given to python module.
    file_path : str
        path to .py file.

    Returns
    -------
    module : python module
        loaded python module.

    """
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def sample_cfg(model_cfg: dict) -> dict:
    """
    Sample a configuration from the given configuration dictionary.
    """
    from collections.abc import Iterable
    import random

    sampled_cfg = {}
    for key, value in model_cfg.items():
        if isinstance(value, Iterable):
            sampled_cfg[key] = random.choice(value)
        else:
            sampled_cfg[key] = value
    return sampled_cfg
