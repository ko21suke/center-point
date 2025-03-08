import yaml
from easydict import EasyDict

def merge(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge(config[key], val)


def cfg_from_yaml(path):
    with open(path, 'r') as f:
        try:
            return yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            return yaml.safe_load(f)


def load_config(path):
    cfg = EasyDict()
    merge(cfg, cfg_from_yaml(path))
    return cfg
