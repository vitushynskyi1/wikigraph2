import tomllib
from pathlib import Path

CONFIG = "basic.toml"

def _current_config_dir():
    return Path(__file__).resolve().parent / "configs" / CONFIG

def get_config():
    with open(_current_config_dir(), "rb") as f:
        config = tomllib.load(f)
        return config

def get_params(*request): 
    config = get_config()
    return (config[key] for key in request)


def change_config(new):
    global CONFIG
    CONFIG  = new
