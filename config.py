import tomllib
from pathlib import Path

CONFIG = "basic.toml"

def _current_config_dir():
    return Path(__file__).resolve().parent / "configs" / CONFIG

def _get_default_params(section):
    with open(_current_config_dir(), "rb") as f:
        config = tomllib.load(f)
        return config[section]

def get_compress_params():
    return _get_default_params("compression")

def get_pagerank_params():
    return _get_default_params("pagerank")

def change_config(new):
    global CONFIG
    CONFIG  = new
