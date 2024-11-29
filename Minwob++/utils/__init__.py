"""工具."""
from .config import Config
from .logger import setup_logger
from .schema import ARGS, KWARGS, dump_instance, json, make_instance

setup_logger()
__all__ = [
    "ARGS",
    "Config",
    "KWARGS",
    "dump_instance",
    "json",
    "make_instance",
]
