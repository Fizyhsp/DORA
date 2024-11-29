"""项目设置."""
from datetime import timedelta, timezone
from os import environ


class Config:
    """项目设置."""

    EXPIRES = 60 * 60 * 24 * 7  # 日志过期时间
    TIMEZONE = timezone(timedelta(hours=8), "CST")  # 时区
    VERBOSE = True  # 是否显示详细日志
    VERBOSE_LEVEL = 15  # 详细日志等级

    USE_MONGO = False  # 是否使用mongo
    USE_REDIS = False  # 是否使用redis
    USE_SQL = False  # 是否使用sql类数据库

    USE_MINIWOB = False  # 是否使用miniwob
    USE_WEB2MIND = False  # 是否使用web2mind
    USE_WEBARENA = False  # 是否使用webarena

    SQL_DATABASE_NAME = "lawen"  # sql数据库名称
    MONGO_DATABASE_NAME = "lawen"  # mongo数据库名称
    REDIS_CHANNEL_NAME = "lawen"  # redis频道名称

    MONGO_URI = environ.get("MONGO_URI", "mongodb://localhost:27017")  # mongo地址
    REDIS_URL = environ.get("REDIS_URL", "redis://localhost:6379/0")  # redis地址
    SQLALCHEMY_DATABASE_URI = environ.get(
        "SQLALCHEMY_DATABASE_URI",
        "sqlite:///lawen.db",  # sql数据库地址
    )
