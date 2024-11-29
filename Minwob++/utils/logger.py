"""日志."""
from datetime import datetime
from logging import (
    INFO,
    WARNING,
    FileHandler,
    Handler,
    LogRecord,
    StreamHandler,
    addLevelName,
    basicConfig,
)
from typing import TYPE_CHECKING

from colorlog import ColoredFormatter
from sqlalchemy import DateTime, String, Text, create_engine, text
from sqlalchemy.exc import ArgumentError, OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from .config import Config

if TYPE_CHECKING:
    if Config.USE_MONGO:
        from pymongo.collection import Collection
        from pymongo.database import Database
    if Config.USE_REDIS:
        from redis import Redis
    from typing_extensions import Self


class RedisHandler(Handler):
    """RedisHandler."""

    def __init__(self: "Self") -> None:
        """Init."""
        super().__init__()
        from redis import from_url

        self.db: "Redis" = from_url(Config.REDIS_URL)

    def emit(self: "Self", record: "LogRecord") -> None:
        """Emit."""
        self.db.publish(Config.REDIS_CHANNEL_NAME, self.format(record))


class MongoHandler(Handler):
    """MongoHandler."""

    def __init__(self: "Self") -> None:
        """Init."""
        super().__init__()
        from pymongo import ASCENDING, MongoClient

        has_created = True
        self.client: "MongoClient" = MongoClient(Config.MONGO_URI)
        if Config.MONGO_DATABASE_NAME not in self.client.list_database_names():
            has_created = False
        self.db: "Database" = self.client.get_database(
            Config.MONGO_DATABASE_NAME,
        )
        if "logs" not in self.db.list_collection_names():
            has_created = False
        self.collection: "Collection" = self.db.get_collection("logs")
        if not has_created:
            self.collection.create_index(
                [("time", ASCENDING)],
                expireAfterSeconds=Config.EXPIRES,
            )

    def emit(self: "Self", record: "LogRecord") -> None:
        """Emit."""
        self.collection.insert_one(
            {
                "time": datetime.now(tz=Config.TIMEZONE),
                "level": record.levelname,
                "message": self.format(record),
            },
        )


class Base(DeclarativeBase):
    """Base."""


class Logs(Base):
    """Log."""

    __tablename__ = "logs"

    log_id: Mapped[int] = mapped_column(primary_key=True)
    time: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now(tz=Config.TIMEZONE),
    )
    level: Mapped[str] = mapped_column(String(10))
    message: Mapped[str] = mapped_column(Text)

    def __repr__(self: "Self") -> str:
        """Repr."""
        return (
            f"Logs(log_id={self.log_id!r}, time={self.time.isoformat()!r}, "
            f"level={self.level!r}, message={self.message!r})"
        )


class SQLHandler(Handler):
    """SQLHandler."""

    def __init__(self: "Self") -> None:
        """Init."""
        super().__init__()
        self.engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)
        self.base_engine = create_engine(
            "/".join(Config.SQLALCHEMY_DATABASE_URI.split("/")[:-1]),
        )
        try:
            Base.metadata.create_all(self.engine)
        except OperationalError:
            if self.engine.dialect.name == "mysql":
                try:
                    with self.base_engine.connect() as connection:
                        connection.execute(
                            text(
                                "create database if not exists "
                                f"{Config.SQL_DATABASE_NAME};",
                            ),
                        )
                    Base.metadata.create_all(self.engine)
                except OperationalError as e:
                    msg = "数据库URI不正确"
                    raise ValueError(msg) from e
            elif self.engine.dialect.name == "postgresql":
                try:
                    with self.base_engine.execution_options(
                        isolation_level="AUTOCOMMIT",
                    ).connect() as connection:
                        connection.execute(
                            text(
                                f"create database {Config.SQL_DATABASE_NAME};",
                            ),
                        )
                    Base.metadata.create_all(self.engine)
                except OperationalError as e:
                    msg = "数据库URI不正确"
                    raise ValueError(msg) from e

        except ArgumentError as e:
            msg = "数据库URI不正确"
            raise ValueError(msg) from e

    def emit(self: "Self", record: "LogRecord") -> None:
        """Emit."""
        with Session(self.engine) as session:
            log = Logs(
                level=record.levelname,
                message=self.format(record),
            )
            session.add(log)
            session.commit()


def setup_logger() -> None:
    """创建日志器."""
    addLevelName(Config.VERBOSE_LEVEL, "VERBOSE")
    formatter = ColoredFormatter(
        (
            "%(asctime)s "
            "[%(log_color)s%(levelname)s%(reset)s] "
            "[%(cyan)s%(name)s%(reset)s] "
            "%(message_log_color)s%(message)s"
        ),
        reset=True,
        log_colors={
            "DEVELOPING": "bold_cyan",
            "INFO": "bold_green",
            "WARNING": "bold_yellow",
            "ERROR": "bold_red",
            "CRITICAL": "bold_red,bg_white",
        },
        secondary_log_colors={
            "message": {
                "DEVELOPING": "white",
                "INFO": "bold_white",
                "WARNING": "bold_yellow",
                "ERROR": "bold_red",
                "CRITICAL": "bold_red",
            },
        },
        style="%",
    )

    file_handler = FileHandler(
        f"logs/{datetime.now(tz=Config.TIMEZONE).strftime('%Y-%m-%d')}.log",
        encoding="utf-8",
    )
    file_handler.setLevel(Config.VERBOSE_LEVEL if Config.VERBOSE else INFO)

    stream_handler = StreamHandler()
    stream_handler.setLevel(INFO if Config.VERBOSE else WARNING)
    stream_handler.setFormatter(formatter)
    handlers = [file_handler, stream_handler]
    if Config.USE_REDIS:
        redis_handler = RedisHandler()
        redis_handler.setLevel(
            Config.VERBOSE_LEVEL if Config.VERBOSE else INFO,
        )
        handlers.append(redis_handler)
    if Config.USE_MONGO:
        mongo_handler = MongoHandler()
        mongo_handler.setLevel(
            Config.VERBOSE_LEVEL if Config.VERBOSE else INFO,
        )
        handlers.append(mongo_handler)
    if Config.USE_SQL:
        sql_handler = SQLHandler()
        sql_handler.setLevel(
            Config.VERBOSE_LEVEL if Config.VERBOSE else INFO,
        )
        handlers.append(sql_handler)
    basicConfig(
        level=Config.VERBOSE_LEVEL if Config.VERBOSE else INFO,
        format=("%(asctime)s [%(levelname)s] [%(name)s] %(message)s"),
        handlers=handlers,
    )
