"""基础智能体."""
from abc import abstractmethod
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

from envs import WebEnv

if TYPE_CHECKING:
    from typing_extensions import Self

    from envs import Action
    from utils import ARGS, KWARGS


@dataclass
class BasicAgent:
    """智能体."""

    name: str
    env: WebEnv
    _: KW_ONLY
    task: str | None = None
    actions: list[dict] | None = None

    def __post_init__(self: "Self") -> None:
        """初始化."""
        if self.env.env_task is not None:
            self.task = self.env.env_task
        if self.task is None:
            msg = "Agent / Env need a task"
            raise ValueError(msg)

    @abstractmethod
    def next_action(
        self: "Self",
        *args: "ARGS",
        **kwargs: "KWARGS",
    ) -> "Action":
        """Predict the next Action given the observation."""
        raise NotImplementedError

    @abstractmethod
    def plan(self: "Self", *args: "ARGS", **kwargs: "KWARGS") -> list["Action"]:
        """Predict the actions plan given the observation."""
        raise NotImplementedError

    @abstractmethod
    def reset(
        self: "Self",
        *args: "ARGS",
        **kwargs: "KWARGS",
    ) -> None:
        """Predict the actions plan given the observation."""
        raise NotImplementedError
