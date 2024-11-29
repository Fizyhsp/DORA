"""基础环境."""
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self

    from utils import ARGS, KWARGS


@dataclass
class Action:
    """action基类."""

    name: str
    description: str
    _: KW_ONLY
    func: Callable | None = None
    input_args_schema: dict = field(default_factory=dict)
    enable: bool = True
    tags: list[str] | None = None
    metadata: list[dict] | None = None

    def valid_args(self: "Self", input_action_args: dict) -> bool:
        """验证参数是否合法.

        Args:
            input_action_args (dict): 输入参数

        Returns:
            bool: 是否合法
        """
        return not any(
            key not in input_action_args
            or not isinstance(
                input_action_args[key],
                value,
            )
            for key, value in self.input_args_schema.items()
        )

    def __call__(
        self: "Self",
        **input_action_args: "KWARGS",
    ) -> str:
        """执行Action.

        Args:
            input_action_args (dict): 输入参数

        Returns:
            str: 执行结果

        """
        if not self.enable:
            msg = "Action不可用"
            raise ValueError(msg)
        if self.func is None:
            msg = "Action未实现"
            raise ValueError(msg)
        if not self.valid_args(input_action_args=input_action_args):
            msg = "参数不合法"
            raise ValueError(msg)
        return self.func(**input_action_args)


@dataclass
class ActionSpace:
    """Action空间."""

    actions: list[Action] = field(init=False)
    action_names: list[str] = field(init=False)

    def get(self: "Self", action_name: str) -> Action:
        """获取Action.

        Args:
            action_name (str): Action名称

        Returns:
            Action: Action实例
        """
        if action_name not in self.action_names:
            msg: str = f"Action {action_name} 不存在"
            raise ValueError(msg)
        return getattr(self, action_name)


@dataclass
class WebEnv:
    """Web环境."""

    name: str
    env_type: str
    _: KW_ONLY
    env_task: str | None = None
    action_space: ActionSpace = field(init=False)

    @abstractmethod
    def reset(self: "Self", *args: "ARGS", **kwargs: "KWARGS") -> None:
        """重置."""
        raise NotImplementedError

    @abstractmethod
    def step(self: "Self", *args: "ARGS", **kwargs: "KWARGS") -> None:
        """执行Action."""
        raise NotImplementedError
