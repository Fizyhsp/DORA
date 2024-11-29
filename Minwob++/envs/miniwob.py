"""miniwob环境."""
from dataclasses import dataclass, field
from functools import cached_property, singledispatchmethod
from typing import TYPE_CHECKING, cast

import numpy as np
from gymnasium import make
from marshmallow.fields import Raw
from miniwob.action import ActionTypes
from miniwob.environment import MiniWoBEnvironment

from utils import make_instance

from .basic import Action, ActionSpace, WebEnv

if TYPE_CHECKING:
    from typing_extensions import Self

    from utils import KWARGS


@dataclass
class MiniwobActionSpace(ActionSpace):
    """miniwob Action空间."""

    @cached_property
    def none(self: "Self") -> Action:
        """无操作."""

        def func(env: MiniWoBEnvironment) -> dict:
            """无操作."""
            return cast(MiniWoBEnvironment, env.unwrapped).create_action(
                action_type=ActionTypes.NONE,
            )

        return make_instance(
            cls=Action,
            name="none",
            func=func,
            description="No-op",
            input_args_schema={},
        )

    @cached_property
    def move_coords(self: "Self") -> Action:
        """移动鼠标."""

        def func(env: MiniWoBEnvironment, left: float, top: float) -> dict:
            """移动鼠标.

            Args:
                env (MiniWoBEnvironment): 环境
                left (float): x坐标
                top (float): y坐标

            Returns:
                dict: MiniWoB动作
            """
            return cast(MiniWoBEnvironment, env.unwrapped).create_action(
                action_type=ActionTypes.MOVE_COORDS,
                coords=np.array([left, top]),
            )

        return make_instance(
            cls=Action,
            name="move_coords",
            func=func,
            description="Move the mouse to the specified coordinates",
            input_args_schema={
                "left": float,
                "top": float,
            },
        )

    @cached_property
    def click_coords(self: "Self") -> Action:
        """点击鼠标."""

        def func(env: MiniWoBEnvironment, left: float, top: float) -> dict:
            """点击鼠标.

            Args:
                env (MiniWoBEnvironment): 环境
                left (float): x坐标
                top (float): y坐标

            Returns:
                dict: MiniWoB动作
            """
            return cast(MiniWoBEnvironment, env.unwrapped).create_action(
                action_type=ActionTypes.CLICK_COORDS,
                coords=np.array([left, top]),
            )

        return make_instance(
            cls=Action,
            name="click_coords",
            func=func,
            description="Click the mouse at the specified coordinates",
            input_args_schema={
                "left": float,
                "top": float,
            },
        )

    @cached_property
    def dbclick_coords(self: "Self") -> Action:
        """双击鼠标."""

        def func(env: MiniWoBEnvironment, left: float, top: float) -> dict:
            """双击鼠标.

            Args:
                env (MiniWoBEnvironment): 环境
                left (float): x坐标
                top (float): y坐标

            Returns:
                dict: MiniWoB动作
            """
            return cast(MiniWoBEnvironment, env.unwrapped).create_action(
                action_type=ActionTypes.DBLCLICK_COORDS,
                coords=np.array([left, top]),
            )

        return make_instance(
            cls=Action,
            name="dbclick_coords",
            func=func,
            description="Double-click the mouse at the specified coordinates",
            input_args_schema={
                "left": float,
                "top": float,
            },
        )

    @cached_property
    def mousedown_coords(self: "Self") -> Action:
        """按下鼠标."""

        def func(env: MiniWoBEnvironment, left: float, top: float) -> dict:
            """按下鼠标.

            Args:
                env (MiniWoBEnvironment): 环境
                left (float): x坐标
                top (float): y坐标

            Returns:
                dict: MiniWoB动作
            """
            return cast(MiniWoBEnvironment, env.unwrapped).create_action(
                action_type=ActionTypes.MOUSEDOWN_COORDS,
                coords=np.array([left, top]),
            )

        return make_instance(
            cls=Action,
            name="mousedown_coords",
            func=func,
            description="Click and hold the mouse at the specified coordinates",
            input_args_schema={
                "left": float,
                "top": float,
            },
        )

    @cached_property
    def mouseup_coords(self: "Self") -> Action:
        """释放鼠标."""

        def func(env: MiniWoBEnvironment, left: float, top: float) -> dict:
            """释放鼠标.

            Args:
                env (MiniWoBEnvironment): 环境
                left (float): x坐标
                top (float): y坐标

            Returns:
                dict: MiniWoB动作
            """
            return cast(MiniWoBEnvironment, env.unwrapped).create_action(
                action_type=ActionTypes.MOUSEUP_COORDS,
                coords=np.array([left, top]),
            )

        return make_instance(
            cls=Action,
            name="mouseup_coords",
            func=func,
            description="Release the mouse at the specified coordinates",
            input_args_schema={
                "left": float,
                "top": float,
            },
        )

    @cached_property
    def scroll_up_coords(self: "Self") -> Action:
        """向上滚动."""

        def func(
                env: MiniWoBEnvironment,
                left: float,
                top: float,
        ) -> dict:
            """向上滚动.

            Args:
                env (MiniWoBEnvironment): 环境
                left (float): x坐标
                top (float): y坐标

            Returns:
                dict: MiniWoB动作
            """
            return cast(MiniWoBEnvironment, env.unwrapped).create_action(
                action_type=ActionTypes.SCROLL_UP_COORDS,
                coords=np.array([left, top]),
            )

        return make_instance(
            cls=Action,
            name="scroll_up_coords",
            func=func,
            description="Scroll up at the specified coordinates",
            input_args_schema={
                "left": float,
                "top": float,
            },
        )

    @cached_property
    def scroll_down_coords(self: "Self") -> Action:
        """向下滚动."""

        def func(
                env: MiniWoBEnvironment,
                left: float,
                top: float,
        ) -> dict:
            """向下滚动.

            Args:
                env (MiniWoBEnvironment): 环境
                left (float): x坐标
                top (float): y坐标

            Returns:
                dict: MiniWoB动作
            """
            return cast(MiniWoBEnvironment, env.unwrapped).create_action(
                action_type=ActionTypes.SCROLL_DOWN_COORDS,
                coords=np.array([left, top]),
            )

        return make_instance(
            cls=Action,
            name="scroll_down_coords",
            func=func,
            description="Scroll down at the specified coordinates",
            input_args_schema={
                "left": float,
                "top": float,
            },
        )

    @cached_property
    def click_element(self: "Self") -> Action:
        """点击."""

        def func(
                env: MiniWoBEnvironment,
                ref: int,
        ) -> dict:
            """点击.

            Args:
                env (MiniWoBEnvironment): 环境
                ref (int): 元素索引

            Returns:
                dict: MiniWoB动作
            """
            return cast(MiniWoBEnvironment, env.unwrapped).create_action(
                action_type=ActionTypes.CLICK_ELEMENT,
                ref=ref,
            )

        return make_instance(
            cls=Action,
            name="click_element",
            func=func,
            description="Click the element in web page",
            input_args_schema={"ref": int},
        )

    @cached_property
    def press_key(self: "Self") -> Action:
        """按键."""

        def func(
                env: MiniWoBEnvironment,
                key: int,
        ) -> dict:
            """按键.

            Args:
                env (MiniWoBEnvironment): 环境
                key (int): 键名

            Returns:
                dict: MiniWoB动作
            """
            return cast(MiniWoBEnvironment, env.unwrapped).create_action(
                action_type=ActionTypes.PRESS_KEY,
                key=key,
            )

        return make_instance(
            cls=Action,
            name="press_key",
            func=func,
            description="Press the key",
            input_args_schema={"key": int},
        )

    @cached_property
    def type_text(self: "Self") -> Action:
        """输入文本."""

        def func(
                env: MiniWoBEnvironment,
                text: str,
        ) -> dict:
            """输入文本.

            Args:
                env (MiniWoBEnvironment): 环境
                text (str): 文本

            Returns:
                dict: MiniWoB动作
            """
            return cast(MiniWoBEnvironment, env.unwrapped).create_action(
                action_type=ActionTypes.TYPE_TEXT,
                text=text,
            )

        return make_instance(
            cls=Action,
            name="type_text",
            func=func,
            description="Type the text",
            input_args_schema={"text": str},
        )

    @cached_property
    def type_field(self: "Self") -> Action:
        """输入指定任务字段的值."""

        def func(
                env: MiniWoBEnvironment,
                field: int,
        ) -> dict:
            """输入指定任务字段的值.

            Args:
                env (MiniWoBEnvironment): 环境
                field (int): 字段索引

            Returns:
                dict: MiniWoB动作
            """
            return cast(MiniWoBEnvironment, env.unwrapped).create_action(
                action_type=ActionTypes.TYPE_FIELD,
                field=field,
            )

        return make_instance(
            cls=Action,
            name="type_field",
            func=func,
            description="Type the text",
            input_args_schema={"field": int},
        )

    @cached_property
    def focus_element_and_type_text(self: "Self") -> Action:
        """使用 JavaScript 单击指定元素,然后键入文本."""

        def func(
                env: MiniWoBEnvironment,
                ref: int,
                text: str,
        ) -> dict:
            """使用 JavaScript 单击指定元素,然后键入文本.

            Args:
                env (MiniWoBEnvironment): 环境
                ref (int): 元素索引
                text (str): 文本

            Returns:
                dict: MiniWoB动作
            """
            return cast(MiniWoBEnvironment, env.unwrapped).create_action(
                action_type=ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT,
                ref=ref,
                text=text,
            )

        return make_instance(
            cls=Action,
            name="focus_element_and_type_text",
            func=func,
            description="Type the text",
            input_args_schema={
                "ref": int,
                "text": str,
            },
        )

    @cached_property
    def focus_element_and_type_field(self: "Self") -> Action:
        """使用 JavaScript 单击指定元素,然后键入指定任务字段的值."""

        def func(
                env: MiniWoBEnvironment,
                ref: int,
                field: int,
        ) -> dict:
            """使用 JavaScript 单击指定元素,然后键入指定任务字段的值.

            Args:
                env (MiniWoBEnvironment): 环境
                ref (int): 元素索引
                field (int): 字段索引

            Returns:
                dict: MiniWoB动作
            """
            return cast(MiniWoBEnvironment, env.unwrapped).create_action(
                action_type=ActionTypes.FOCUS_ELEMENT_AND_TYPE_FIELD,
                ref=ref,
                field=field,
            )

        return make_instance(
            cls=Action,
            name="focus_element_and_type_field",
            func=func,
            description="Type the text",
            input_args_schema={
                "ref": int,
                "field": int,
            },
        )

    @cached_property
    def actions(self: "Self") -> list[Action]:
        """actions."""
        return [
            self.none,
            self.move_coords,
            self.click_coords,
            self.dbclick_coords,
            self.mousedown_coords,
            self.mouseup_coords,
            self.scroll_up_coords,
            self.scroll_down_coords,
            self.click_element,
            self.press_key,
            self.type_text,
            self.type_field,
            self.focus_element_and_type_text,
            self.focus_element_and_type_field,
        ]

    @cached_property
    def action_names(self: "Self") -> list[str]:
        """Action names."""
        return [action.name for action in self.actions]


@dataclass
class MiniwobEnv(WebEnv):
    """miniwob环境."""

    render_mode: str = None
    EPISODE_MAX_TIME: int | None = None
    situations: MiniWoBEnvironment = field(
        init=False,
        metadata={"marshmallow_field": Raw()},
    )
    action_space: MiniwobActionSpace = field(
        default_factory=MiniwobActionSpace,
        init=False,
    )

    def __post_init__(self: "Self", **kwargs: "KWARGS") -> None:
        """初始化."""
        self.situations = make(id=self.name, render_mode=self.render_mode)
        if self.EPISODE_MAX_TIME is not None:
            self.situations.instance.driver.execute_script(
                f"core.EPISODE_MAX_TIME={self.EPISODE_MAX_TIME};"
            )
        if kwargs is not None:
            (setattr(self, key, value) for key, value in kwargs.items())

    def reset(self: "Self", seed: int = 0) -> tuple[dict, dict]:
        """重置.

        Returns:
            observation (dict): An element of the environment's observation_space.
            info (dict) : Infos.
        """
        return self.situations.reset(seed=seed)

    @singledispatchmethod
    def step(
            self: "Self",
            action: "str|Action",
            **kwargs: "KWARGS",
    ) -> tuple[dict, float, bool, bool, dict]:
        """执行动作.

        Args:
            action (str|Action): 动作/动作名称
            kwargs (dict): 动作参数

        Returns:
            observation (dict): An element of the environment's observation_space.
            reward (float) : Amount of reward returned after previous action.
            terminated (bool): Whether the episode has ended.
            truncated (bool): Whether the step limit has been reached.
            info (dict) : Infos.
        """

    @step.register(str)
    def _(
            self: "Self",
            action: str,
            **kwargs: "KWARGS",
    ) -> tuple[dict, float, bool, bool, dict]:
        """执行动作.

        Args:
            action (str): 动作名称
            kwargs (dict): 动作参数

        Returns:
            observation (dict): An element of the environment's observation_space.
            reward (float) : Amount of reward returned after previous action.
            terminated (bool): Whether the episode has ended.
            truncated (bool): Whether the step limit has been reached.
            info (dict) : Infos.
        """
        return self.situations.step(
            action=self.action_space.get(action_name=action)(
                env=self.situations,
                **kwargs,
            ),
        )

    @step.register(Action)
    def _(
            self: "Self",
            action: Action,
            **kwargs: "KWARGS",
    ) -> tuple[dict, float, bool, bool, dict]:
        """执行动作.

        Args:
            action (Action): 动作实例
            kwargs (dict): 动作参数

        Returns:
            observation (dict): An element of the environment's observation_space.
            reward (float) : Amount of reward returned after previous action.
            terminated (bool): Whether the episode has ended.
            truncated (bool): Whether the step limit has been reached.
            info (dict) : Infos.
        """
        return self.situations.step(
            action=action(
                env=self.situations,
                **kwargs,
            ),
        )

    def close(self: "Self") -> None:
        """关闭."""
        self.situations.close()
