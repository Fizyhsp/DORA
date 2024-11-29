"""Prompt."""
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, Template, meta
from langchain.prompts import StringPromptTemplate

if TYPE_CHECKING:
    from typing_extensions import Self

    from utils import KWARGS


class AgentPrompt(StringPromptTemplate):
    """智能体Prompt."""

    template_format: str = "jinja2"
    template: Template

    def __init__(self: "Self", **kwargs: "KWARGS") -> None:
        """智能体Prompt."""
        super().__init__(**kwargs)

    @classmethod
    def from_template(
        cls: type["Self"],
        template_str: str,
        **kwargs: "KWARGS",
    ) -> "Self":
        """从模板字符串中创建Prompt.

        Args:
            template_str (str): 模板字符串
            kwargs: 其他参数
        """
        template = Template(source=template_str)
        env = Environment(autoescape=True)
        ast: Template = env.parse(source=template_str)
        input_variables: set[str] = meta.find_undeclared_variables(ast=ast)
        return cls(
            input_variables=sorted(input_variables),
            template=template,
            **kwargs,
        )

    @classmethod
    def from_file(
        cls: type["Self"],
        template_path: str,
        **kwargs: "KWARGS",
    ) -> "Self":
        """从模板文件中创建Prompt.

        Args:
            template_path (str): 模板文件路径
            kwargs: 其他参数
        """
        if Path(template_path).exists():
            with Path(template_path).open(mode="r", encoding="utf-8") as f:
                template_str: str = f.read()
            return cls.from_template(template_str=template_str, **kwargs)
        msg: str = f"{template_path} not found"
        raise FileNotFoundError(msg)

    def format(self: "Self", **kwargs: "KWARGS") -> str:  # noqa: A003
        """格式化Prompt."""
        return self.template.render(**kwargs)
