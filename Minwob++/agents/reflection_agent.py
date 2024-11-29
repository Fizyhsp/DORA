"""LLM智能体."""
from ast import literal_eval
from dataclasses import KW_ONLY, dataclass, field
from logging import log, warning
from typing import TYPE_CHECKING

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from envs import Action
from prompts import AgentPrompt
from utils import Config, dump_instance, json, make_instance

from .basic import BasicAgent

if TYPE_CHECKING:
    from typing_extensions import Self

    from utils import KWARGS




@dataclass
class ActionSchema:
    """ActionSchema."""

    name: str = field(default="action name")
    input_args_schema: dict = field(
        default_factory=lambda: {"arg name": "value"},
    )


@dataclass
class OutputFormat:
    """输出格式模式."""
    think: str = field(default_factory=str)
    action: ActionSchema = field(default_factory=ActionSchema)
    observation: dict = field(init=False)


@dataclass
class ReflectionAgent(BasicAgent):
    """LLM智能体."""

    api_base: str
    api_key: str
    model_name: str
    template: str
    trajectory: list[dict] = field(default_factory=list)
    observation: tuple = field(default_factory=tuple)
    reflection: str = field(default_factory=str)
    trajectory_max_length: int = 5
    index: int = 0
    reflection: str = ''
    temperature: float = 0.3
    output_format: dict = field(default_factory=dict)
    _: KW_ONLY
    proxy: str | None = None

    def __post_init__(self: "Self", **kwargs: "KWARGS") -> None:
        self.llm = ChatOpenAI(
            temperature=self.temperature,
            openai_api_base=self.api_base,
            openai_api_key=self.api_key,
            openai_proxy=self.proxy,
            model=self.model_name,
        )
        self.prompt: AgentPrompt = AgentPrompt.from_file(
            template_path=self.template,
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        if kwargs is not None:
            (setattr(self, key, value) for key, value in kwargs.items())

        print(self.prompt)
        self.actions = dump_instance(
            Action,
            self.env.action_space.actions,
            many=True,
        )

        self.output_format = dump_instance(
            OutputFormat,
            OutputFormat(),
        )
        self.memory: list[dict] = []

    def update_memory(self: "Self", memory: list):
        """Updates the reflections."""
        if len(memory) > 3:
            self.memory = memory[-3:]
        else:
            self.memory = memory

    def next_action(self: "Self", observation: str) -> tuple["Action", dict]:
        """Predict the next action given the observation."""
        prompt_variables: dict = dump_instance(
            self.__class__,
            self,
            only=self.prompt.input_variables,
        )
        if len(self.memory) > 0:
            self.reflection = self.memory

        reflection_info_str = f"Reflection: {self.reflection}"
        log(Config.VERBOSE_LEVEL, reflection_info_str)
        prompt_info_str = f"Prompt variables: {prompt_variables}"
        log(Config.VERBOSE_LEVEL, prompt_info_str)
        llm_return_str: str = self.chain.run(**prompt_variables)
        return_info_str = f"LLM return: {llm_return_str}"
        log(Config.VERBOSE_LEVEL, return_info_str)
        action, llm_return = self.output_parse(llm_return_str=llm_return_str)
        llm_return.observation = observation
        self.observation = observation
        self.trajectory.append(dump_instance(OutputFormat, llm_return))
        self.trajectory.pop(0) if len(
            self.trajectory,
        ) > self.trajectory_max_length else None
        return (action, llm_return.action.input_args_schema)

    def plan(self: "Self", observation: tuple) -> list["Action"]:
        """Plan the actions given the observation."""
        action_plan = []
        think_plan = []

        if len(self.memory) > 0:
            self.reflection = self.memory

        self.observation = observation
        prompt_variables: dict = dump_instance(
            self.__class__,
            self,
            only=self.prompt.input_variables,
        )

        prompt_info_str = f"Prompt variables: {prompt_variables}"
        log(Config.VERBOSE_LEVEL, prompt_info_str)
        llm_return_str: str = self.chain.run(**prompt_variables)
        return_info_str = f"LLM return: {llm_return_str}"
        log(Config.VERBOSE_LEVEL, return_info_str)
        llm_return_list = llm_return_str.splitlines()
        for lms in llm_return_list:
            think = lms[9:(lms.find('action') - 3)]
            think = think[1:]
            think_plan.append(think)
            lms = lms[(lms.find('action') - 1):-1]

            lms = '{' + lms + '}'
            if lms == '' or lms.count(
                    '}') != 3 or 'action' not in lms or 'name' not in lms or 'input_args_schema' not in lms:
                continue
            else:
                while lms[0] != '{':
                    lms = lms[1:]
                lms = lms[:lms.rfind('}') + 1]
            action, llm_return = self.output_parse(llm_return_str=lms)
            action_plan.append((action, llm_return.action.input_args_schema))


        return action_plan, think_plan

    def reset(self: "Self", *_args: tuple, memory) -> None:
        """重置."""
        self.trajectory = []
        if len(memory) > 3:
            self.memory = memory[-3:]
        else:
            self.memory = memory
        self.actions = dump_instance(
            Action,
            self.env.action_space.actions,
            many=True,
        )
        if self.env.env_task is not None:
            self.task = self.env.env_task
        self.env.reset()

    def output_parse(
            self: "Self",
            llm_return_str: str,
            none_action_name: str = "none",
    ) -> tuple["Action", OutputFormat]:
        """解析LLM返回.

        Args:
            llm_return_str (str): LLM返回
            none_action_name (str, optional): 无动作名称. Defaults to "none".

        Returns:
            (Action, dict): action, llm_return
        """
        try:
            llm_return: OutputFormat = make_instance(
                OutputFormat,
                **literal_eval(
                    llm_return_str.replace("\r", "").replace("\n", ""),
                ),
            )
        except Exception as e:
            if none_action_name not in self.env.action_space.action_names:
                msg = f"Action {none_action_name} not in action space"
                raise ValueError(msg) from e
            action = self.env.action_space.get(action_name=none_action_name)
            llm_return = OutputFormat()
        else:
            if llm_return.action.name not in self.env.action_space.action_names:
                msg = f"Action {llm_return.action.name} not in action space"
                warning(msg)
                action = self.env.action_space.get(action_name=none_action_name)
            else:
                action: Action = self.env.action_space.get(
                    action_name=llm_return.action.name,
                )
        return action, llm_return

    def set_task(self: "Self", task: str) -> None:
        """设置任务."""
        self.task = task