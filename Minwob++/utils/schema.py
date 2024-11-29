"""工具包."""
from collections.abc import Callable
from importlib.util import find_spec, module_from_spec
from typing import TYPE_CHECKING, ClassVar, Literal, TypeVar

from marshmallow import INCLUDE
from marshmallow import Schema as OriginalSchema
from marshmallow.fields import Field
from marshmallow_dataclass import class_schema

if TYPE_CHECKING:
    from importlib.machinery import ModuleSpec
    from types import ModuleType

    from typing_extensions import Self

    _C = TypeVar("_C")
    V = TypeVar("V")
spec: "ModuleSpec" = (
    find_spec(name="orjson")
    or find_spec(name="ujson")
    or find_spec(name="json")
)
json: "ModuleType" = module_from_spec(spec=spec)
spec.loader.exec_module(module=json)
ARGS = TypeVar("ARGS")
KWARGS = TypeVar("KWARGS")


class CallableField(Field):
    """Function."""

    default_error_messages: ClassVar[dict[Literal["invalid"], str]] = {
        "invalid": "Not a valid function.",
    }

    def __init__(self: "Self", *args: "ARGS", **kwargs: "KWARGS") -> None:
        """Init."""
        super().__init__(*args, **kwargs)

    def _deserialize(
        self: "Self",
        value: "V",
        _attr: str,
        _obj: object,
    ) -> Callable:
        """Deserialize."""
        if not callable(value):
            self.fail(key="invalid")
        return value

    def _serialize(
        self: "Self",
        value: "Callable",
        _attr: str | None,
        _obj: object,
    ) -> str:
        return value.__doc__


class Schema(OriginalSchema):
    """Change default json."""

    TYPE_MAPPING: ClassVar[dict[type, Field]] = {
        Callable: CallableField,
    }

    class Meta:
        """Change default json."""

        render_module = json
        unknown = INCLUDE


def make_instance(cls: type["_C"], **data: dict) -> "_C":
    """Make instance.

    Args:
        cls (type[_C]): dataclass
        data (dict): dict data to deserialize
    """
    return class_schema(clazz=cls, base_schema=Schema)().load(data=data)


def dump_instance(
    cls: type["_C"],
    instance: "_C",
    *,
    many: bool = False,
    **kwargs: "KWARGS",
) -> dict | list[dict]:
    """Dump instance.

    Args:
        cls (type[_C]): dataclass
        instance (_C): dataclass instance
        many (bool, optional): dump list or not. Defaults to False.
        kwargs: kwargs for create schema

    Returns:
        dict | list[dict]: dict data
    """
    return class_schema(clazz=cls, base_schema=Schema)(**kwargs).dump(
        instance,
        many=many,
    )
