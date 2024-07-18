"""Some utils for QuKit."""

import sys
import warnings
from inspect import getmembers, isclass, signature
from threading import Lock
from typing import Any, Callable, Optional

lock = Lock()


def get_imported_classes(target_cls: Optional[type] = None) -> list[str]:
    """Get the imported classes. If a class is provided, only the classes that are subclasses of the provided
    class are returned.

    Parameters
    ----------
    target_cls : Optional[type], optional
        The class to check for subclasses, by default None.

    Returns
    -------
    list[str]
        The imported classes.
    """
    classes = []
    for module_name, module in sys.modules.copy().items():  # pylint: disable=too-many-nested-blocks
        if module and not module_name.startswith("_"):  # Skip built-in modules and those starting with _
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with lock:  # IDK why it is needed, check FIXME # pylint: disable=fixme
                        module_classes = getmembers(module, isclass)

                for class_name, class_obj in module_classes:
                    if class_obj.__module__ == module_name:  # Ensure the class is defined in the module
                        if (
                            target_cls is None
                            or issubclass(class_obj, target_cls)
                            and class_name != target_cls.__name__
                        ):
                            classes.append(class_name)
            except Exception as e:  # pragma: no cover # pylint: disable=broad-except
                if "No module named" not in str(e):
                    raise e
    return classes


def call_fun_with_kwargs(fun: Callable[..., Any], kwargs: Any) -> Any:
    """Call a function with the right kwargs.

    Parameters
    ----------
    fun : Callable[..., Any]
        The function to call.
    kwargs : Any
        The keyword arguments.

    Returns
    -------
    Any
        The return value of the function.
    """
    right_kwargs = kwargs.copy()
    sig = signature(fun)
    if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        for missing in right_kwargs.keys() - sig.parameters.keys():
            del right_kwargs[missing]
    return fun(**right_kwargs)
