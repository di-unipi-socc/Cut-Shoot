"""Module to ensure immutability of function arguments."""

from copy import deepcopy
from inspect import currentframe, getframeinfo, getsourcelines
from typing import Any, Callable

from frozendict import frozendict  # type: ignore


def get_decorators(cls: type) -> list[str]:
    """Return the decorators of the class.

    Returns
    -------
    list[str]
        The decorators of the class.
    """

    decorators = []
    try:
        for line in getsourcelines(cls)[0]:
            if line.strip().startswith("@"):
                decorators.append(line.strip()[1:])
    except OSError:
        pass
    return decorators


def immutable_copy(obj: object) -> object:
    """Return a deep copy of the object.

    If the object is decorated with the immutable decorator, return the object itself.

    Parameters
    ----------
    obj : object
        The object to copy.

    Returns
    -------
    object
        The deep copy of the object.
    """

    decorators = []
    try:
        decorators = get_decorators(obj.__class__)
    except TypeError:
        pass

    if "immutable" in decorators:
        return obj
    return deepcopy(obj)


def passbyvalue(func: Callable[..., object]) -> Callable[..., object]:
    """Decorator to make sure that the arguments are passed by value.

    Parameters
    ----------
    func : Callable[..., object]
        The function to decorate.

    Returns
    -------
    Callable[..., object]
        The decorated function.
    """

    def _passbyvalue(*args: Any, **kwargs: Any) -> Any:
        """The decorated function.

        Parameters
        ----------
        *args : Any
            The arguments of the function.
        **kwargs : Any
            The keyword arguments of the function.

        Returns
        -------
        Any
            The return value of the function.
        """

        cargs = [immutable_copy(arg) for arg in args]
        ckwargs = {key: immutable_copy(value) for key, value in kwargs.items()}
        return func(*cargs, **ckwargs)

    return _passbyvalue


def initbyvalue(cls: type) -> type:
    """Decorator to make sure that the arguments are passed by value in the __init__ method.

    Returns
    -------
    type
        The decorated class.
    """

    init_func = cls.__init__  # type: ignore

    def __init__wrapper(self: object, *args: Any, **kwargs: Any) -> None:
        """The decorated __init__ method.

        Parameters
        ----------
        *args : Any
            The arguments of the __init__ method.
        **kwargs : Any
            The keyword arguments of the __init__ method.
        """
        cargs = [immutable_copy(arg) for arg in args]
        ckwargs = {key: immutable_copy(value) for key, value in kwargs.items()}
        init_func(self, *cargs, **ckwargs)

    cls.__init__ = __init__wrapper  # type: ignore

    return cls


def freeze(value: Any) -> Any:
    """Freeze the value.

    Make the value immutable.

    Parameters
    ----------
    value : Any
        The value to freeze.

    Returns
    -------
    Any
        The frozen value.
    """

    if isinstance(value, (int, float, str, bool, type(None))):
        return value

    if isinstance(value, set):
        return frozenset(freeze(item) for item in value)

    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)

    if isinstance(value, dict):
        return frozendict({freeze(key): freeze(value) for key, value in value.items()})

    forzen_value = immutable_copy(value)

    def frozen__setattr__(self: object, name: str, value: Any) -> None:
        """Raise an TypeError when trying to set an attribute.

        Attributes can only be set in the __init__ method.

        Parameters
        ----------
        name : str
            The name of the attribute to set.
        value : Any
            The value of the attribute to set.

        Raises
        ------
        TypeError
            If the attribute is not set in the __init__ method.
        """

        raise TypeError(f"Cannot set attribute '{name}' of immutable object '{self.__class__.__name__}'")

    forzen_value.__dict__["__setattr__"] = frozen__setattr__

    def frozen__delattr__(self: object, name: str) -> None:
        """Raise an TypeError when trying to delete an attribute.

        Parameters
        ----------
        name : str
            The name of the attribute to delete.

        Raises
        ------
        TypeError
            If the attribute is tried to be deleted.
        """
        raise TypeError(f"Cannot delete attribute '{name}' of immutable object '{self.__class__.__name__}'")

    forzen_value.__dict__["__delattr__"] = frozen__delattr__

    return forzen_value


def immutable(cls: type) -> type:
    """Decorator to make sure that the class is immutable.

    ATTENTION: Currenlty immutablity works only for objects, not classes.

    E.g.,
        Test.new_dict = {"key": "value"}
        Test.existing_dict = {"key": "new_value"}
        del Test.existing_dict

    are allowed. TODO: find a way to implement immutability for classes.

    Returns
    -------
    type
        The decorated class.
    """

    cls = initbyvalue(cls)

    def __setattr__(self: object, name: str, value: Any) -> None:
        """Raise an TypeError when trying to set an attribute.

        Attributes can only be set in the __init__ method.

        Parameters
        ----------
        name : str
            The name of the attribute to set.
        value : Any
            The value of the attribute to set.

        Raises
        ------
        TypeError
            If the attribute is not set in the __init__ method.
        """

        current_frame = currentframe()
        if current_frame is None:
            raise RuntimeError("Cannot set attribute of immutable object in the current context")

        f_back = current_frame.f_back
        if f_back is None:
            raise RuntimeError("Cannot set attribute of immutable object in the current context")

        current_instance = f_back.f_locals["self"]

        if (  # pylint: disable=too-many-boolean-expressions
            not hasattr(self, name)
            and getframeinfo(f_back).function == "__init__"
            and current_instance is self
        ) or (
            not hasattr(self, name)
            and getframeinfo(f_back).function == "__setattr__"
            and getframeinfo(f_back.f_back).function == "__init__"  # type: ignore
            and current_instance is self
        ):

            super(cls, self).__setattr__(name, freeze(value))  # type: ignore
        else:
            raise TypeError(f"Cannot set attribute '{name}' of immutable object '{self.__class__.__name__}'")

    cls.__setattr__ = __setattr__  # type: ignore

    def __delattr__(self: object, name: str) -> None:
        """Raise an TypeError when trying to delete an attribute.

        Parameters
        ----------
        name : str
            The name of the attribute to delete.

        Raises
        ------
        TypeError
            If the attribute is tried to be deleted.
        """
        raise TypeError(f"Cannot delete attribute '{name}' of immutable object '{self.__class__.__name__}'")

    cls.__delattr__ = __delattr__  # type: ignore

    for key, value in cls.__dict__.items():
        try:
            setattr(cls, key, freeze(value))
        except TypeError:
            pass
        except AttributeError:
            pass

    return cls
