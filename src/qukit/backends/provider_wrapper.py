"""Interface for wrapping quantum providers."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Optional, TypeVar

from qukit.backends.backend_wrapper import BackendWrapper
from qukit.utils.immutabiliy import immutable

BT = TypeVar("BT", bound=BackendWrapper[Any])


@immutable
class ProviderWrapper(ABC, Generic[BT]):
    """An abstract wrapper for providers."""

    @abstractmethod
    def _backends(self) -> list[BT]:
        """Return the backends.

        Returns
        -------
        list[BT]
            The available backends.
        """

    def backends(  # pylint: disable=too-many-branches
        self,
        available: bool = True,
        simulators: bool = True,
        qpus: bool = True,
        predicate: Callable[[BT], bool] = lambda x: True,
        **kwargs: Any,
    ) -> list[BT]:
        """Return the backends.

        Parameters
        ----------
        available : bool, optional
            If True, return only the available backends, by default True.
        simulators : bool, optional
            If True, return the simulators, by default True.
        qpus : bool, optional
            If True, return the quantum processing units, by default True.
        predicate : Callable[[BT], bool], optional
            The predicate function, by default lambda x: True.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        list[BT]
            The backends.
        """
        backends = self._backends(**kwargs)
        _backends = []
        for backend in backends:
            if predicate(backend):
                _backends.append(backend)

        backends = _backends
        _backends = []
        if available:
            for backend in backends:
                if backend.available:
                    _backends.append(backend)
        else:
            _backends = backends

        backends = _backends
        _backends = []
        if not simulators:
            for backend in backends:
                if not backend.simulator:
                    _backends.append(backend)
        else:
            _backends = backends

        backends = _backends
        _backends = []
        if not qpus:
            for backend in backends:
                if backend.simulator:
                    _backends.append(backend)
        else:
            _backends = backends

        return _backends

    def get_backend(self, name: str) -> Optional[BT]:
        """Return the backend with the given name.

        Parameters
        ----------
        name : str
            The name of the backend.

        Returns
        -------
        BT
            The backend.
        """
        backends = self.backends(available=False)
        for backend in backends:
            if backend.name == name:
                return backend
        return None

    @property
    def provider_id(self) -> str:
        """Return the provider ID.

        Returns
        -------
        str
            The provider ID.
        """
        return self.__class__.__name__.lower().replace("providerwrapper", "")

    @property
    def backend_type(self) -> Any:
        """Return the backend type of the provider.

        Returns
        -------
        Any
            The backend type of the provider.
        """
        return self.__orig_bases__[0].__args__[0]  # type: ignore # pylint: disable=no-member

    def __str__(self) -> str:
        """Return the name of the provider.

        Returns
        -------
        str
            The name of the provider.
        """
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        """Return the name of the provider.

        Returns
        -------
        str
            The name of the provider.
        """
        return self.__str__()
