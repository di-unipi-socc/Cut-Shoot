"""A class to virtualize the quantum provider wrappers."""

from typing import Any, Callable, Optional

from qukit.backends.backend_wrapper import BackendWrapper
from qukit.backends.provider_wrapper import ProviderWrapper
from qukit.utils.immutabiliy import immutable
from qukit.utils.parallel import ThreadWithReturnValue as ThreadRV
from qukit.utils.utils import get_imported_classes


@immutable
class VirtualProvider(ProviderWrapper[BackendWrapper[Any]]):
    """A class to virtualize the quantum provider wrappers.

    Parameters
    ----------
    providers : Optional[list[str]], optional
        The providers, by default None.
        If None, all the providers are used.
        If not None, only the providers in the list are used.
    **kwargs : Any
        The keyword arguments.
    """

    def __init__(self, providers: Optional[list[str]] = None, **kwargs: Any):
        """Initialize the class.

        Parameters
        ----------
        providers : Optional[list[str]], optional
            The providers, by default None.
            If None, all the providers are used.
            If not None, only the providers in the list are used.
        **kwargs : Any
            The keyword arguments.
        """
        self._providers = providers
        self._kwargs = kwargs

    def _backends(self, **kwargs: Any) -> list[BackendWrapper[Any]]:
        """Return the backends of all imported providers.

        Parameters
        ----------
        **kwargs : Any
            The keyword arguments.

        Returns
        -------
        list[BackendWrapper[Any]]
            The backends of all imported providers.
        """

        threads: list[ThreadRV] = []
        backends: list[BackendWrapper[Any]] = []
        for provider in ProviderWrapper.__subclasses__():
            if (
                provider.__name__ != VirtualProvider.__name__
                and provider.__name__ in get_imported_classes(ProviderWrapper)
                and (not self._providers or provider().provider_id in self._providers)  # type: ignore
            ):

                thread = ThreadRV(
                    target=provider(  # type: ignore  # pylint: disable=protected-access
                        **kwargs, **self._kwargs
                    )._backends,
                    kwargs=kwargs,
                )
                thread.start()
                threads.append(thread)

        for thread in threads:
            backends.extend(thread.join())

        return backends

    def backends_by_provider(
        self,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
        predicate: Callable[[BackendWrapper[Any]], bool] = lambda x: True,
        **kwargs: Any
    ) -> dict[str, list[BackendWrapper[Any]]]:
        """Return the backends by provider.

        Parameters
        ----------
        include : Optional[list[str]], optional
            The providers to include, by default None.
            If None, all providers are included.
            If not None, only the providers in the list are included.
        exclude : Optional[list[str]], optional
            The providers to exclude, by default None.
            If None, no providers are excluded.
            If not None, the providers in the list are excluded.
        predicate : Callable[[BackendWrapper[Any]], bool], optional
            The predicate to filter the backends, by default lambda x: True.
        **kwargs : Any
            The keyword arguments.

        Returns
        -------
        dict[str, list[BackendWrapper[Any]]]
            The backends by provider id.
        """

        def _predicate(backend: BackendWrapper[Any]) -> bool:
            """The predicate to filter the backends.

            Parameters
            ----------
            backend : BackendWrapper[Any]
                The backend.

            Returns
            -------
            bool
                True if the backend is accepted, False otherwise.
            """
            return (
                predicate(backend)
                and (backend.provider in include if include else True)
                and (backend.provider not in exclude if exclude else True)
            )

        _backends = self.backends(available=False, predicate=_predicate, **kwargs)
        backends: dict[str, list[BackendWrapper[Any]]] = {}
        for backend in _backends:
            if backend.provider not in backends:
                backends[backend.provider] = []
            backends[backend.provider].append(backend)

        return backends

    @property
    def providers(self) -> list[type[ProviderWrapper[Any]]]:
        """Return the providers.

        Returns
        -------
        list[str]
            The providers.
        """
        _providers = []
        for provider in ProviderWrapper.__subclasses__():
            if (
                provider.__name__ != VirtualProvider.__name__
                and provider.__name__ in get_imported_classes(ProviderWrapper)
                and (not self._providers or provider().provider_id in self._providers)  # type: ignore
            ):
                _providers.append(provider)

        return _providers

    def get_backend(self, name: str, provider: Optional[str] = None) -> Optional[BackendWrapper[Any]]:
        """Return the backend with the given name. The name can be the name of the backend or the name
        "{provider}:{backend}". The provider can be specified also as the second argument. If the provider is
        not specified, the first backend with the given name is returned.

        Parameters
        ----------
        name : str
            The name of the backend.
        provider : Optional[str], optional
            The provider of the backend, by default None.

        Returns
        -------
        Optional[BackendWrapper[Any]]
            The backend.
        """
        if ":" in name:
            provider, name = name.split(":")

        if provider:
            for _provider in self.providers:
                if _provider().provider_id == provider:
                    try:
                        return _provider().get_backend(name)
                    except Exception:  # pylint: disable=broad-except
                        pass
            return None

        def name_predicate(x: BackendWrapper[Any]) -> bool:
            """The predicate to filter the backends by name.

            Parameters
            ----------
            x : BackendWrapper[Any]
                The backend.

            Returns
            -------
            bool
                True if the backend is accepted, False otherwise.
            """
            return x.name == name

        backends = self.backends(available=False, predicate=name_predicate)
        for backend in backends:
            if backend.name == name:
                return backend
        return None
