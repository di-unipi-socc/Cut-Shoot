"""This module implements the Dispatcher class, which is used to dispatch circuits to backends."""

import json
from typing import Any, Callable, Optional
from uuid import uuid4

from qukit.backends import BackendWrapper, Job, VirtualProvider
from qukit.circuits import VirtualCircuit
from qukit.utils.parallel import ThreadWithReturnValue as ThreadRV


class Dispatcher:
    """A class to dispatch circuits to multiple backends from multiple providers.

    If the virtual provider is not provided, a new one is created importing all available providers.

    Parameters
    ----------
    vp : Optional[VirtualProvider], optional
        The virtual provider, by default None.
    """

    def __init__(self, vp: Optional[VirtualProvider] = None):
        """Initialize the class.

        If the virtual provider is not provided, a new one is created importing all available providers.

        Parameters
        ----------
        vp : Optional[VirtualProvider], optional
            The virtual provider, by default None.
        """

        if vp is None:
            vp = VirtualProvider()
        self._vp = vp
        self._dispatches: dict[str, dict[str, dict[str, list[tuple[list[VirtualCircuit], int]]]]] = {}
        self._threads: dict[str, dict[str, dict[str, list[ThreadRV]]]] = {}
        self._waiting: dict[str, dict[str, dict[str, list[Job]]]] = {}

    @property
    def virtual_provider(self) -> VirtualProvider:
        """Return the virtual provider.

        Returns
        -------
        VirtualProvider
            The virtual provider.
        """

        return self._vp

    def providers(self) -> list[str]:
        """Return the providers.

        Returns
        -------
        list[str]
            The providers.
        """

        return [provider().provider_id for provider in self._vp.providers]

    def backends(
        self,
        predicate: Callable[[BackendWrapper[Any]], bool] = lambda x: True,
    ) -> dict[str, list[str]]:
        """Return the backends of all providers.

        Parameters
        ----------
        predicate : Callable[[BackendWrapper[Any]], bool], optional
            The predicate to filter the backends, by default lambda x: True.

        Returns
        -------
        dict[str, list[str]]
            The backends by provider.
        """

        backends = self._vp.backends_by_provider(predicate=predicate)
        return {provider: [backend.name for backend in backends[provider]] for provider in backends}

    def add(  # pylint: disable=too-many-arguments
        self,
        circuits: Any | list[Any],
        backend: str,
        shots: int = 1024,
        provider: Optional[str] = None,
        dispatch: str = "default",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Add circuits to a dispatch.

        Parameters
        ----------
        circuits : Any | list[Any]
            The circuits.
        backend : str
            The backend.
        shots : int, optional
            The number of shots, by default 1024.
        provider : Optional[str], optional
            The provider, by default None.
        dispatch : str, optional
            The dispatch, by default "default".
        metadata : Optional[dict[str, Any]], optional
            The metadata, by default None.

        Raises
        ------
        ValueError
            If the dispatch is already running.
        ValueError
            If the backend is not found in the provider.
        """

        if dispatch in self._threads:  # pragma: no cover
            raise ValueError(f"Dispatch {dispatch} already running")

        if not isinstance(circuits, list):
            circuits = [circuits]

        if dispatch not in self._dispatches:
            self._dispatches[dispatch] = {}

        _backend = self._vp.get_backend(backend, provider)
        if _backend is None:
            raise ValueError(f"Backend {backend} not found in provider {provider}")

        if _backend.provider not in self._dispatches[dispatch]:
            self._dispatches[dispatch][_backend.provider] = {}
        if _backend.name not in self._dispatches[dispatch][_backend.provider]:
            self._dispatches[dispatch][_backend.provider][_backend.name] = []

        _circuits = []
        for circuit in circuits:
            if isinstance(circuit, VirtualCircuit):
                _circuits.append(circuit)
            else:
                _circuits.append(VirtualCircuit(circuit, metadata=metadata))

        self._dispatches[dispatch][_backend.provider][_backend.name].append((_circuits, shots))

    def run(  # pylint: disable=too-many-branches too-many-locals too-many-statements
        self,
        dispatch: Optional[str | dict[str, dict[str, list[tuple[Any, int]]]]] = None,
        metadata: Optional[dict[str, Any]] = None,
        batchify: bool = False,
        asynch: bool = False,
    ) -> dict[str, dict[str, list[Job]]]:
        """Run the dispatch.

        If the dispatch is None, the default dispatch is used.
        If the dispatch is batched, then the circuits are batched in a single run for each backend
        and the shots are the maximum of all the shots for that backend.

        Parameters
        ----------
        dispatch : Optional[str | dict[str, dict[str, list[tuple[Any, int]]]]], optional
            The dispatch, by default None.
            If None, the default dispatch is used.
            If a string, the dispatch with that name is used.
            If a dictionary, the dispatch is used.
        metadata : Optional[dict[str, Any]], optional
            The metadata, by default None.
        batchify : bool, optional
            If True, the circuits are batched in a single run, by default False.
        asynch : bool, optional
            If False, the dispatcher waits for the results, by default False.

        Returns
        -------
        dict[str, dict[str, list[Job]]]
            The results of the dispatch.

        Raises
        ------
        ValueError
            If the dispatch is already running.
        ValueError
            If the dispatch is not found.
        ValueError
            If the backend is not found in the provider.
        """

        if dispatch is None:
            dispatch = "default"

        if isinstance(dispatch, str):  # pylint: disable=too-many-nested-blocks
            if dispatch in self._threads:  # pragma: no cover
                raise ValueError(f"Dispatch {dispatch} already running")
            if dispatch not in self._dispatches:
                raise ValueError(f"Dispatch {dispatch} not found")
            dispatch_name = dispatch
        else:
            dispatch_name = str(uuid4())
            _dispatch: dict[str, dict[str, list[tuple[list[VirtualCircuit], int]]]] = {}
            for provider in dispatch:
                _dispatch[provider] = {}
                for backend in dispatch[provider]:
                    _dispatch[provider][backend] = []
                    for req in dispatch[provider][backend]:
                        if not isinstance(req[0], list):
                            _req_0 = [req[0]]
                        else:
                            _req_0 = req[0]

                        _circuits = []
                        for c in _req_0:
                            if not isinstance(c, VirtualCircuit):
                                c = VirtualCircuit(c, metadata=metadata)
                            _circuits.append(c)
                        _dispatch[provider][backend].append((_circuits, req[1]))

            self._dispatches[dispatch_name] = _dispatch

        dispatch = self._dispatches[dispatch_name]
        self._threads[dispatch_name] = {}

        for provider in dispatch:
            self._threads[dispatch_name][provider] = {}
            for backend in dispatch[provider]:
                self._threads[dispatch_name][provider][backend] = []
                _backend = self._vp.get_backend(backend, provider)

                if _backend is None:
                    raise ValueError(f"Backend {backend} not found in provider {provider}")

                if batchify:
                    _circuits = []
                    _shots = 0
                    for req in dispatch[provider][_backend.name]:
                        for c in req[0]:
                            _circuits.append(c)
                        _shots = max(_shots, req[1])
                    thread = ThreadRV(target=_backend.run, args=(_circuits, _shots, asynch, None, metadata))
                    thread.start()
                    self._threads[dispatch_name][provider][_backend.name].append(thread)
                else:
                    for req in dispatch[provider][_backend.name]:
                        thread = ThreadRV(target=_backend.run, args=(req[0], req[1], asynch, None, metadata))
                        thread.start()
                        self._threads[dispatch_name][provider][_backend.name].append(thread)

        self._dispatches.pop(dispatch_name)

        results: dict[str, dict[str, list[Job]]] = {}
        for provider in dispatch:
            results[provider] = {}
            for backend in dispatch[provider]:
                results[provider][backend] = []
                for thread in self._threads[dispatch_name][provider][backend]:
                    results[provider][backend].append(thread.join())

        if asynch:
            self._waiting[dispatch_name] = results

        self._threads.pop(dispatch_name)
        return results

    def wait(
        self, dispatch: str | dict[str, dict[str, list[Job]]] = "default"
    ) -> dict[str, dict[str, list[Job]]]:
        """Wait for the dispatch to finish.

        Parameters
        ----------
        dispatch : str | dict[str, dict[str, list[Job]]], optional
            The dispatch, by default "default".
            If a string, the dispatch with that name is used.
            If a dictionary, the dispatch is used.

        Returns
        -------
        dict[str, dict[str, list[Job]]]
            The results of the dispatch.

        Raises
        ------
        ValueError
            If the dispatch is not found.
        """

        dispatch_name = None
        if isinstance(dispatch, str):
            if dispatch not in self._waiting:
                raise ValueError(f"Dispatch {dispatch} not found")
            dispatch_name = dispatch
            dispatch = self._waiting[dispatch]

        for provider in dispatch:
            for backend in dispatch[provider]:
                for job in dispatch[provider][backend]:
                    job.wait()

        if dispatch_name is not None:
            self._waiting.pop(dispatch_name)

        return dispatch

    @staticmethod
    def counts(
        dispatch: dict[str, dict[str, list[Job]]]
    ) -> dict[str, dict[str, list[list[dict[str, dict[str, int]]]]]]:
        """Return the counts of the dispatch.

        Parameters
        ----------
        dispatch : dict[str, dict[str, list[Job]]
            The dispatch.

        Returns
        -------
        dict[str, dict[str, list[list[dict[str, dict[str, int]]]]]]
            The counts of the dispatch.
        """

        res: dict[str, dict[str, list[list[dict[str, dict[str, int]]]]]] = {}
        for provider in dispatch:
            res[provider] = {}
            for backend in dispatch[provider]:
                res[provider][backend] = []
                for job in dispatch[provider][backend]:
                    res[provider][backend].append([])
                    for result in job.wait():
                        res[provider][backend][-1].append(json.loads(json.dumps(result.counts)))

        return res

    @staticmethod
    def dispatch(
        dispatch: dict[str, dict[str, list[tuple[Any, int]]]], batchify: bool = False
    ) -> dict[str, dict[str, list[Job]]]:
        """Dispatch the circuits.

        If the dispatch is batched, then the circuits are batched in a single run for each backend
        and the shots are the maximum of all the shots for that backend.

        Parameters
        ----------
        dispatch : dict[str, dict[str, list[tuple[Any, int]]]]
            The dispatch.
        batchify : bool, optional
            If True, the circuits are batched in a single run, by default False.

        Returns
        -------
        dict[str, dict[str, list[Job]]
            The results of the dispatch.
        """

        dispatcher = Dispatcher()
        return dispatcher.run(dispatch, None, batchify, False)
