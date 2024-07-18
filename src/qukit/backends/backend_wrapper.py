"""Interface for wrapping quantum backends."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Generic, Optional, Sequence, TypeVar, Union

from qukit.backends.job import Job, JobStatus
from qukit.backends.result import Result
from qukit.circuits.circuit_wrapper import CircuitWrapper
from qukit.circuits.virtual_circuit import VirtualCircuit
from qukit.utils.immutabiliy import initbyvalue, passbyvalue
from qukit.utils.utils import call_fun_with_kwargs

CT = TypeVar("CT", bound=CircuitWrapper[Any])


class BackendStatus(Enum):
    """The status of the backend."""

    ONLINE = "online"
    AVAILABLE = "available"
    OFFLINE = "offline"
    PAUSED = "paused"
    MAINTENANCE = "maintenance"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@initbyvalue
class BackendWrapper(ABC, Generic[CT]):
    """An abstract wrapper for quantum backends.

    Parameters
    ----------
    name : str
        The name of the backend.
    config : Optional[dict[str, Any]], optional
        The configuration of the backend, by default None.
    device : Optional[str], optional
        The device name, by default is the same as the name.
    """

    def __init__(self, name: str, config: Optional[dict[str, Any]] = None, *, device: Optional[str] = None):
        """Initialize the backend wrapper.

        Parameters
        ----------
        name : str
            The name of the backend.
        config : Optional[dict[str, Any]], optional
            The configuration of the backend, by default None.
        device : Optional[str], optional
            The device name, by default is the same as the name.
        """
        self._name = name
        self._config = config or {}
        if device is None:
            self._device = name
        else:
            self._device = device

        self.update()

    @property
    @abstractmethod
    def status(self) -> BackendStatus:
        """Return the status of the backend.

        Returns
        -------
        BackendStatus
            The status of the backend.
        """

    @property
    @abstractmethod
    def qubits(self) -> int:
        """Return the number of qubits of the backend.

        Returns
        -------
        int
            The number of qubits of the backend.
        """

    @property
    @abstractmethod
    def simulator(self) -> bool:
        """Return True if the backend is a simulator.

        Returns
        -------
        bool
            True if the backend is a simulator.
        """

    @property
    @abstractmethod
    def noisy(self) -> bool:
        """Return True if the backend is noisy.

        Returns
        -------
        bool
            True if the backend is noisy.
        """

    @abstractmethod
    def update(self) -> None:
        """Update the backend information."""

    @abstractmethod
    def get_job(self, job_id: str) -> Optional[Job]:
        """Return the job with the given id.

        Parameters
        ----------
        job_id : str
            The id of the job.

        Returns
        -------
        Optional[Job]
            The job with the given id.
        """

    @abstractmethod
    def job_status(self, job: Job) -> JobStatus:
        """Return the status of the job.

        Parameters
        ----------
        job : Job
            The job.

        Returns
        -------
        JobStatus
            The status of the job.
        """

    @abstractmethod
    def job_results(self, job: Job) -> Optional[list[Result]]:
        """Return the results of the job.

        Parameters
        ----------
        job : Job
            The job.

        Returns
        -------
        Optional[list[Result]]
            The results of the job.
        """

    @passbyvalue
    @abstractmethod
    def _run(
        self,
        circuits: Sequence[CT],
        shots: int = 1024,
        asynch: bool = False,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Job:
        """Run the circuits on the backend.

        Parameters
        ----------
        circuits : Sequence[CT]
            The circuits to run.
        shots : int, optional
            The number of shots, by default 1024.
        asynch : bool, optional
            If True, run the circuits asynchronously, by default False.
        metadata : dict[str, Any], optional
            The metadata of the job, by default None.
        **kwargs : Any
            Additional keyword arguments depending on the actual backend wrapper.

        Returns
        -------
        Job
            The job of the run.
        """

    def run(  # pylint: disable=too-many-arguments
        self,
        circuits: Union[CT, Sequence[CT], VirtualCircuit, list[VirtualCircuit]],
        shots: int = 1024,
        asynch: bool = False,
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Job:
        """Run the circuits on the backend.

        Parameters
        ----------
        circuits : Union[CT, Sequence[CT]]
            The circuits to run.
        shots : int, optional
            The number of shots, by default 1024.
        asynch : bool, optional
            If True, run the circuits asynchronously, by default False.
        name : str, optional
            The name of the job, by default None.
        metadata : dict[str, Any], optional
            The metadata of the job, by default None.
        **kwargs : Any
            Additional keyword arguments depending on the actual backend wrapper.

        Returns
        -------
        Job
            The job of the run.
        """
        if not isinstance(circuits, list):
            circuits = [circuits]  # type: ignore

        if self.status != BackendStatus.AVAILABLE:  # pragma: no cover
            raise ValueError(f"Backend {self.device} is not available to run circuits")

        _circuits = []

        for circuit in circuits:  # type: ignore
            if isinstance(circuit, VirtualCircuit):
                circuit = circuit.translate(self.accepts().language())

            if not self.accepts().is_valid(circuit.circuit):
                raise ValueError(f"Invalid circuit: {circuit} for {self.__class__.__name__}")
            _circuits.append(circuit)

        if metadata is None:
            metadata = {}

        metadata["shots"] = shots
        if name is not None:
            metadata["name"] = name

        metadata["circuits"] = [
            circuit.circuit for circuit in _circuits
        ]  # TODO: change with circuit.to_openqasm2().circuit when IonQ supports it # pylint: disable=fixme
        metadata["circuits_metadata"] = [circuit.metadata for circuit in _circuits]

        metadata["backend"] = {}
        metadata["backend"]["device"] = self.device
        metadata["backend"]["provider"] = self.provider
        metadata["backend"]["name"] = self.name
        metadata["backend"] = metadata["backend"]

        return self._run(_circuits, shots, asynch, metadata, **kwargs)  # type: ignore

    def wait_for_job(self, job: Job, **kwargs: Any) -> list[Result]:
        """Wait for the job to finish and return the results.

        Parameters
        ----------
        job : Job
            The job.
        **kwargs : Any
            Additional keyword arguments depending on the actual backend wrapper.

        Returns
        -------
        list[Result]
            The results of the job.
        """

        while not call_fun_with_kwargs(self.job_finished, {"job": job, **kwargs}):
            pass
        return call_fun_with_kwargs(self.job_results, {"job": job, **kwargs})  # type: ignore

    def job_finished(self, job: Job, **kwargs: Any) -> bool:
        """Return True if the job has finished.

        Parameters
        ----------
        job : Job
            The job.
        **kwargs : Any
            Additional keyword arguments depending on the actual backend wrapper.

        Returns
        -------
        bool
            True if the job has finished.
        """
        return call_fun_with_kwargs(self.job_status, {"job": job, **kwargs}) in [
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELED,
        ]

    @property
    def name(self) -> str:
        """Return the name of the backend.

        Returns
        -------
        str
            The name of the backend.
        """
        return self._name

    @property
    def config(self) -> dict[str, Any]:
        """Return the configuration of the backend.

        Returns
        -------
        dict[str, Any]
            The configuration of the backend.
        """
        return self._config

    @property
    def device(self) -> str:
        """Return the device name of the backend.

        Returns
        -------
        str
            The device name of the backend.
        """
        return self._device

    @property
    def provider(self) -> str:
        """Return the provider for the backend.

        Returns
        -------
        str
            The provider for the backend.
        """
        return self.__class__.__name__.lower().replace("backendwrapper", "")

    @property
    def available(self) -> bool:
        """Return True if the backend is online.

        Returns
        -------
        bool
            True if the backend is online and accessible.
        """
        return self.status is BackendStatus.AVAILABLE

    @classmethod
    def accepts(cls) -> Any:
        """Return the circuit wrapper that the backend accepts.

        Returns
        -------
        Any
            The circuit wrapper that the backend accepts.
        """
        return cls.__orig_bases__[0].__args__[0]  # type: ignore # pylint: disable=no-member

    def info(self) -> str:
        """Return the information of the backend.

        Returns
        -------
        str
            The information of the backend.
        """
        return f"""{self.name} ({self.device}) from {self.provider} provider

\tStatus: {self.status}
\tQubits: {self.qubits}
\tSimulator: {self.simulator}
\tAccepts: {self.accepts().language()}

\tConfiguration: {self.config}
    """

    def __str__(self) -> str:
        """Return the string representation of the backend wrapper.

        Returns
        -------
        str
            The string representation of the backend wrapper.
        """
        s = ""
        if self._device == self._name:
            s = f"{self.__class__.__name__}({self._name})"
        else:
            s = f"{self.__class__.__name__}({self._name}<{self._device}>)"
        if self._config:
            s += f"[{self._config}]"

        return s

    def __repr__(self) -> str:
        """Return the string representation of the backend wrapper.

        Returns
        -------
        str
            The string representation of the backend wrapper.
        """
        return str(self)
