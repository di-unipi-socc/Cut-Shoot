"""This module defines the Job class for quantum circuits execution."""

from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Sequence

from qukit.utils.immutabiliy import initbyvalue

if TYPE_CHECKING:  # pragma: no cover
    from qukit.backends.backend_wrapper import BackendWrapper
    from qukit.backends.result import Result
    from qukit.circuits.circuit_wrapper import CircuitWrapper


class JobStatus(Enum):
    """Enum class for job status."""

    COMPLETED = "COMPLETED"
    RUNNING = "RUNNING"
    QUEUED = "QUEUED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"
    UNKNOWN = "UNKNOWN"


@initbyvalue
class Job:
    """Job class for quantum circuits execution.

    Parameters
    ----------
    job_id : str
        The id of the job.
    backend : BackendWrapper[Any]
        The backend on which the circuit(s) was executed.
    circuits : CircuitWrapper[Any] | Sequence[CircuitWrapper[Any]]
        The circuit(s) that was executed.
    shots : int
        The number of shots used in the execution.
    metadata : Optional[dict[str, Any]]
        The user-defined metadata of the execution.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        job_id: str,
        backend: "BackendWrapper[Any]",
        circuits: "CircuitWrapper[Any] | Sequence[CircuitWrapper[Any]]",
        shots: int,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """Initialize the job.

        Parameters
        ----------
        job_id : str
            The id of the job.
        backend : BackendWrapper[Any]
            The backend on which the circuit(s) was executed.
        circuits : CircuitWrapper[Any] | Sequence[CircuitWrapper[Any]]
            The circuit(s) that was executed.
        shots : int
            The number of shots used in the execution.
        metadata : Optional[dict[str, Any]]
            The user-defined metadata of the execution.
        """
        self._job_id = job_id
        self._backend = backend
        if not isinstance(circuits, Sequence):
            circuits = [circuits]
        self._circuits = circuits
        self._shots = shots
        self._metadata = metadata if metadata is not None else {}
        self._error: Optional[str] = None

    @property
    def id(self) -> str:
        """Return the id of the job.

        Returns
        -------
        str
            The id of the job.
        """
        return self._job_id

    @property
    def backend(self) -> "BackendWrapper[Any]":
        """Return the backend on which the circuit was executed.

        Returns
        -------
        BackendWrapper[Any]
            The backend on which the circuit was executed.
        """
        return self._backend

    @property
    def circuits(self) -> Sequence["CircuitWrapper[Any]"]:
        """Return the circuit(s) that was executed.

        Returns
        -------
        Sequence[CircuitWrapper[Any]]
            The circuit that was executed.
        """
        return self._circuits

    @property
    def shots(self) -> int:
        """Return the number of shots used in the execution.

        Returns
        -------
        int
            The number of shots used in the execution.
        """
        return self._shots

    @property
    def name(self) -> Optional[str]:
        """Return the name of the job.

        Returns
        -------
        Optional[str]
            The name of the job.
        """
        metadata = self.metadata
        name = None
        if "name" in metadata:
            name = str(metadata["name"])
        return name

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the user-defined metadata of the execution.

        Returns
        -------
        dict[str, Any]
            The user-defined metadata of the execution.
        """
        return self.get_metadata(verbose=False)

    def get_metadata(self, verbose: bool = False) -> dict[str, Any]:
        """Return the user-defined metadata of the execution.

        Parameters
        ----------
        verbose : bool
            Whether to return the full metadata
            (comprising the request and response fields)
            or just the metadata field.

        Returns
        -------
        dict[str, Any]
            The user-defined metadata of the execution.
        """
        if verbose:
            return self._metadata
        return self._metadata["metadata"]  # type: ignore

    @property
    def finished(self) -> bool:
        """Return whether the job has finished.

        Returns
        -------
        bool
            Whether the job has finished.
        """
        return self.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELED]

    @property
    def completed(self) -> bool:
        """Return whether the job is completed (i.e., finished and successful).

        Returns
        -------
        bool
            Whether the job is completed (i.e., finished and successful).
        """
        return self.status == JobStatus.COMPLETED

    @property
    def error(self) -> Optional[str]:
        """Return the error message of the job.

        Returns
        -------
        Optional[str]
            The error message of the job.
        """
        return self._error

    @property
    def status(self) -> JobStatus:
        """Return the status of the job.

        Returns
        -------
        str
            The status of the job.
        """
        return self.backend.job_status(self)

    @property
    def results(self) -> Optional[list["Result"]]:
        """Return the results of the job.

        Returns
        -------
        Optional[list[Result]]
            The results of the job if available, None otherwise.
        """
        return self.backend.job_results(self)

    def wait(self) -> list["Result"]:
        """Wait for the job to finish and return the results.

        Returns
        -------
        list[Result]
            The results of the job.
        """
        return self.backend.wait_for_job(self)

    def __str__(self) -> str:
        """Return the string representation of the job.

        Returns
        -------
        str
            The string representation of the job.
        """
        return f"""Job(id={self.id}, backend={self.backend},
circuits={self.circuits}, shots={self.shots},
metadata={self.metadata})"""

    def __repr__(self) -> str:
        """Return the string representation of the job.

        Returns
        -------
        str
            The string representation of the job.
        """
        return self.__str__()
