"""Module for the results of quantum circuits execution."""

from typing import TYPE_CHECKING, Any

from qukit.utils.immutabiliy import immutable

if TYPE_CHECKING:  # pragma: no cover
    from qukit.backends.backend_wrapper import BackendWrapper
    from qukit.backends.job import Job
    from qukit.circuits.circuit_wrapper import CircuitWrapper


@immutable
class Result:
    """Base class for simple circuit executions results.

    Parameters
    ----------
    job : Job
        The job of the result.
    circuit : CircuitWrapper[Any]
        The circuit that was executed.
    counts : dict[str, dict[str, int]]
        The counts of the results of the execution per classical register.
    """

    def __init__(
        self,
        job: "Job",
        circuit: "CircuitWrapper[Any]",
        counts: dict[str, dict[str, int]],
    ):
        """Initialize the result.

        Parameters
        ----------
        job : Job
            The job of the result.
        circuit : CircuitWrapper[Any]
            The circuit that was executed.
        counts : dict[str, dict[str, int]]
            The counts of the results of the execution per classical register.
        """
        self._job = job
        self._circuit = circuit
        self._counts = counts

    @property
    def job(self) -> "Job":
        """Return the job of the result.

        Returns
        -------
        Job
            The job of the result.
        """
        return self._job

    @property
    def backend(self) -> "BackendWrapper[Any]":
        """Return the backend on which the circuit was executed.

        Returns
        -------
        BackendWrapper
            The backend on which the circuit was executed.
        """
        return self._job.backend

    @property
    def circuit(self) -> "CircuitWrapper[Any]":
        """Return the circuit that was executed.

        Returns
        -------
        CircuitWrapper[Any]
            The circuit that was executed.
        """
        return self._circuit

    @property
    def shots(self) -> int:
        """Return the number of shots used in the execution.

        Returns
        -------
        int
            The number of shots used in the execution.
        """
        return self._job.shots

    @property
    def counts(self) -> dict[str, dict[str, int]]:
        """Return the counts of the results of the execution.

        Returns
        -------
        dict[str, dict[str, int]]
            The counts of the results of the execution per classical register.
        """
        return self._counts

    @property
    def distribution(self) -> dict[str, dict[str, float]]:
        """Return the distribution of the results of the execution.

        Returns
        -------
        dict[str, dict[str, float]]
            The distribution of the results of the execution per classical register.
        """
        counts = self.counts
        shots = self.shots
        return {
            creg: {key: value / shots for key, value in creg_counts.items()}
            for creg, creg_counts in counts.items()
        }

    def __str__(self) -> str:
        """Return a string representation of the result.

        Returns
        -------
        str
            A string representation of the result.
        """
        return f"""Result(job={self.job.id}, backend={self.backend},
shots={self.shots}, circuit={self.circuit},
counts={self.counts}, metadata={self.job.metadata})"""

    def __repr__(self) -> str:
        """Return a string representation of the result.

        Returns
        -------
        str
            A string representation of the result.
        """
        return self.__str__()


# TODO: what happens with multiple classical registers for statevector result? check # pylint: disable=fixme
# class StateVectorResult(Result):
#     """Result of quantum circuits execution with statevector results.

#     Parameters
#     ----------
#     job : Job
#         The job of the result.
#     circuit : CircuitWrapper[Any]
#         The circuit that was executed.
#     statevector : list[complex]
#         The statevector of the results of the execution.
#     """

#     def __init__(
#         self,
#         job: "Job",
#         circuit: "CircuitWrapper[Any]",
#         statevector: "list[complex]",
#     ):
#         """Initialize the statevector result.

#         Parameters
#         ----------
#         job : Job
#             The job of the result.
#         circuit : CircuitWrapper[Any]
#             The circuit that was executed.
#         statevector : list[complex]
#             The statevector of the results of the execution.
#         """
#         super().__init__(job, circuit, {})
#         self._statevector = statevector

#     @property
#     def statevector(self) -> list[complex]:
#         """Return the statevector of the results of the execution.

#         Returns
#         -------
#         list[complex]
#             The statevector of the results of the execution.
#         """
#         return self._statevector.copy()

#     @property
#     def distribution(self) -> dict[str, float]:
#         """Return the distribution of the results of the execution from the statevector.

#         Returns
#         -------
#         dict[str, int]
#             The distribution of the results of the execution
#             from the statevector.
#         """
#         statevector = self.statevector
#         probabilities = [abs(x) ** 2 for x in statevector]
#         _sum_prob = sum(probabilities)
#         probabilities = [p / _sum_prob for p in probabilities]
#         qubits = len(statevector).bit_length() - 1
#         probs: dict[str, float] = {}
#         for i in range(2**qubits):
#             probs[bin(i)[2:].zfill(qubits)] = probabilities[i]
#         return probs

#     @property
#     def counts(self) -> dict[str, int]:
#         """Return the counts of the results of the execution from the statevector.

#         Returns
#         -------
#         dict[str, int]
#             The counts of the results of the execution from the statevector.
#         """

#         if self._counts == {}:
#             probabilities = self.distribution
#             probs_list = list(probabilities.values())
#             _counts = np.random.multinomial(self.shots, probs_list)
#             counts = {}
#             for key, value in zip(probabilities.keys(), _counts):
#                 counts[key] = value
#             self._counts = counts.copy()
#             return counts

#         return self._counts.copy()

#     def __str__(self) -> str:
#         """Return a string representation of the result.

#         Returns
#         -------
#         str
#             A string representation of the result.
#         """
#         return f"""StateVectorResult(job={self.job.id}, backend={self.backend},
# shots={self.shots}, circuit={self.circuit},
# statevector={self.statevector}, metadata={self.job.metadata()})"""

#     def __repr__(self) -> str:
#         """Return a string representation of the result.

#         Returns
#         -------
#         str
#             A string representation of the result.
#         """
#         return self.__str__()
