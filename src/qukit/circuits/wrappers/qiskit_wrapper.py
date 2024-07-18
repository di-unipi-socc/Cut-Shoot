"""Qiskit circuit wrapper."""

from typing import Any, Dict, Optional

from qiskit import QuantumCircuit, qasm2  # type: ignore

from qukit.circuits.circuit_wrapper import CircuitWrapper, OpenQASM2CircuitWrapper
from qukit.utils.immutabiliy import immutable

from threading import Lock


@immutable
class QiskitCircuitWrapper(CircuitWrapper[QuantumCircuit]):
    """Wrapper for Qiskit circuits.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit.
    metadata : dict, optional
        The metadata.
    """

    def __init__(self, circuit: QuantumCircuit, metadata: Optional[Dict[str, Any]] = None):
        """Initialize the class.

        Parameters
        ----------
        circuit : CT
            The circuit.
        metadata : dict, optional
            The metadata.
        """

        metadata = circuit.metadata.copy().update(metadata or {})  # TODO: test it # pylint: disable=fixme
        super().__init__(circuit, metadata)

    @staticmethod
    def _to_openqasm2(circuit: QuantumCircuit) -> str:
        """Return the OpenQASM2 circuit.

        Parameters
        ----------
        circuit : QuantumCircuit
            The Qiskit circuit.

        Returns
        -------
        str
            The OpenQASM2 circuit.
        """
        with OpenQASM2CircuitWrapper._lock:
            return str(qasm2.dumps(circuit))

    @staticmethod
    def _from_openqasm2(qasm: str) -> QuantumCircuit:
        """Return the Qiskit circuit.

        Parameters
        ----------
        qasm : str
            The OpenQASM2 circuit.

        Returns
        -------
        QuantumCircuit
            The Qiskit circuit.
        """
        with OpenQASM2CircuitWrapper._lock:
            return QuantumCircuit.from_qasm_str(qasm)
