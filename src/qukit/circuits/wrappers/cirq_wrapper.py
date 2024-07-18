"""A wrapper for Cirq circuits."""

import cirq  # type: ignore
from cirq.contrib.qasm_import import circuit_from_qasm  # type: ignore

from qukit.circuits.circuit_wrapper import CircuitWrapper
from qukit.utils.immutabiliy import immutable


@immutable
class CirqCircuitWrapper(CircuitWrapper[cirq.Circuit]):  # pylint: disable=too-few-public-methods
    """A class to wrap a Cirq circuits."""

    @staticmethod
    def _to_openqasm2(circuit: cirq.Circuit) -> str:
        """Return the OpenQASM2 circuit.

        Parameters
        ----------
        circuit : cirq.Circuit
            The Cirq circuit.

        Returns
        -------
        str
            The OpenQASM2 circuit.
        """
        return str(circuit.to_qasm())

    @staticmethod
    def _from_openqasm2(qasm: str) -> cirq.Circuit:
        """Return the Cirq circuit.

        Parameters
        ----------
        qasm : str
            The OpenQASM2 circuit.

        Returns
        -------
        cirq.Circuit
            The Cirq circuit.
        """
        qasm_no_barriers = "\n".join(line for line in qasm.split("\n") if "barrier" not in line)
        return circuit_from_qasm(qasm_no_barriers)
