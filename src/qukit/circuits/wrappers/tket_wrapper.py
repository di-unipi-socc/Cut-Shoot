"""A wrapper for Tket circuits."""

from pytket import Circuit  # type: ignore
from pytket.qasm import circuit_from_qasm_str, circuit_to_qasm_str  # type: ignore

from qukit.circuits.circuit_wrapper import CircuitWrapper
from qukit.utils.immutabiliy import immutable


@immutable
class TketCircuitWrapper(CircuitWrapper[Circuit]):  # pylint: disable=too-few-public-methods
    """A class to wrap a Tket Circuit."""

    @staticmethod
    def _to_openqasm2(circuit: Circuit) -> str:
        """Return the OpenQASM2 circuit.

        Parameters
        ----------
        circuit : Circuit
            The Tket circuit.

        Returns
        -------
        str
            The OpenQASM2 circuit.
        """
        return str(circuit_to_qasm_str(circuit))

    @staticmethod
    def _from_openqasm2(qasm: str) -> Circuit:
        """Return the Tket circuit.

        Parameters
        ----------
        qasm : str
            The OpenQASM2 circuit.

        Returns
        -------
        Circuit
            The Tket circuit.
        """
        return circuit_from_qasm_str(qasm)
