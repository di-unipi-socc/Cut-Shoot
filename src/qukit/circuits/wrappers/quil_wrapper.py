"""A wrapper for Quil circuits."""

import cirq_rigetti  # type: ignore
import pyquil  # type: ignore
from cirq.contrib.qasm_import import circuit_from_qasm  # type: ignore
from qbraid.transpiler import transpile  # type: ignore

from qukit.circuits.circuit_wrapper import CircuitWrapper
from qukit.utils.immutabiliy import immutable


@immutable
class QuilCircuitWrapper(CircuitWrapper[str]):  # pylint: disable=too-few-public-methods
    """A class to wrap a Quil circuits."""

    @staticmethod
    def _to_openqasm2(circuit: str) -> str:
        """Return the OpenQASM2 circuit.

        Parameters
        ----------
        circuit : str
            The Quil circuit.

        Returns
        -------
        str
            The OpenQASM2 circuit.
        """
        program = pyquil.Program(circuit)
        return str(cirq_rigetti.circuit_from_quil(program).to_qasm())

    @staticmethod
    def _from_openqasm2(qasm: str) -> str:
        """Return the Quil circuit.

        Parameters
        ----------
        qasm : str
            The OpenQASM2 circuit.

        Returns
        -------
        str
            The Quil circuit.
        """
        cirq_circuit = circuit_from_qasm(qasm)
        return str(transpile(cirq_circuit, "pyquil").out())
