"""A wrapper for PyQuil circuits."""

import cirq_rigetti  # type: ignore
import pyquil  # type: ignore
from cirq.contrib.qasm_import import circuit_from_qasm  # type: ignore
from qbraid.transpiler import transpile  # type: ignore

from qukit.circuits.circuit_wrapper import CircuitWrapper
from qukit.utils.immutabiliy import immutable


@immutable
class PyQuilCircuitWrapper(CircuitWrapper[pyquil.Program]):  # pylint: disable=too-few-public-methods
    """A class to wrap a PyQuil circuits."""

    @staticmethod
    def _to_openqasm2(circuit: pyquil.Program) -> str:
        """Return the OpenQASM2 circuit.

        Parameters
        ----------
        circuit : pyquil.Program
            The PyQuil circuit.

        Returns
        -------
        str
            The OpenQASM2 circuit.
        """
        return str(cirq_rigetti.circuit_from_quil(circuit).to_qasm())

    @staticmethod
    def _from_openqasm2(qasm: str) -> pyquil.Program:
        """Return the PyQuil circuit.

        Parameters
        ----------
        qasm : str
            The OpenQASM2 circuit.

        Returns
        -------
        pyquil.Program
            The PyQuil circuit.
        """
        cirq_circuit = circuit_from_qasm(qasm)
        return transpile(cirq_circuit, "pyquil")
