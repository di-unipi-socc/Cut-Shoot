"""A wrapper for Pennylane tapes."""

import pennylane as qml  # type: ignore

from qukit.circuits.circuit_wrapper import CircuitWrapper
from qukit.utils.immutabiliy import immutable


@immutable
class PennylaneCircuitWrapper(CircuitWrapper[qml.tape.QuantumTape]):  # pylint: disable=too-few-public-methods
    """A class to wrap a Pennylane tape."""

    @classmethod
    def is_valid(cls, circuit: qml.tape.QuantumTape) -> bool:
        """Check if the circuit is valid.

        Parameters
        ----------
        circuit : CT
            The circuit.

        Returns
        -------
        bool
            True if the circuit is valid, False otherwise.
        """
        try:
            cls._to_openqasm2(circuit)
            return True
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        return False

    @staticmethod
    def _to_openqasm2(circuit: qml.tape.QuantumTape) -> str:
        """Return the OpenQASM2 circuit.

        Parameters
        ----------
        circuit : qml.tape.QuantumTape
            The Pennylane tape.

        Returns
        -------
        str
            The OpenQASM2 circuit.
        """
        if not isinstance(circuit, qml.tape.QuantumTape):
            ops = list(circuit.keys())
            tape = qml.tape.QuantumTape(ops)
        else:
            tape = circuit
        return str(tape.to_openqasm())

    @staticmethod
    def _from_openqasm2(qasm: str) -> qml.tape.QuantumTape:
        """Return the Pennylane tape.

        Parameters
        ----------
        qasm : str
            The OpenQASM2 circuit.

        Returns
        -------
        qml.tape.QuantumTape
            The Pennylane tape.
        """
        with qml.tape.QuantumTape() as tape:
            qml.from_qasm(qasm)()
        return tape
