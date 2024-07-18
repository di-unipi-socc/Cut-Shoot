"""A wrapper for Braket circuits."""

import braket.circuits.circuit  # type: ignore
from cirq.contrib.qasm_import import circuit_from_qasm  # type: ignore
from qbraid.transpiler import transpile  # type: ignore
from qiskit import qasm2  # type: ignore

from qukit.circuits.circuit_wrapper import CircuitWrapper
from qukit.utils.immutabiliy import immutable


@immutable
class BraketCircuitWrapper(
    CircuitWrapper[braket.circuits.circuit.Circuit]
):  # pylint: disable=too-few-public-methods
    """A class to wrap a Braket circuits."""

    @staticmethod
    def _to_openqasm2(circuit: braket.circuits.circuit.Circuit, measure: bool = True) -> str:
        """Return the OpenQASM2 circuit.

        Parameters
        ----------
        circuit : braket.circuits.circuit.Circuit
            The Braket circuit.
        measure : bool, optional
            If True, measure all qubits at the end of the circuit.

        Returns
        -------
        str
            The OpenQASM2 circuit.
        """
        qiskit_circuit = transpile(circuit, "qiskit")
        if measure:
            qiskit_circuit.measure_all()
        return str(qasm2.dumps(qiskit_circuit)).replace("meas[", "c[")

    @staticmethod
    def _from_openqasm2(qasm: str) -> braket.circuits.circuit.Circuit:
        """Return the Braket circuit.

        Parameters
        ----------
        qasm : str
            The OpenQASM2 circuit.

        Returns
        -------
        braket.circuits.circuit.Circuit
            The Braket circuit.
        """
        qasm_no_barriers = "\n".join(line for line in qasm.split("\n") if "barrier" not in line)
        cirq_circuit = circuit_from_qasm(qasm_no_barriers)
        return transpile(cirq_circuit, "braket")
