"""A wrapper for IonQ circuits."""

from typing import Any, Dict

from qiskit import QuantumCircuit, qasm2  # type: ignore
from qiskit.circuit.library import (  # type: ignore
    CCXGate,
    CHGate,
    CPhaseGate,
    CRXGate,
    CRYGate,
    CRZGate,
    CXGate,
    CYGate,
    CZGate,
    HGate,
    IGate,
    MCPhaseGate,
    MCXGate,
    MCXGrayCode,
    PhaseGate,
    RXGate,
    RXXGate,
    RYGate,
    RYYGate,
    RZGate,
    RZZGate,
    SdgGate,
    SGate,
    SwapGate,
    SXdgGate,
    SXGate,
    TdgGate,
    TGate,
    XGate,
    YGate,
    ZGate,
)
from qiskit_ionq import GPI2Gate, GPIGate, MSGate  # type: ignore
from qiskit_ionq.helpers import qiskit_circ_to_ionq_circ  # type: ignore

from qukit.circuits.circuit_wrapper import CircuitWrapper
from qukit.circuits.wrappers.qiskit_wrapper import QiskitCircuitWrapper
from qukit.utils.immutabiliy import immutable


# TODO: prototype code, needs to be tested # pylint: disable=fixme
def ionq_to_qiskit(  # pylint: disable=too-many-branches
    ionq_circuit: list[dict[str, Any]]
) -> QuantumCircuit:  # typed -- GB # pragma: no cover
    """Convert an IonQ circuit to a Qiskit circuit.

    Kindly provided by IonQ Support.

    Parameters
    ----------
    ionq_circuit : list[Any]
        The IonQ circuit.

    Returns
    -------
    QuantumCircuit:
        The converted Qiskit quantum circuit.
    """

    # Calculate the number of qubits
    max_qubit_index = 0
    for gate in ionq_circuit:
        targets = list(gate.get("targets", []))
        controls = list(gate.get("controls", []))
        max_qubit_index = max(max_qubit_index, *targets, *controls)
    num_qubits = max_qubit_index + 1

    qiskit_circuit = QuantumCircuit(num_qubits)

    gate_map = {
        "ccx": CCXGate,
        "ch": CHGate,
        "cnot": CXGate,
        "cp": CPhaseGate,
        "crx": CRXGate,
        "cry": CRYGate,
        "crz": CRZGate,
        "csx": CPhaseGate,
        "cx": CXGate,
        "cy": CYGate,
        "cz": CZGate,
        "h": HGate,
        "i": IGate,
        "id": IGate,
        "mcp": MCPhaseGate,
        "mcphase": MCPhaseGate,
        "mct": MCXGate,
        "mcx": MCXGate,
        "mcx_gray": MCXGrayCode,
        "p": PhaseGate,
        "rx": RXGate,
        "rxx": RXXGate,
        "ry": RYGate,
        "ryy": RYYGate,
        "rz": RZGate,
        "rzz": RZZGate,
        "s": SGate,
        "sdg": SdgGate,
        "swap": SwapGate,
        "sx": SXGate,
        "sxdg": SXdgGate,
        "t": TGate,
        "tdg": TdgGate,
        "toffoli": CCXGate,
        "x": XGate,
        "y": YGate,
        "z": ZGate,
    }

    alias_map = {
        "cp": "cz",
        "csx": "cv",
        "mcphase": "cz",
        "ccx": "cx",  # just one C for all mcx
        "mcx": "cx",  # just one C for all mcx
        "mcx_gray": "cx",  # just one C for all mcx
        "tdg": "ti",
        "p": "z",
        "rxx": "xx",
        "ryy": "yy",
        "rzz": "zz",
        "sdg": "si",
        "sx": "v",
        "sxdg": "vi",
    }

    for gate in ionq_circuit:
        name = gate["gate"]
        targets = list(gate.get("targets", []))
        controls = list(gate.get("controls", []))
        rotation = gate.get("rotation", 0)
        phases = list(gate.get("phases", []))  # Added support for IonQ Native Gates -- GB

        # Apply alias mapping if present
        name = alias_map.get(name, name)

        if name in gate_map:
            qiskit_gate = gate_map[name]
            if name in ["rx", "ry", "rz", "p", "crx", "cry", "crz"]:
                qiskit_circuit.append(qiskit_gate(rotation), targets)
            elif len(controls) > 0:
                if len(controls) == 1:
                    if name in list(
                        g[1:] for g in gate_map if g.startswith("c")
                    ):  # check if the gate is a controlled gate -- GB
                        qiskit_gate = gate_map["c" + name]
                    qiskit_circuit.append(qiskit_gate(), [controls[0]] + targets)
                else:
                    # Handling multi-control gates
                    qc_mcx = MCXGate(len(controls))
                    qiskit_circuit.append(qc_mcx, controls + targets)

            else:
                qiskit_circuit.append(qiskit_gate(), targets)
        # Added support for IonQ Native Gates -- GB
        elif name == "ms":
            qiskit_circuit.append(MSGate(phases[0], phases[1], rotation), targets)
        elif name == "gpi":
            qiskit_circuit.append(GPIGate(phases[0]), targets)
        elif name == "gpi2":
            qiskit_circuit.append(GPI2Gate(phases[0]), targets)
        else:
            raise ValueError(f"Unsupported gate: {name}")

    return qiskit_circuit


@immutable
class IonQCircuitWrapper(CircuitWrapper[dict]):  # type: ignore
    """A class to wrap an IonQ Circuit."""

    @staticmethod
    def _to_openqasm2(circuit: Dict[str, Any], measure: bool = True) -> str:
        """Return the OpenQASM2 circuit.

        Parameters
        ----------
        circuit : Dict[str, Any]
            The IonQ circuit.
        measure : bool, optional
            If True, measure all qubits at the end of the circuit.

        Returns
        -------
        str
            The OpenQASM2 circuit.
        """
        qiskit_circuit = ionq_to_qiskit(list(circuit["circuit"]))
        if measure:
            qiskit_circuit.measure_all()
        return str(qasm2.dumps(qiskit_circuit)).replace("meas[", "c[")

    @staticmethod
    def _from_openqasm2(qasm: str) -> Dict[str, Any]:
        """Return the IonQ circuit.

        Parameters
        ----------
        qasm : str
            The OpenQASM2 circuit.

        Returns
        -------
        Circuit
            The IonQ circuit.
        """
        qiskit_circuit = QiskitCircuitWrapper.from_openqasm2(qasm)
        ionq_circuit = {
            "circuit": qiskit_circ_to_ionq_circ(qiskit_circuit.circuit, gateset="qis")[
                0
            ]  # TODO: gateset # pylint: disable=fixme
        }
        ionq_circuit["qubits"] = len(qiskit_circuit.circuit.qubits)
        return ionq_circuit
