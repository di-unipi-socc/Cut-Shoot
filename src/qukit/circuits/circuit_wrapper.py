"""Interface for wrapping circuits with metadata and OpenQASM2 wrapper implementation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, Self, TypeVar

from qiskit import QuantumCircuit, qasm2  # type: ignore

from qukit.utils.immutabiliy import immutable

CT = TypeVar("CT")

from threading import Lock

@immutable
class CircuitWrapper(ABC, Generic[CT]):
    """An abstract wrapper for circuits with metadata.

    Parameters
    ----------
    circuit : CT
        The circuit.
    metadata : dict, optional
        The metadata.
    """
    

    def __init__(self, circuit: CT, metadata: Optional[Dict[str, Any]] = None):
        """Initialize the class.

        Parameters
        ----------
        circuit : CT
            The circuit.
        metadata : dict, optional
            The metadata.
        """

        self._metadata = metadata or {}
        self._circuit = circuit
        assert self.is_valid(circuit), f"Invalid circuit: {circuit} for {self.__class__.__name__}"

    @classmethod
    def is_valid(cls, circuit: CT) -> bool:
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
        
        if issubclass(type(circuit), cls.circuit_type()) or cls.circuit_type() is Any:
            try:
                cls._to_openqasm2(circuit)
                return True
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(e, flush=True)
        return False

    def to_openqasm2(self) -> "OpenQASM2CircuitWrapper":
        """Return the circuit in OpenQASM2 format.

        Returns
        -------
        OpenQASM2Wrapper
            The circuit in OpenQASM2 format.
        """
        return OpenQASM2CircuitWrapper(self._to_openqasm2(self._circuit), self._metadata)

    @classmethod
    def from_openqasm2(cls, qasm: "str | OpenQASM2CircuitWrapper") -> Self:
        """Return the circuit from OpenQASM2 format.

        Parameters
        ----------
        qasm : str | OpenQASM2Wrapper
            The circuit in OpenQASM2 format.

        Returns
        -------
        Self
            The circuit.
        """
        metadata = {}
        if isinstance(qasm, OpenQASM2CircuitWrapper):
            metadata = qasm.metadata
            qasm = qasm.circuit
        return cls(cls._from_openqasm2(qasm), metadata)

    @staticmethod
    @abstractmethod
    def _to_openqasm2(circuit: CT) -> str:
        """Return the circuit in OpenQASM2 format.

        Parameters
        ----------
        circuit : CT
            The circuit.

        Returns
        -------
        str
            The circuit in OpenQASM2 format.
        """

    @staticmethod
    @abstractmethod
    def _from_openqasm2(qasm: str) -> CT:
        """Return the circuit from OpenQASM2 format.

        Parameters
        ----------
        qasm : str
            The circuit in OpenQASM2 format.

        Returns
        -------
        Self
            The circuit.
        """

    @classmethod
    def circuit_type(cls) -> type:
        """Return the type of the circuit that the wrapper wraps.

        Returns
        -------
        type
            The type of the circuit that the wrapper wraps.
        """
        return cls.__orig_bases__[0].__args__[0]  # type: ignore # pylint: disable=no-member

    @property
    def circuit(self) -> CT:
        """Return the wrapped circuit.

        Returns
        -------
        CT
            The wrapped circuit.
        """
        return self._circuit

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return the circuit metadata.

        Returns
        -------
        Dict[str, Any]
            The circuit metadata.
        """
        return self._metadata

    @classmethod
    def language(cls) -> str:
        """Return the language of the circuit.

        Returns
        -------
        str
            The language of the circuit.
        """
        return cls.__name__.lower().replace("circuitwrapper", "")

    def __str__(self) -> str:
        """Return the string representation of the class.

        Returns
        -------
        str
            The string representation of the class.
        """
        return f"{self.__class__.__name__}(circuit={self._circuit}, metadata={self._metadata})"

    def __repr__(self) -> str:
        """Return the string representation of the class.

        Returns
        -------
        str
            The string representation of the class.
        """
        return str(self)


@immutable
class OpenQASM2CircuitWrapper(CircuitWrapper[str]):
    """Wrapper for OpenQASM2 circuits."""
    
    _lock = Lock()

    @staticmethod
    def _to_openqasm2(circuit: str) -> str:
        """Return the OpenQASM2 circuit.

        Parameters
        ----------
        circuit : str
            The circuit.

        Returns
        -------
        str
            The OpenQASM2 circuit.
        """
        with OpenQASM2CircuitWrapper._lock:
            return str(qasm2.dumps(QuantumCircuit.from_qasm_str(circuit)))

    @staticmethod
    def _from_openqasm2(qasm: str) -> str:
        """Return the OpenQASM2 circuit.

        Parameters
        ----------
        qasm : str
            The OpenQASM2 circuit.

        Returns
        -------
        str
            The OpenQASM2 circuit.
        """
        
        with OpenQASM2CircuitWrapper._lock:
            return str(qasm2.dumps(QuantumCircuit.from_qasm_str(qasm)))
