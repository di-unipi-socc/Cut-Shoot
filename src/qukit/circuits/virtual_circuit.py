"""A class to virtualize the circuit wrappers."""

from typing import Any, Optional

from qukit.circuits.circuit_wrapper import CircuitWrapper
from qukit.utils.immutabiliy import immutable
from qukit.utils.utils import get_imported_classes


@immutable
class VirtualCircuit(CircuitWrapper[Any]):
    """A class to virtualize the circuit wrappers.

    Parameters
    ----------
    circuit : Any
        The circuit.
    metadata : dict, optional
        The metadata.
    """

    def __init__(self, circuit: Any, metadata: Optional[dict[str, Any]] = None):
        """Initialize the class.

        Parameters
        ----------
        circuit : Any
            The circuit.
        metadata : dict, optional
            The metadata.
        """

        metadata = metadata or {}
        if isinstance(circuit, CircuitWrapper):
            metadata = {**circuit.metadata, **metadata}
            circuit = circuit.circuit

        assert self.is_valid(circuit), f"Invalid circuit: {circuit}"
        super().__init__(circuit, metadata)

    @staticmethod
    def available_languages() -> list[str]:
        """Return the available languages.

        Returns
        -------
        list[str]
            The available languages.
        """
        wrappers = CircuitWrapper.__subclasses__()
        return [
            wrapper.language()
            for wrapper in wrappers
            if wrapper.__name__ != VirtualCircuit.__name__
            and wrapper.__name__ in get_imported_classes(CircuitWrapper)
        ]

    @staticmethod
    def get_wrapper(language: str) -> Any:
        """Return the wrapper for the language.

        Parameters
        ----------
        language : str
            The language.

        Returns
        -------
        Any
            The wrapper for the language.

        Raises
        ------
        ValueError
            If there is no wrapper for the language.
        """
        wrappers = CircuitWrapper.__subclasses__()
        for wrapper in wrappers:
            if (
                wrapper.language() == language
                and wrapper is not VirtualCircuit
                and wrapper.__name__ in get_imported_classes(CircuitWrapper)
            ):
                return wrapper
        raise ValueError("Invalid language")

    @staticmethod
    def which_wrapper(circuit: Any) -> Optional[Any]:
        """Return the wrapper for the circuit.

        Parameters
        ----------
        circuit : Any
            The circuit.

        Returns
        -------
        Optional[Any]
            The wrapper for the circuit.
        """
        wrappers = CircuitWrapper.__subclasses__()
        for wrapper in wrappers:
            if wrapper is not VirtualCircuit and wrapper.__name__ in get_imported_classes(CircuitWrapper):
                if wrapper.is_valid(circuit):
                    return wrapper
        return None

    @staticmethod
    def which_language(circuit: Any) -> Optional[str]:
        """Return the language of the circuit.

        Parameters
        ----------
        circuit : Any
            The circuit.

        Returns
        -------
        Optional[str]
            The language of the circuit.
        """
        wrapper = VirtualCircuit.which_wrapper(circuit)
        if wrapper is not None:
            return str(wrapper.language())
        return None

    @staticmethod
    def wrap(circuit: Any) -> Any:
        """Wrap the circuit.

        Parameters
        ----------
        circuit : Any
            The circuit.

        Returns
        -------
        Any
            The wrapped circuit.

        Raises
        ------
        ValueError
            If there is no wrapper for the circuit.
        """
        wrapper = VirtualCircuit.which_wrapper(circuit)
        if wrapper is not None:
            return wrapper(circuit)
        raise ValueError("Invalid circuit")

    @property
    def wrapper(self) -> Any:
        """Return the wrapper of the circuit.

        Returns
        -------
        Any
            The wrapper of the circuit.
        """
        return self.which_wrapper(self.circuit)

    @property
    def circuit_language(self) -> str:
        """Return the language of the circuit.

        Returns
        -------
        str
            The language of the circuit.
        """
        # existence of wrapper enforced during __init__
        return self.which_language(self.circuit)  # type: ignore

    @staticmethod
    def _to_openqasm2(circuit: Any) -> str:
        """Return the circuit in OpenQASM 2.0 format.

        Parameters
        ----------
        circuit : Any
            The circuit.

        Returns
        -------
        str
            The circuit in OpenQASM 2.0 format.
        """
        return str((VirtualCircuit.wrap(circuit)).to_openqasm2().circuit)  # pylint: disable=not-callable

    @staticmethod
    def _from_openqasm2(qasm: str) -> Any:
        """Return the circuit virtualised from OpenQASM 2.0 format.

        Parameters
        ----------
        qasm : str
            The circuit in OpenQASM 2.0 format.

        Returns
        -------
        Any
            The circuit virtualised from OpenQASM 2.0 format.
        """
        return VirtualCircuit(qasm).circuit

    def translate(self, language: str) -> Any:
        """Translate the circuit to the language.

        Parameters
        ----------
        language : str
            The language.

        Returns
        -------
        Any
            The circuit translated to the language.
        """
        wrapper = VirtualCircuit.get_wrapper(language)
        qasm = self.to_openqasm2()
        return VirtualCircuit(wrapper.from_openqasm2(qasm), self.metadata)

    @staticmethod
    def is_valid(circuit: Any) -> bool:
        """Return whether the circuit is valid.

        Parameters
        ----------
        circuit : Any
            The circuit.

        Returns
        -------
        bool
            Whether the circuit is valid.
        """

        for wrapper in CircuitWrapper.__subclasses__():
            if wrapper is not VirtualCircuit and wrapper.__name__ in get_imported_classes(CircuitWrapper):
                if wrapper.is_valid(circuit):
                    return True
        return False
