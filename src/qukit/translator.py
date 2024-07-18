"""This module contains the Translator class."""

from typing import Any

from qukit.circuits import VirtualCircuit

from threading import Lock


class Translator:  # pylint: disable=too-few-public-methods
    """A class to translate circuits.

    Methods
    -------
    translate(circuit: Any, language: str) -> Any
        Translate the circuit to the specified language.
    """

    _lock = Lock()

    @staticmethod
    def languages() -> list[str]:
        """Return the available languages.

        Returns
        -------
        list[str]
            The available languages.
        """
        return VirtualCircuit.available_languages()

    @staticmethod
    def translate(circuit: Any, language: str) -> Any:
        """Translate the circuit to the specified language.

        Parameters
        ----------
        circuit : Any
            The circuit.
        language : str
            The language.

        Returns
        -------
        Any
            The translated circuit.

        Raises
        ------
        ValueError
            If the language is not supported.
        """
        
        with Translator._lock:
            vc = VirtualCircuit(circuit)
            translated = vc.translate(language)
            return translated.circuit
