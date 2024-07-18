"""Parallelisation utilities."""

import threading
from typing import Any, Optional


class ThreadWithReturnValue(threading.Thread):
    """Thread class with a return value.

    This class is a subclass of threading.Thread that allows the target function to return a value.

    Parameters
        ----------
        group : None
            The group.
        target : None
            The target function.
        name : None
            The name.
        args : tuple
            The arguments.
        kwargs : dict
            The keyword arguments.
        daemon : None
            The daemon.
        throw_exc : bool
            Whether to throw exceptions.
    """

    maximumNumberOfRuningThreads: Optional[int] = None

    def __init__(  # type: ignore  # pylint: disable=too-many-arguments
        self, group=None, target=None, name=None, args=(), kwargs=None, daemon=None, throw_exc=True
    ) -> None:
        """Initialize the thread.

        Parameters
        ----------
        group : None
            The group.
        target : None
            The target function.
        name : None
            The name.
        args : tuple
            The arguments.
        kwargs : dict
            The keyword arguments.
        daemon : None
            The daemon.
        throw_exc : bool
            Whether to throw exceptions.
        """

        if kwargs is None:
            kwargs = {}
        self._target = None

        threading.Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)
        self._return = None
        self._throw_exc = throw_exc
        self._exc = None

        self._thread_limiter: Optional[threading.Semaphore] = None
        if ThreadWithReturnValue.maximumNumberOfRuningThreads is not None:
            self._thread_limiter = threading.Semaphore(ThreadWithReturnValue.maximumNumberOfRuningThreads)

    def run(self) -> None:
        """Run the target function."""
        if self._target is not None:

            if self._thread_limiter is not None:
                self._thread_limiter.acquire()  # pylint: disable=consider-using-with

            try:
                self._return = self._target(*self._args, **self._kwargs)
            except Exception as e:  # pylint: disable=broad-except
                self._exc = e

            if self._thread_limiter is not None:
                self._thread_limiter.release()

            if self._throw_exc and self._exc is not None:
                raise self._exc

    def join(self, *args: Any) -> Any:
        """Join the thread.

        Parameters
        ----------
        *args : Any
            The arguments.

        Returns
        -------
        Any
            The return value.
        """
        threading.Thread.join(self, *args)
        return self._return

    @property
    def result(self) -> Any:
        """Return the result.

        Returns
        -------
        Any
            The return value.
        """
        return self._return

    @property
    def exception(self) -> Optional[Exception]:
        """Return the exception.

        Returns
        -------
        Optional[Exception]
            The exception.
        """
        return self._exc
