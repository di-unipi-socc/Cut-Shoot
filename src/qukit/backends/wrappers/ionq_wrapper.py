"""The IonQ Provider and Backend Wrappers."""

import json
import os
import random
from copy import deepcopy
from typing import Any, Optional

import requests

from qukit.backends.backend_wrapper import BackendStatus, BackendWrapper
from qukit.backends.job import Job, JobStatus
from qukit.backends.provider_wrapper import ProviderWrapper
from qukit.backends.result import Result
from qukit.circuits.wrappers import IonQCircuitWrapper
from qukit.utils.immutabiliy import immutable, initbyvalue, passbyvalue
from qukit.utils.parallel import ThreadWithReturnValue as ThreadRV


class IonQHTTPInterface:
    """The IonQ HTTP interface.

    Parameters
    ----------
    **kwargs : Optional[str]
            The configuration parameters,
            currently only the
            API key is supported: api_key: str
            (if not specified,
            the environment variable
            IONQ_API_KEY is used).
    """

    _ENV_VARS = {
        "api_key": "IONQ_API_KEY",
    }

    def __init__(self, **kwargs: Optional[str]):
        """Initialize the IonQ HTTP interface.

        Parameters
        ----------
        **kwargs : Optional[str]
            The configuration parameters,
            currently only the
            API key is supported: api_key: str
            (if not specified,
            the environment variable
            IONQ_API_KEY is used).
        """
        self._interface = "https://api.ionq.co"

        for key, env_var in self._ENV_VARS.items():
            setattr(self, f"_{key}", os.environ.get(env_var))

        for key, value in kwargs.items():
            setattr(self, f"_{key}", value)

        if not hasattr(self, "_api_key"):  # pragma: no cover
            self._api_key = None

        if self._api_key is None:
            raise ValueError("API key not provided for IonQ Cloud")

    def get_backends(
        self, characterization: bool = True, timeout: Optional[float] = None
    ) -> list[dict[str, Any]]:
        """Return the backends.

        Parameters
        ----------
        characterization : bool, optional
            If True, return also the characterization of the backends, by default True.
        timeout : Optional[float], optional
            The timeout for the request, by default None.

        Returns
        -------
        list[dict[str, Any]]
            The backends.

        Raises
        ------
        RuntimeError
            If the request fails.
        """
        url = self._interface + "/v0.3/backends"
        headers = {"Authorization": f"apiKey {self._api_key}"}
        res = requests.get(url, headers=headers, timeout=timeout)
        if res.status_code == 200:
            backends = res.json()
            if characterization:
                for i in range(len(backends.copy())):
                    if "characterization_url" in backends[i]:
                        backends[i]["characterization"] = self.get_characterization(
                            backends[i]["characterization_url"],
                            timeout=timeout,
                        )
            return backends  # type: ignore
        raise RuntimeError(f"Error {res.status_code} getting backends: {str(res.json())}")  # pragma: no cover

    def get_characterization(self, url: str, timeout: Optional[float] = None) -> dict[str, Any]:
        """Return the characterization.

        Parameters
        ----------
        url : str
            The URL of the characterization.
        timeout : Optional[float], optional
            The timeout for the request, by default None.

        Returns
        -------
        dict[str, Any]
            The characterization.

        Raises
        ------
        RuntimeError
            If the request fails.
        """

        url = self._interface + "/v0.3" + url
        headers = {"Authorization": f"apiKey {self._api_key}"}
        res = requests.get(url, headers=headers, timeout=timeout)
        if res.status_code == 200:
            return res.json()  # type: ignore
        raise RuntimeError(
            f"Error {res.status_code} getting characterization: {str(res.json())}"
        )  # pragma: no cover

    def run_job(self, request: dict[str, Any], timeout: Optional[float] = None) -> dict[str, Any]:
        """Run an IonQ job.

        Raises
        ------
        RuntimeError
            If the request fails.

        Parameters
        ----------
        request : dict[str, Any]
            The request to run the job.
        timeout : Optional[float]
            The timeout for the request.

        Returns
        -------
        dict[str, Any]
            The IonQ job.
        """
        url = self._interface + "/v0.3/jobs"
        headers = {
            "Authorization": f"apiKey {self._api_key}",
            "Content-Type": "application/json",
        }
        res = requests.post(url, json=request, headers=headers, timeout=timeout)
        if res.status_code == 200:
            return res.json()  # type: ignore
        raise RuntimeError(f"Error {res.status_code} running job: {str(res.json())}")  # pragma: no cover

    def get_job(self, job_id: str, timeout: Optional[float] = None) -> dict[str, Any]:
        """Get the IonQ job.

        Raises
        ------
        RuntimeError
            If the request fails.

        Parameters
        ----------
        job_id : str
            The id of the job.
        timeout : Optional[float]
            The timeout for the request.

        Returns
        -------
        dict[str, Any]
            The IonQ job.
        """
        url = self._interface + f"/v0.3/jobs/{job_id}"
        headers = {"Authorization": f"apiKey {self._api_key}"}
        res = requests.get(url, headers=headers, timeout=timeout)
        if res.status_code == 200:
            return res.json()  # type: ignore
        raise RuntimeError(f"Error {res.status_code} getting job: {str(res.json())}")  # pragma: no cover

    def get_results(
        self,
        results_url: str,
        sharpening: bool,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        """Get the IonQ job results.

        Raises
        ------
        RuntimeError
            If the request fails.

        Parameters
        ----------
        results_url : str
            The URL of the results.
        sharpening : bool
            Whether to apply sharpening.
        timeout : Optional[float]
            The timeout for the request.

        Returns
        -------
        dict[str, Any]
            The IonQ job results.
        """
        url = self._interface + results_url
        headers = {"Authorization": f"apiKey {self._api_key}"}
        res = requests.get(
            url,
            headers=headers,
            timeout=timeout,
            params={"sharpening": sharpening},
        )
        if res.status_code == 200:
            return res.json()  # type: ignore
        raise RuntimeError(f"Error {res.status_code} getting results: {str(res.json())}")  # pragma: no cover


@initbyvalue
class IonQBackendWrapper(BackendWrapper[IonQCircuitWrapper]):  # pylint: disable=too-many-instance-attributes
    """A wrapper for IonQ backends.

    Currently, the only configuration parameters are:
    - noise_model: str
    - noise_seed: int
    - api_key: str (if not specified, the environment variable IONQ_API_KEY is used)

    wait_for_job andd job_finished accept as additional keyword arguments:
    - timeout: Optional[float]

    wait_for_job accepts as additional keyword arguments:
    - sharpening: bool

    Parameters
    ----------
    name : str
        The name of the backend.
    config : Optional[dict[str, Any]], optional
        The configuration of the backend, by default None.
    device : Optional[str], optional
        The device name, by default is the same as the name.
    **kwargs : Any
        Other parameters, currently not used.
    """

    def __init__(
        self,
        name: str,
        config: Optional[dict[str, Any]] = None,
        *,
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize the wrapper.

        Currently, the only configuration parameters are:
        - noise_model: str
        - noise_seed: int

        Parameters
        ----------
        name : str
            The name of the backend.
        config : Optional[dict[str, Any]], optional
            The configuration of the backend, by default None.
        device : Optional[str], optional
            The device name, by default is the same as the name.
        **kwargs : Any
            Other parameters, currently not used.

        Raises
        ------
        ValueError
            If the backend is not found.
        """
        self._interface = IonQHTTPInterface(**kwargs)
        self._noise_model = config.get("noise_model", None) if config is not None else None
        self._noise_seed = config.get("noise_seed", None) if config is not None else None
        self._noise_models = None
        super().__init__(name, config, device=device)

        # queue time, last_update, degraded #TODO: add? # pylint: disable=fixme

    def info(self) -> str:
        """Return the information of the backend.

        Returns
        -------
        str
            The information of the backend.
        """

        s = super().info()
        s += f"\tNoise Model: {self._noise_model}\n"
        try:
            s += f"\tAvailable Noise Models: {self.noise_models}\n"
        except ValueError:
            pass
        return s

    def update(self) -> None:  # pylint: disable=too-many-branches
        """Update the backend information.

        Raises
        ------
        ValueError
            If the backend is not found.

        ValueError
            If the noise model is not found.
        """
        backends_info = self._interface.get_backends()
        try:
            self._backend_info = next(
                backend_info for backend_info in backends_info if backend_info["backend"] == self._device
            )
        except StopIteration as exc:
            raise ValueError(f"Backend {self._device} not found in IonQ backends") from exc

        try:
            self._noise_models = self._backend_info["noise_models"]
        except KeyError:
            self._noise_models = None

        noise_model_backend_info = None
        if self._noise_model is not None:
            try:
                noise_model_backend_info = next(
                    backend_info
                    for backend_info in backends_info
                    if backend_info["backend"] == "qpu." + str(self._noise_model)
                )
            except StopIteration as exc:
                raise ValueError(
                    f"Noise model backend {self._noise_model} not found in IonQ backends"
                ) from exc

        if noise_model_backend_info is not None:
            self._qubits = int(noise_model_backend_info["qubits"])
        else:
            self._qubits = int(self._backend_info["qubits"])

        if (
            self._backend_info["status"] == "running" or self._backend_info["status"] == "available"
        ):  # pragma: no cover
            if self._backend_info["has_access"]:  # pragma: no cover
                self._status = BackendStatus.AVAILABLE
            else:
                self._status = BackendStatus.ONLINE
        elif (
            self._backend_info["status"] == "reserved" or self._backend_info["status"] == "unavailable"
        ):  # pragma: no cover
            self._status = BackendStatus.UNAVAILABLE
        elif self._backend_info["status"] == "offline":  # pragma: no cover
            self._status = BackendStatus.OFFLINE
        elif self._backend_info["status"] == "calibrating":  # pragma: no cover
            self._status = BackendStatus.MAINTENANCE
        else:  # pragma: no cover
            self._status = BackendStatus.UNKNOWN

        self._characterization = self._backend_info.get("characterization", None)

    @property
    def noise_models(self) -> list[str]:
        """Return the noise models of the backend.

        Returns
        -------
        list[str]
            The noise models of the backend.
        """
        if self._noise_models is None:
            raise ValueError(f"Noise models not found for backend {self._device}")
        return self._noise_models

    @property
    def status(self) -> BackendStatus:
        """Return the status of the backend.

        Returns
        -------
        BackendStatus
            The status of the backend.
        """
        return self._status

    @property
    def qubits(self) -> int:
        """Return the number of qubits of the backend.

        If a noise model is used, the number of qubits of the noise model is returned.

        Returns
        -------
        int
            The number of qubits of the backend.
        """
        return self._qubits

    @property
    def simulator(self) -> bool:
        """Return True if the backend is a simulator.

        Returns
        -------
        bool
            True if the backend is a simulator.
        """
        return "simulator" in self._device

    @property
    def noisy(self) -> bool:
        """Return True if the backend is noisy.

        Returns
        -------
        bool
            True if the backend is noisy.
        """
        return (self._noise_model) is not None or (not self.simulator)

    @passbyvalue
    def _run(  # pylint: disable=too-many-arguments too-many-locals
        self,
        circuits: list[IonQCircuitWrapper],
        shots: int = 1024,
        asynch: bool = False,
        metadata: Optional[dict[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
        seed: Optional[int] = None,
        debiasing: bool = False,
        **kwargs: Any,
    ) -> Job:
        """Run the circuits on the backend.

        Parameters
        ----------
        circuits : IonQCircuitWrapper
            The circuits to run.
        shots : int, optional
            The number of shots, by default 1024.
        asynch : bool, optional
            If True, run the circuits asynchronously, by default False.
        metadata : dict[str, Any], optional
            The metadata of the job, by default None.
        timeout : float, optional
            The timeout, by default None.
        seed : int, optional
            The seed for the noise model, by default None.
            If not specified, the seed specified in the backend initialization is used.
        debiasing : bool, optional
            If True, apply debiasing, by default False.
        **kwargs : Any
            Other parameters, currently not used.

        Returns
        -------
        Job
            The job.
        """

        gateset = "qis"
        native_gateset = ["ms", "gpi", "gpi2", "zz"]

        gateset = "qis"
        # just checking the first one because IonQ does not support mixing gatesets
        for gate in circuits[0].circuit["circuit"]:
            if gate["gate"] in native_gateset:
                gateset = "native"
                break

        body = {
            "target": self._device,
            "shots": shots,
            "input": {
                "format": "ionq.circuit.v0",
                "gateset": gateset,
                "circuits": [c.circuit for c in circuits],
            },
            "error_mitigation": {"debias": debiasing},
        }

        metadata = metadata or {}
        metadata["gateset"] = gateset
        metadata["debiasing"] = debiasing

        if metadata is not None:
            body["metadata"] = metadata
            if "name" in metadata:
                body["name"] = metadata["name"]
        if self._noise_model is not None:
            body["noise"] = {
                "model": self._noise_model,
            }
            if seed is not None:
                body["noise"]["seed"] = seed  # type: ignore # idk why this is an error
            elif self._noise_seed is not None:
                body["noise"]["seed"] = self._noise_seed  # type: ignore

        body["metadata"] = {
            "request": json.dumps(deepcopy(body)),
            "metadata": json.dumps(metadata),
        }

        res = self._interface.run_job(body, timeout)

        job_id = res["id"]
        job_data = self._interface.get_job(job_id, timeout)

        # dumps and then loads to remove frozendict.forzendict
        job_metadata = {
            "request": json.loads(json.dumps(body)),
            "response": job_data,
            "metadata": json.loads(json.dumps(metadata)),
        }
        job = Job(job_id, self, circuits, shots, job_metadata)

        if not asynch:
            self.wait_for_job(job, timeout=timeout)
        return job

    def job_info(self, job: Job, timeout: Optional[float] = None) -> dict[str, Any]:
        """Return the information of the job.

        Parameters
        ----------
        job : Job
            The Job.
        timeout : Optional[float]
            The timeout for the request.

        Returns
        -------
        dict[str, Any]
            The information of the job.
        """
        return self._interface.get_job(job.id, timeout)

    def job_status(  # pylint: disable=too-many-return-statements
        self, job: Job, timeout: Optional[float] = None
    ) -> JobStatus:
        """Return the status of the job.

        Parameters
        ----------
        job : Job
            The job.
        timeout : Optional[float]
            The timeout for the request.

        Returns
        -------
        BackendStatus
            The status of the job.
        """
        job_data = self.job_info(job, timeout)
        if job_data is None:  # pragma: no cover
            return JobStatus.UNKNOWN
        if job_data["status"] == "completed":  # pragma: no cover
            return JobStatus.COMPLETED
        if job_data["status"] == "failed":  # pragma: no cover
            return JobStatus.FAILED
        if job_data["status"] == "canceled":  # pragma: no cover
            return JobStatus.CANCELED
        if job_data["status"] == "running":  # pragma: no cover
            return JobStatus.RUNNING
        if job_data["status"] == "ready" or job_data["status"] == "submitted":  # pragma: no cover
            return JobStatus.QUEUED
        return JobStatus.UNKNOWN  # pragma: no cover

    def get_job(self, job_id: str, timeout: Optional[float] = None) -> Optional[Job]:
        """Return the job with the given id.

        Parameters
        ----------
        job_id : str
            The id of the job.
        timeout : Optional[float]
            The timeout for the request.

        Returns
        -------
        Optional[Job]
            The job with the given id.
        """
        try:
            job_data = self._interface.get_job(job_id, timeout)
        except RuntimeError:
            return None

        try:
            job_data["metadata"]["request"] = json.loads(job_data["metadata"]["request"])
            job_data["metadata"]["metadata"] = json.loads(job_data["metadata"]["metadata"])
            return Job(
                job_id,
                self,
                job_data["metadata"]["metadata"]["circuits"],
                int(job_data["metadata"]["metadata"]["shots"]),
                {
                    "response": job_data,
                    "metadata": job_data["metadata"]["metadata"],
                    "request": job_data["metadata"]["request"],
                },
            )
        except KeyError:
            return Job(job_id, self, [], 0, {"response": job_data, "metadata": {}})

    def job_results(  # pylint: disable=too-many-locals
        self,
        job: Job,
        sharpening: bool = False,
        timeout: Optional[float] = None,
    ) -> Optional[list[Result]]:
        """Return the results of the job.

        Parameters
        ----------
        job : Job
            The job.
        sharpening : bool, optional
            Whether to apply sharpening, by default False.
        timeout : Optional[float]
            The timeout for the request.

        Returns
        -------
        list[Result]
            The results of the job.
        """
        if not self.job_finished(job, timeout=timeout):  # pragma: no cover
            return None

        job_data = self._interface.get_job(job.id, timeout=timeout)

        job_metadata = job.get_metadata(verbose=True)
        job_metadata["response"] = job_data

        job._metadata = job_metadata  # pylint: disable=protected-access

        if job.status == JobStatus.FAILED:
            job._error = job_data["failure"]["error"]  # pylint: disable=protected-access
            raise RuntimeError(f"Job {job.id} failed: {job.get_metadata(verbose=True)['response']}")

        if job.status == JobStatus.CANCELED:  # pragma: no cover
            raise RuntimeError(f"Job {job.id} was canceled")

        results_data = self._interface.get_results(job_data["results_url"], sharpening, timeout)

        children = job_data["children"]
        circuits = job.circuits

        results = []
        if len(circuits) == 1:

            data = {
                bin(int(k))[2:].zfill(circuits[0].circuit["qubits"]): int(v * job.shots)
                for k, v in results_data.items()
            }

            if sum(data.values()) != job.shots:
                keys = list(data.keys())
                diff = job.shots - sum(data.values())
                random_subset = random.sample(keys, diff)
                for key in random_subset:
                    data[key] += 1

            results.append(
                Result(
                    job=job,
                    circuit=circuits[0],
                    counts={"c": data},
                )
            )
        else:
            for i, circuit in enumerate(circuits):
                child = children[i]
                _counts = results_data[child]
                data = {
                    bin(int(k))[2:].zfill(circuit.circuit["qubits"]): int(v * job.shots)
                    for k, v in _counts.items()
                }
                if sum(data.values()) != job.shots:
                    keys = list(data.keys())
                    diff = job.shots - sum(data.values())
                    random_subset = random.sample(keys, diff)
                    for key in random_subset:
                        data[key] += 1
                counts = data
                results.append(Result(job=job, circuit=circuit, counts={"c": counts}))

        return results


@immutable
class IonQProviderWrapper(ProviderWrapper[IonQBackendWrapper]):
    """A wrapper for IonQ provider.

    Parameters
    ----------
    **kwargs : Optional[str]
            The configuration parameters,
            currently only the
            API key is supported: api_key: str
            (if not specified,
            the environment variable
            IONQ_API_KEY is used).
    """

    def __init__(self, **kwargs: Any):
        """Initialize the class.

        Parameters
        ----------
        **kwargs : Optional[str]
            The configuration parameters,
            currently only the
            API key is supported: api_key: str
            (if not specified,
            the environment variable
            IONQ_API_KEY is used).
        """
        self._interface = IonQHTTPInterface(**kwargs)
        self._kwargs = kwargs

    def _wrap_backend(self, backend_name: str) -> list[IonQBackendWrapper]:
        """Wrap the backend.

        Parameters
        ----------
        backend_name : str
            The name of the backend.

        Returns
        -------
        list[IonQBackendWrapper]
            The wrapped backend.
        """
        backends = []
        backend = IonQBackendWrapper(backend_name, **self._kwargs)
        if backend.simulator:
            backends.append(backend)
            noise_models = backend.noise_models
            for noise_model in noise_models:
                if noise_model == "ideal":
                    continue
                backends.append(
                    IonQBackendWrapper(
                        name=f"{backend.device}_{noise_model}",
                        config={"noise_model": noise_model},
                        device=backend.device,
                        **self._kwargs,
                    )
                )
        else:
            backends.append(backend)

        return backends

    def _backends(self, timeout: Optional[float] = None) -> list[IonQBackendWrapper]:
        """Return the backends.

        Parameters
        ----------
        timeout : Optional[float], optional
            The timeout for the request, by default None.

        Returns
        -------
        list[IonQBackendWrapper]
            The backends.
        """
        threads = []
        for backend_info in self._interface.get_backends(timeout=timeout, characterization=False):
            thread = ThreadRV(target=self._wrap_backend, args=(backend_info["backend"],))
            thread.start()
            threads.append(thread)

        backends = []
        for thread in threads:
            backend = thread.join()
            if backend is None and thread.exception is not None:
                raise thread.exception
            backends.extend(backend)

        return backends
