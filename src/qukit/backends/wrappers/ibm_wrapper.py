"""The IBM Provider and Backend Wrappers."""

import os
import time
import warnings
from copy import deepcopy
from typing import Any, Optional

from qiskit import qasm2  # type: ignore
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # type: ignore
from qiskit_aer import AerProvider, AerSimulator  # type: ignore
from qiskit_aer.noise import NoiseModel  # type: ignore
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2  # type: ignore
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2  # type: ignore
from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2  # type: ignore

from qukit.backends.backend_wrapper import BackendStatus, BackendWrapper
from qukit.backends.job import Job, JobStatus
from qukit.backends.provider_wrapper import ProviderWrapper
from qukit.backends.result import Result
from qukit.circuits.wrappers import QiskitCircuitWrapper
from qukit.utils.immutabiliy import immutable, initbyvalue, passbyvalue
from qukit.utils.parallel import ThreadWithReturnValue as ThreadRV


@initbyvalue
class IBMBackendWrapper(BackendWrapper[QiskitCircuitWrapper]):
    """A wrapper for IBM backends.

    Currently, the only configuration parameters are:
    - channel: str
    - token: str
    - instance: str

    Alternatively, the environment
    variables QISKIT_IBM_CHANNEL,
    QISKIT_IBM_TOKEN, and
    QISKIT_IBM_INSTANCE (respectively)
    are used.
    Currently channel can be either
    "ibm_cloud" or "ibm_quantum".

    Parameters
    ----------
    name : str
        The name of the backend.
    config : Optional[dict[str, Any]], optional
        The configuration of the backend, by default None.
    device : Optional[str], optional
        The device name, by default is the same as the name.
    provider : Optional[str], optional
        The provider id of the backend, by default None.
        If not specified, the provider id is the same as the name.
    **kwargs : Any
        Other parameters, currently not used.
    """

    _ENV_VARS = {
        "channel": "QISKIT_IBM_CHANNEL",
        "token": "QISKIT_IBM_TOKEN",
        "instance": "QISKIT_IBM_INSTANCE",
    }

    _jobs: dict[str, Any] = {}
    _aer_backends = {backend.name: backend for backend in AerProvider().backends()}

    def __init__(
        self,
        name: str,
        config: Optional[dict[str, Any]] = None,
        *,
        device: Optional[str] = None,
        provider: Optional[str] = None,
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
            If the name starts with "fake.", a fake backend is used.
        config : Optional[dict[str, Any]], optional
            The configuration of the backend, by default None.
        device : Optional[str], optional
            The device name, by default is the same as the name.
        provider : Optional[str], optional
            The provider id of the backend, by default None.
            If not specified, the provider id is the same as the name.
        **kwargs : Any
            Other parameters, currently not used.

        Raises
        ------
        ValueError
            If the backend is not found.
        """

        for key, env_var in self._ENV_VARS.items():
            setattr(self, f"_{key}", os.environ.get(env_var))

        for key, value in kwargs.items():
            setattr(self, f"_{key}", value)

        if device is None:
            device = name

        if device.startswith("fake_"):
            self._backend = FakeProviderForBackendV2().backend(device)
        elif device in IBMBackendWrapper._aer_backends.keys():
            self._backend = IBMBackendWrapper._aer_backends[device]
        elif device.startswith("aer."):
            backend_name = device[4:]
            if backend_name.startswith("fake_"):
                noise_model = NoiseModel.from_backend(FakeProviderForBackendV2().backend(backend_name))
            else:
                try:
                    self._service = QiskitRuntimeService(
                        channel=self._channel,  # type: ignore # pylint: disable=no-member
                        token=self._token,  # type: ignore # pylint: disable=no-member
                        instance=self._instance,  # type: ignore # pylint: disable=no-member
                    )
                except Exception:  # pylint: disable=broad-except # pragma: no cover
                    self._service = None
                noise_model = NoiseModel.from_backend(self._service.backend(backend_name))
            self._backend = AerSimulator(noise_model=noise_model, method="automatic", **kwargs)
        else:
            try:
                    self._service = QiskitRuntimeService(
                        channel=self._channel,  # type: ignore # pylint: disable=no-member
                        token=self._token,  # type: ignore # pylint: disable=no-member
                        instance=self._instance,  # type: ignore # pylint: disable=no-member
                    )
            except Exception:  # pylint: disable=broad-except # pragma: no cover
                self._service = None
            self._backend = self._service.backend(device)

        self._sampler_options = {
            "default_shots": 1024,
            "dynamical_decoupling": {"enable": False},
            "execution": {
                "init_qubits": True,
            },
            "twirling": {"enable_gates": False, "enable_measure": False},
            "experimental": {},
            "simulator": {},
        }

        super().__init__(name, config, device=device)
        self._provider = provider

    @property
    def provider(self) -> str:
        """Return the provider of the backend.

        Returns
        -------
        str
            The provider of the backend.
        """
        if self._provider is None:
            return super().provider
        return self._provider

    @property
    def status(self) -> BackendStatus:
        """Return the status of the backend.

        Returns
        -------
        BackendStatus
            The status of the backend.
        """
        if isinstance(self._backend, (AerSimulator, FakeBackendV2)):
            return BackendStatus.AVAILABLE
        if self._backend.status().to_dict()["operational"]:  # pragma: no cover
            return BackendStatus.AVAILABLE
        return BackendStatus.UNAVAILABLE  # pragma: no cover

    @property
    def qubits(self) -> int:
        """Return the number of qubits of the backend.

        Returns
        -------
        int
            The number of qubits of the backend.
        """
        return int(self._backend.num_qubits)

    @property
    def simulator(self) -> bool:
        """Return True if the backend is a simulator.

        Returns
        -------
        bool
            True if the backend is a simulator.
        """
        if isinstance(self._backend, FakeBackendV2):
            return True
        if hasattr(self._backend, "configuration"):
            return bool(self._backend.configuration().simulator)
        return False  # pragma: no cover

    @property
    def noisy(self) -> bool:
        """Return True if the backend is noisy.

        Returns
        -------
        bool
            True if the backend is noisy.
        """
        if isinstance(self._backend, FakeBackendV2):
            return True
        if self.device.startswith("aer."):
            return True
        if hasattr(self._backend, "configuration"):
            return not self._backend.configuration().simulator
        return True  # pragma: no cover

    def update(self) -> None:
        """Update the backend information.

        In this case, it does nothing.
        """

    @passbyvalue
    def _run(  # pylint: disable=too-many-arguments too-many-locals
        self,
        circuits: list[QiskitCircuitWrapper],
        shots: int = 1024,
        asynch: bool = False,
        metadata: Optional[dict[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
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
            The timeout for waiting for the job to finish, by default None.
        seed : int, optional
            The seed for the noise model, by default None.
            If not specified, the seed specified in the backend initialization is used.
        options : dict[str, Any], optional
            The options for the sampler, by default None.
            See:
            https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/qiskit_ibm_runtime.options.SamplerOptions.
        **kwargs : Any
            Other parameters, currently not used.

        Returns
        -------
        Job
            The job.
        """

        options = self._sampler_options.copy()
        metadata = metadata or {}

        if options is not None:
            options.update(options)

        if seed is not None:
            metadata.update(seed=seed)
            options["simulator"]["seed_simulator"] = seed  # type: ignore

        sampler = SamplerV2(backend=self._backend, options=options)

        for i in range(len(metadata["circuits"])):
            metadata["circuits"][i] = qasm2.dumps(metadata["circuits"][i])

        metadata = {
            "metadata": deepcopy(metadata),
            "request": options,
        }

        pm = generate_preset_pass_manager(optimization_level=0, backend=self._backend)

        _circuits = []
        for c in circuits:
            qc = c.circuit
            qc = pm.run(qc)
            qc.metadata = metadata
            _circuits.append(qc)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            job = sampler.run(_circuits, shots=shots)

        IBMBackendWrapper._jobs[job.job_id()] = job

        if not asynch:
            if hasattr(job, "wait_for_final_state"):
                job.wait_for_final_state(timeout=timeout)
            else:
                now = time.time()
                while not job.in_final_state():
                    if timeout is not None and time.time() - now > timeout:
                        raise TimeoutError("Timeout waiting for job to finish.")  # pragma: no cover
                    time.sleep(0.1)

        job_metadata = {
            "request": options,
            "response": str(job),  # TODO: better representation? # pylint: disable=fixme
            "metadata": metadata["metadata"],
        }

        return Job(
            job_id=job.job_id(),
            backend=self,
            circuits=circuits,
            shots=shots,
            metadata=job_metadata,
        )

    def _get_job(self, job_id: str) -> Optional[Any]:
        """Return the job with the given id.

        Parameters
        ----------
        job_id : str
            The id of the job.

        Returns
        -------
        Optional[Any]
            The job with the given id.
        """
        try:
            return IBMBackendWrapper._jobs[job_id]
        except KeyError:
            try:
                return self._service.job(job_id)
            except Exception:  # pylint: disable=broad-except
                return None

    def get_job(self, job_id: str) -> Optional[Job]:
        """Return the job with the given id.

        Parameters
        ----------
        job_id : str
            The id of the job.

        Returns
        -------
        Optional[Job]
            The job with the given id.
        """

        job = self._get_job(job_id)
        if job is None:
            return None

        if hasattr(job, "inputs"):
            circuits = [QiskitCircuitWrapper(c[0]) for c in job.inputs["pubs"]]
            shots = job.inputs["pubs"][0][2]
            job_metadata = {
                "request": job.inputs["options"],
                "response": str(job),  # TODO: better representation? # pylint: disable=fixme
                "metadata": job.inputs["pubs"][0][0].metadata["metadata"],
            }
        else:
            try:
                job_metadata = job.metadata
                shots = job.metadata["metadata"]["shots"]
                circuits = [
                    QiskitCircuitWrapper(c) for c in job.metadata["metadata"]["circuits"]
                ]  # pragma: no cover
            except Exception:  # pylint: disable=broad-except
                shots = -1
                circuits = []
                job_metadata = {
                    "request": {},
                    "response": str(job),  # TODO: better representation? # pylint: disable=fixme
                    "metadata": job.metadata,
                }

        return Job(
            job_id=job.job_id(),
            backend=self,
            circuits=circuits,
            shots=-shots,
            metadata=job_metadata,
        )

    def job_status(self, job: Job) -> JobStatus:  # pylint: disable=too-many-return-statements
        """Return the status of the job.

        Parameters
        ----------
        job : Job
            The job.

        Returns
        -------
        JobStatus
            The status of the job.
        """
        status = str(self._get_job(job.id).status())  # type: ignore

        if "INITIALIZING" in status:  # pragma: no cover
            return JobStatus.QUEUED
        if "QUEUED" in status:  # pragma: no cover
            return JobStatus.QUEUED
        if "RUNNING" in status:  # pragma: no cover
            return JobStatus.RUNNING
        if "CANCELLED" in status:  # pragma: no cover
            return JobStatus.CANCELED
        if "DONE" in status:  # pragma: no cover
            return JobStatus.COMPLETED
        if "ERROR" in status:  # pragma: no cover
            return JobStatus.FAILED
        return JobStatus.UNKNOWN  # pragma: no cover

    def job_results(self, job: Job) -> Optional[list[Result]]:
        """Return the results of the job.

        Parameters
        ----------
        job : Job
            The job.

        Returns
        -------
        Optional[list[Result]]
            The results of the job.
        """
        try:
            _job = self._get_job(job.id)
            results = []
            for i, res in enumerate(_job.result()):  # type: ignore
                if hasattr(_job, "inputs"):
                    circuit = QiskitCircuitWrapper(_job.inputs["pubs"][i][0])  # type: ignore
                else:
                    circuit = job.circuits[i]  # type: ignore
                counts = {creg: data.get_counts() for creg, data in vars(res.data).items()}
                results.append(Result(job, circuit, counts))
            return results
        except Exception:  # pylint: disable=broad-except
            return None


@immutable
class IBMProviderWrapper(ProviderWrapper[IBMBackendWrapper]):
    """A wrapper for IBM provider.

    Parameters
    ----------
    **kwargs : Optional[str]
            The configuration parameters,
            currently the following
            parameters are supported:
            - channel: str
            - token: str
            - instance: str
            Alternatively, the environment
            variables QISKIT_IBM_CHANNEL,
            QISKIT_IBM_TOKEN, and
            QISKIT_IBM_INSTANCE (respectively)
            are used.
            Currently channel can be either
            "ibm_cloud" or "ibm_quantum".
    """

    _ENV_VARS = {
        "channel": "QISKIT_IBM_CHANNEL",
        "token": "QISKIT_IBM_TOKEN",
        "instance": "QISKIT_IBM_INSTANCE",
    }

    def __init__(self, **kwargs: Any):
        """Initialize the class.

        Parameters
        ----------
        **kwargs : Optional[str]
            The configuration parameters,
            currently the following
            parameters are supported:
            - channel: str
            - token: str
            - instance: str
            Alternatively, the environment
            variables QISKIT_IBM_CHANNEL,
            QISKIT_IBM_TOKEN, and
            QISKIT_IBM_INSTANCE (respectively)
            are used.
            Currently channel can be either
            "ibm_cloud" or "ibm_quantum".
        """

        channel = os.environ.get(self._ENV_VARS["channel"])
        token = os.environ.get(self._ENV_VARS["token"])
        instance = os.environ.get(self._ENV_VARS["instance"])

        if "channel" in kwargs:
            channel = kwargs.get("channel")
        if "token" in kwargs:
            token = kwargs.get("token")
        if "instance" in kwargs:
            instance = kwargs.get("instance")

        self._service = QiskitRuntimeService(
            channel=channel,
            token=token,
            instance=instance,
        )
        self._kwargs = kwargs

    def _wrap_backend(self, backend: str) -> IBMBackendWrapper:
        """Wrap the backend.

        Parameters
        ----------
        backend : str
            The backend.

        Returns
        -------
        IBMBackendWrapper
            The wrapped backend.
        """
        return IBMBackendWrapper(backend, **self._kwargs, provider=self.provider_id)

    def _backends(self) -> list[IBMBackendWrapper]:
        """Return the backends.

        Returns
        -------
        list[IBMBackendWrapper]
            The backends.
        """
        threads = []
        for backend in self._service.backends():
            thread = ThreadRV(target=self._wrap_backend, args=(backend.name,))
            thread.start()
            threads.append(thread)

        return [thread.join() for thread in threads]

    def get_backend(self, name: str) -> Optional[IBMBackendWrapper]:
        """Return the backend with the given name.

        Parameters
        ----------
        name : str
            The name of the backend.

        Returns
        -------
        Optional[IBMBackendWrapper]
            The backend with the given name.
        """
        try:
            return IBMBackendWrapper(name, **self._kwargs, provider=self.provider_id)
        except Exception:  # pylint: disable=broad-except
            return None


@immutable
class IBMFakeProviderWrapper(ProviderWrapper[IBMBackendWrapper]):
    """A wrapper for IBM fake provider.

    Parameters
    ----------
    **kwargs : Optional[str]
        The configuration parameters,
        currently not used.
    """

    def __init__(self, **kwargs: Any):
        """Initialize the class.

        Parameters
        ----------
        **kwargs : Optional[str]
            The configuration parameters,
            currently not used.
        """
        self._kwargs = kwargs
        self._provider = FakeProviderForBackendV2()

    def _wrap_backend(self, backend: str) -> IBMBackendWrapper:
        """Wrap the backend.

        Parameters
        ----------
        backend : str
            The backend.

        Returns
        -------
        IBMBackendWrapper
            The wrapped backend.
        """
        return IBMBackendWrapper(backend, **self._kwargs, provider=self.provider_id)

    def _backends(self) -> list[IBMBackendWrapper]:
        """Return the backends.

        Returns
        -------
        list[IBMBackendWrapper]
            The backends.
        """
        threads = []
        for backend in self._provider.backends():
            thread = ThreadRV(target=self._wrap_backend, args=(backend.name,))
            thread.start()
            threads.append(thread)

        return [thread.join() for thread in threads]

    def get_backend(self, name: str) -> Optional[IBMBackendWrapper]:
        """Return the backend with the given name.

        Parameters
        ----------
        name : str
            The name of the backend.

        Returns
        -------
        Optional[IBMBackendWrapper]
            The backend with the given name.
        """
        try:
            return IBMBackendWrapper(name, **self._kwargs, provider=self.provider_id)
        except Exception:  # pylint: disable=broad-except
            return None

    @property
    def provider_id(self) -> str:
        """Return the provider ID.

        Returns
        -------
        str
            The provider ID.
        """
        return "ibm_fake"


@immutable
class IBMAerProviderWrapper(ProviderWrapper[IBMBackendWrapper]):
    """A wrapper for IBM aer provider.

    All other IBM providers backends can be instantiated as AerSimulator
    by prefixing the backend name with "aer.".

    Parameters
    ----------
    **kwargs : Optional[str]
        The configuration parameters,
        currently not used.
    """

    def __init__(self, **kwargs: Any):
        """Initialize the class.

        Parameters
        ----------
        **kwargs : Optional[str]
            The configuration parameters,
            currently not used.
        """
        self._kwargs = kwargs
        self._provider = AerProvider()

    def _wrap_backend(self, backend: str) -> IBMBackendWrapper:
        """Wrap the backend.

        Parameters
        ----------
        backend : str
            The backend.

        Returns
        -------
        IBMBackendWrapper
            The wrapped backend.
        """
        return IBMBackendWrapper(backend, **self._kwargs, provider=self.provider_id)

    def _backends(self) -> list[IBMBackendWrapper]:
        """Return the backends.

        Returns
        -------
        list[IBMBackendWrapper]
            The backends.
        """
        noisy_models = []
        try:
            for device in QiskitRuntimeService().backends(simulator=False):
                noisy_models.append(device.name)
        except Exception:  # pylint: disable=broad-except # pragma: no cover
            pass

        threads = []
        for backend in self._provider.backends():
            thread = ThreadRV(target=self._wrap_backend, args=(backend.name,))
            thread.start()
            threads.append(thread)

        # for backend in noisy_models:
        #     thread = ThreadRV(target=self._wrap_backend, args=("aer." + backend,))
        #     thread.start()
        #     threads.append(thread)

        # for backend in FakeProviderForBackendV2().backends():
        #     thread = ThreadRV(target=self._wrap_backend, args=("aer." + backend.name,))
        #     thread.start()
        #     threads.append(thread)

        return [thread.join() for thread in threads]

    def get_backend(self, name: str) -> Optional[IBMBackendWrapper]:
        """Return the backend with the given name.

        Parameters
        ----------
        name : str
            The name of the backend.

        Returns
        -------
        Optional[IBMBackendWrapper]
            The backend with the given name.
        """
        try:
            return IBMBackendWrapper(name, **self._kwargs, provider=self.provider_id)
        except Exception:  # pylint: disable=broad-except
            return None

    @property
    def provider_id(self) -> str:
        """Return the provider ID.

        Returns
        -------
        str
            The provider ID.
        """
        return "ibm_aer"
