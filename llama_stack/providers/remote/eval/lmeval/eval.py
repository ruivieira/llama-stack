from __future__ import annotations

import logging
from datetime import time
from typing import Any, Dict, List, Optional

from kubernetes import client as k8s_client
from kubernetes import config as k8s_config
from kubernetes.client.rest import ApiException
from pydantic import BaseModel, Field

from llama_stack.apis.benchmarks import Benchmark, ListBenchmarksResponse
from llama_stack.apis.common.job_types import Job, JobStatus
from llama_stack.apis.eval import BenchmarkConfig, Eval, EvaluateResponse
from llama_stack.providers.datatypes import BenchmarksProtocolPrivate
from llama_stack.providers.remote.eval.lmeval.config import LMEvalEvalProviderConfig
from llama_stack.providers.remote.eval.lmeval.errors import LMEvalConfigError

logger = logging.getLogger(__name__)


class ModelArg(BaseModel):
    """Model argument for the LMEval CR."""

    name: str
    value: str


class ContainerConfig(BaseModel):
    """Container configuration for the LMEval CR."""

    env: Optional[List[Dict[str, str]]] = None


class PodConfig(BaseModel):
    """Pod configuration for the LMEval CR."""

    container: ContainerConfig
    serviceAccountName: Optional[str] = None


class TaskList(BaseModel):
    """Task list configuration for the LMEval Custom Resource."""

    taskNames: List[str]


class LMEvalSpec(BaseModel):
    """Specification for the LMEval Custom Resource."""

    allowOnline: bool = True
    allowCodeExecution: bool = True
    model: str = "local-completions"
    taskList: TaskList
    logSamples: bool = True
    batchSize: str = "1"
    modelArgs: List[ModelArg]
    pod: Optional[PodConfig] = None


class LMEvalMetadata(BaseModel):
    """Metadata for the LMEval Custom Resource."""

    name: str
    namespace: str


class LMEvalCR(BaseModel):
    """LMEval Custom Resource model."""

    apiVersion: str = "trustyai.opendatahub.io/v1alpha1"
    kind: str = "LMEvalJob"
    metadata: LMEvalMetadata
    spec: LMEvalSpec


class LMEvalCRBuilder:
    """An utility class which creates LMEval Custom Resources from BenchmarkConfigs."""

    def __init__(self, namespace: str = "default", service_account: Optional[str] = None):
        """Initialize the LMEvalCRBuilder.

        Args:
            namespace: The Kubernetes namespace to use
            service_account: Optional service account to use for the LMEval Custom Resource
        """
        self._namespace = namespace
        self._service_account = service_account

    def create_cr(self, benchmark_id: str, task_config: BenchmarkConfig, base_url: Optional[str] = None) -> dict:
        """Create LMEval Custom Resource from a Llama Stack BenchmarkConfig.

        Args:
            benchmark_id: The benchmark identifier
            task_config: Configuration for the evaluation task
            base_url: Optional base URL for the model service

        Returns:
            dict: LMEval CR specification

        Raises:
            LMEvalConfigError: If the configuration is invalid
        """
        # Model information
        eval_candidate = task_config.eval_candidate
        if eval_candidate.type != "model":
            # FIXME: Support other candidate types?
            raise LMEvalConfigError("LMEval only supports model candidates for now")

        model_name = eval_candidate.model
        # FIXME: Unused
        sampling_params = eval_candidate.sampling_params

        # Create model args from the configuration
        model_args = [
            ModelArg(name="model", value=model_name),
        ]

        if base_url:
            model_args.append(ModelArg(name="base_url", value=base_url))

        # Default parameters
        model_args.extend(
            [
                ModelArg(name="num_concurrent", value="1"),
                ModelArg(name="max_retries", value="3"),
                ModelArg(name="tokenized_requests", value="False"),
            ]
        )

        # Add tokenizer, if specified
        if hasattr(eval_candidate, "tokenizer"):
            model_args.append(ModelArg(name="tokenizer", value=eval_candidate.tokenizer))

        env_vars = []
        if hasattr(task_config, "env_vars") and task_config.env_vars:
            env_vars = task_config.env_vars

        # Extract the lm-evaluation-harness from the benchmark id, by removing the qualifier
        # e.g. If benchmark_id is "lmeval::task_name", extract "task_name"
        # FIXME: Improve this
        if "::" in benchmark_id:
            task_name = benchmark_id.split("::")[-1]

        # Generate timestamp-based uid
        import time

        job_id = int(time.time() * 1000)

        pod_config = None
        if env_vars or self._service_account:
            container_config = ContainerConfig(env=env_vars) if env_vars else ContainerConfig()
            pod_config = PodConfig(container=container_config, serviceAccountName=self._service_account)

        # FIXME: Assert task_name is valid and non-empty
        cr = LMEvalCR(
            metadata=LMEvalMetadata(name=f"lmeval-llama-stack-job-{job_id}", namespace=self._namespace),
            spec=LMEvalSpec(taskList=TaskList(taskNames=[task_name]), modelArgs=model_args, pod=pod_config),
        )

        return cr.model_dump()


class LMEval(Eval, BenchmarksProtocolPrivate):
    def __init__(self, config: LMEvalEvalProviderConfig):
        self._config = config
        logger.info(f"LMEval provider initialized with namespace: {getattr(self._config, 'namespace', 'default')}")
        logger.info(f"LMEval provider config values: {vars(self._config)}")
        self.benchmarks = {}
        self._jobs: List[Job] = []

        self._k8s_client = None
        self._k8s_custom_api = None
        self._namespace = getattr(self._config, "namespace", "default")
        logger.info(f"Initialized Kubernetes client with namespace: {self._namespace}")
        if self.use_k8s:
            self._init_k8s_client()
            self._cr_builder = LMEvalCRBuilder(
                namespace=self._namespace, service_account=getattr(self._config, "service_account", None)
            )

    def _init_k8s_client(self):
        """Initialize the Kubernetes client."""
        # FIXME: Support in-cluster kubeconfig only?
        try:
            k8s_config.load_incluster_config()
            logger.info("Loaded Kubernetes config from within the cluster")
        except k8s_config.ConfigException:
            try:
                k8s_config.load_kube_config()
                logger.info("Loaded Kubernetes config from kubeconfig file")
            except k8s_config.ConfigException as e:
                logger.error(f"Failed to load Kubernetes config: {e}")
                raise LMEvalConfigError(f"Failed to initialize Kubernetes client: {e}")

        self._k8s_client = k8s_client.ApiClient()
        self._k8s_custom_api = k8s_client.CustomObjectsApi(self._k8s_client)

    @property
    def use_k8s(self) -> bool:
        """Check if K8s mode is enabled."""
        return getattr(self._config, "use_k8s", True)  # Default to True if not specified

    async def initialize(self):
        logger.info("Initializing Base LMEval")
        print("Initializing Base LMEval")


    async def _register_bundled_benchmarks(self):
        """Register bundled benchmarks from lm-evaluation-harness."""
        bundled_benchmarks = self._get_benchmarks()

        for benchmark in bundled_benchmarks:
            await self.register_benchmark(benchmark)
            logger.info(f"Registered bundled benchmark: {benchmark.identifier}")

    async def list_benchmarks(self) -> ListBenchmarksResponse:
        """List all registered benchmarks.

        Returns:
            ListBenchmarksResponse: Response containing all registered benchmarks
        """
        return ListBenchmarksResponse(data=list(self.benchmarks.values()))

    async def get_benchmark(self, benchmark_id: str) -> Optional[Benchmark]:
        """Get a specific benchmark by ID.

        Args:
            benchmark_id: The benchmark identifier

        Returns:
            Optional[Benchmark]: The benchmark if found, None otherwise
        """
        return self.benchmarks.get(benchmark_id)

    async def register_benchmark(self, benchmark: Benchmark) -> None:
        """Register a benchmark for evaluation.

        Args:
            benchmark: The benchmark to register
        """
        logger.info(f"Registering benchmark: {benchmark.identifier}")
        self.benchmarks[benchmark.identifier] = benchmark

    async def run_eval(
        self,
        benchmark_id: str,
        benchmark_config: BenchmarkConfig,
    ) -> Job:
        """Run an evaluation for a specific benchmark and configuration.

        Args:
            benchmark_id: The benchmark id
            benchmark_config: Configuration for the evaluation task

        Returns:
            Job: Job identifier for tracking the evaluation
        """
        if self.use_k8s:
            if not isinstance(benchmark_config, BenchmarkConfig):
                raise LMEvalConfigError("K8s mode requires BenchmarkConfig")

            cr = self._cr_builder.create_cr(
                benchmark_id=benchmark_id,
                task_config=benchmark_config,
                base_url=getattr(self._config, "base_url", None),
            )
            logger.info(f"Generated LMEval CR for benchmark {benchmark_id}: {cr}")

            _job_id = len(self._jobs)
            _job = Job(job_id=f"lmeval-job-{_job_id}", status=JobStatus.scheduled, metadata={"created_at": str(time())})
            self._jobs.append(_job)

            # Deploy LMEvalJob
            try:
                group = "trustyai.opendatahub.io"
                version = "v1alpha1"
                plural = "lmevaljobs"

                logger.info(f"Deploying LMEval CR to Kubernetes namespace: {self._namespace}")

                import json

                logger.info(f"Full CR being submitted: {json.dumps(cr, indent=2)}")

                response = self._k8s_custom_api.create_namespaced_custom_object(
                    group=group, version=version, namespace=self._namespace, plural=plural, body=cr
                )

                logger.info(f"Successfully deployed LMEval CR to Kubernetes: {response['metadata']['name']}")

                _job.metadata = {"k8s_name": cr["metadata"]["name"]}

            except ApiException as e:
                logger.error(f"Failed to deploy LMEval CR to Kubernetes: {e}")
                raise LMEvalConfigError(f"Failed to deploy LMEval CR: {e}")

            return _job
        else:
            # TODO: Handle non-K8s evaluation
            raise NotImplementedError("Non-K8s evaluation not implemented yet")

    async def evaluate_rows(
        self,
        benchmark_id: str,
        input_rows: List[Dict[str, Any]],
        scoring_functions: List[str],
        benchmark_config: BenchmarkConfig,
    ) -> EvaluateResponse:
        """Evaluate a list of rows on a benchmark.

        Args:
            benchmark_id: The ID of the benchmark to run the evaluation on.
            input_rows: The rows to evaluate.
            scoring_functions: The scoring functions to use for the evaluation.
            benchmark_config: The configuration for the benchmark.

        Returns:
            EvaluateResponse: Object containing generations and scores
        """
        if self.use_k8s:
            # FIXME: Placeholder
            from llama_stack.apis.scoring import ScoringResult

            # FIXME: Placeholder
            generations = []
            for row in input_rows:
                generation = {**row, "generated_answer": "Placeholder answer from LMEval"}
                generations.append(generation)

            scores = {}
            for scoring_fn in scoring_functions:
                score_rows = [{"score": 0.5} for _ in input_rows]
                scores[scoring_fn] = ScoringResult(aggregated_results={"accuracy": 0.5}, score_rows=score_rows)

            return EvaluateResponse(generations=generations, scores=scores)
        else:
            raise NotImplementedError("Non-K8s evaluation not implemented yet")

    async def job_status(self, benchmark_id: str, job_id: str) -> Optional[JobStatus]:
        """Get the status of a running evaluation job.

        Args:
            benchmark_id: The benchmark id
            job_id: The job id

        Returns:
            JobStatus: Current status of the job
        """
        if self.use_k8s:
            job = next((j for j in self._jobs if j.job_id == job_id), None)
            if not job:
                logger.warning(f"Job {job_id} not found")
                return None

            try:
                k8s_name = job.metadata.get("k8s_name") if hasattr(job, "metadata") else None
                if not k8s_name:
                    logger.warning(f"No K8s resource name found for job {job_id}")
                    return JobStatus.unknown

                group = "trustyai.opendatahub.io"
                version = "v1alpha1"
                plural = "lmevaljobs"

                cr = self._k8s_custom_api.get_namespaced_custom_object(
                    group=group, version=version, namespace=self._namespace, plural=plural, name=k8s_name
                )

                status = cr.get("status", {}).get("state", "unknown")

                if status == "Completed":
                    return JobStatus.completed
                elif status == "Failed":
                    return JobStatus.failed
                elif status in ["Running", "Pending"]:
                    return JobStatus.in_progress
                else:
                    return JobStatus.unknown

            except ApiException as e:
                logger.error(f"Failed to get job status from Kubernetes: {e}")
                return JobStatus.unknown
        else:
            raise NotImplementedError("Non-K8s evaluation not implemented yet")

    async def job_cancel(self, benchmark_id: str, job_id: str) -> None:
        """Cancel a running evaluation job.

        Args:
            benchmark_id: The benchmark identifier
            job_id: The job identifier
        """
        if self.use_k8s:
            job = next((j for j in self._jobs if j.job_id == job_id), None)
            if not job:
                logger.warning(f"Job {job_id} not found")
                return

            try:
                k8s_name = job.metadata.get("k8s_name") if hasattr(job, "metadata") else None
                if not k8s_name:
                    logger.warning(f"No K8s resource name found for job {job_id}")
                    return

                # Delete the LMEvalJob
                group = "trustyai.opendatahub.io"
                version = "v1alpha1"
                plural = "lmevaljobs"

                self._k8s_custom_api.delete_namespaced_custom_object(
                    group=group, version=version, namespace=self._namespace, plural=plural, name=k8s_name
                )

                logger.info(f"Successfully cancelled job {job_id} (K8s resource: {k8s_name})")

            except ApiException as e:
                logger.error(f"Failed to cancel job in Kubernetes: {e}")
                raise LMEvalConfigError(f"Failed to cancel job: {e}")
        else:
            raise NotImplementedError("Non-K8s evaluation not implemented yet")

    async def job_result(self, benchmark_id: str, job_id: str) -> EvaluateResponse:
        """Get the results of a completed evaluation job.

        Args:
            benchmark_id: The benchmark id
            job_id: The job id

        Returns:
            EvaluateResponse: Results of the evaluation
        """
        if self.use_k8s:
            job = next((j for j in self._jobs if j.job_id == job_id), None)
            if not job:
                logger.warning(f"Job {job_id} not found")
                return EvaluateResponse(generations=[], scores={})

            try:
                k8s_name = job.metadata.get("k8s_name") if hasattr(job, "metadata") else None
                if not k8s_name:
                    logger.warning(f"No K8s resource name found for job {job_id}")
                    return EvaluateResponse(generations=[], scores={})

                group = "trustyai.opendatahub.io"
                version = "v1alpha1"
                plural = "lmevaljobs"

                cr = self._k8s_custom_api.get_namespaced_custom_object(
                    group=group, version=version, namespace=self._namespace, plural=plural, name=k8s_name
                )

                status = cr.get("status", {}).get("state", "unknown")
                if status != "Completed":
                    logger.warning(f"Job {job_id} is not completed yet (status: {status})")
                    return EvaluateResponse(generations=[], scores={})

                results = cr.get("status", {}).get("results", {})

                from llama_stack.apis.scoring import ScoringResult

                generations = []
                if "samples" in results:
                    for sample in results["samples"]:
                        generation = {"generated_answer": sample.get("output", "")}
                        if "input" in sample:
                            generation["input"] = sample["input"]
                        generations.append(generation)

                scores = {}
                if "metrics" in results:
                    metrics = results["metrics"]
                    for metric_name, metric_value in metrics.items():
                        # Create a scoring result for each metric
                        score_rows = [{"score": metric_value}]
                        scores[metric_name] = ScoringResult(
                            aggregated_results={metric_name: metric_value}, score_rows=score_rows
                        )

                return EvaluateResponse(generations=generations, scores=scores)

            except ApiException as e:
                logger.error(f"Failed to get job results from Kubernetes: {e}")
                return EvaluateResponse(generations=[], scores={})
        else:
            raise NotImplementedError("Non-K8s evaluation not implemented yet")

    async def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        logger.info("Shutting down LMEval provider")
        if self._k8s_client:
            self._k8s_client.close()
            logger.info("Closed Kubernetes client connection")
