from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from kubernetes import client as k8s_client
from kubernetes import config as k8s_config
from kubernetes.client.rest import ApiException

from llama_stack.apis.benchmarks import Benchmark, ListBenchmarksResponse
from llama_stack.apis.common.job_types import Job, JobStatus
from llama_stack.apis.eval import BenchmarkConfig, Eval, EvaluateResponse
from llama_stack.providers.datatypes import BenchmarksProtocolPrivate
from llama_stack.providers.remote.eval.lmeval.config import LMEvalEvalProviderConfig
from llama_stack.providers.remote.eval.lmeval.errors import LMEvalConfigError

# Configure logging
logger = logging.getLogger(__name__)


class LMEval(Eval, BenchmarksProtocolPrivate):
    def __init__(self, config: LMEvalEvalProviderConfig):
        self._config = config
        _benches = self._get_benchmarks()
        self.benchmarks = {b.identifier: b for b in _benches}
        self._register_bundled_benchmarks()
        self._jobs: List[Job] = []

        # Initialize Kubernetes client if using K8s
        self._k8s_client = None
        self._k8s_custom_api = None
        if self.use_k8s:
            self._init_k8s_client()

    def _init_k8s_client(self):
        """Initialize the Kubernetes client."""
        try:
            # Try to load config from within the cluster
            k8s_config.load_incluster_config()
            logger.info("Loaded Kubernetes config from within the cluster")
        except k8s_config.ConfigException:
            # If that fails, try to load from kubeconfig file
            try:
                k8s_config.load_kube_config()
                logger.info("Loaded Kubernetes config from kubeconfig file")
            except k8s_config.ConfigException as e:
                logger.error(f"Failed to load Kubernetes config: {e}")
                raise LMEvalConfigError(f"Failed to initialize Kubernetes client: {e}")

        # Create the Kubernetes API clients
        self._k8s_client = k8s_client.ApiClient()
        self._k8s_custom_api = k8s_client.CustomObjectsApi(self._k8s_client)

        # Set namespace from config or default to 'default'
        self._namespace = getattr(self._config, "namespace", "default")
        logger.info(f"Initialized Kubernetes client with namespace: {self._namespace}")

    @property
    def use_k8s(self) -> bool:
        """Check if K8s mode is enabled in the configuration."""
        return getattr(self._config, "use_k8s", True)  # Default to True if not specified

    async def initialize(self):
        logger.info("Initializing Base LMEval")
        print("Initializing Base LMEval")

        # Register bundled tasks from lm-evaluation-harness
        await self._register_bundled_benchmarks()

    def _get_benchmarks(self) -> List[Benchmark]:
        return [
            # MMLU benchmark
            Benchmark(
                identifier="lmeval:mmlu",
                dataset_id="mmlu",
                scoring_functions=["basic::accuracy"],
                metadata={
                    "description": "Massive Multitask Language Understanding - A benchmark testing knowledge across 57 subjects",
                    "lmeval_task": "mmlu",
                    "categories": ["knowledge", "reasoning"],
                    "languages": ["english"],
                },
                provider_id="lmeval",
                provider_resource_id="mmlu",
            ),
            # GSM8K benchmark
            Benchmark(
                identifier="lmeval:gsm8k",
                dataset_id="gsm8k",
                scoring_functions=["basic::accuracy"],
                metadata={
                    "description": "Grade School Math 8K - A dataset of 8.5K high quality grade school math word problems",
                    "lmeval_task": "gsm8k",
                    "categories": ["math", "reasoning"],
                    "languages": ["english"],
                },
                provider_id="lmeval",
                provider_resource_id="gsm8k",
            ),
            # HumanEval benchmark
            Benchmark(
                identifier="lmeval:humaneval",
                dataset_id="humaneval",
                scoring_functions=["basic::pass@k"],
                metadata={
                    "description": "HumanEval - A benchmark for evaluating code generation capabilities",
                    "lmeval_task": "humaneval",
                    "categories": ["code", "programming"],
                    "languages": ["python"],
                },
                provider_id="lmeval",
                provider_resource_id="humaneval",
            ),
            # HellaSwag benchmark
            Benchmark(
                identifier="lmeval:hellaswag",
                dataset_id="hellaswag",
                scoring_functions=["basic::accuracy"],
                metadata={
                    "description": "HellaSwag - A challenging commonsense NLI dataset",
                    "lmeval_task": "hellaswag",
                    "categories": ["commonsense", "reasoning"],
                    "languages": ["english"],
                },
                provider_id="lmeval",
                provider_resource_id="hellaswag",
            ),
            # TruthfulQA benchmark
            Benchmark(
                identifier="lmeval:truthfulqa",
                dataset_id="truthfulqa",
                scoring_functions=["basic::accuracy"],
                metadata={
                    "description": "TruthfulQA - A benchmark to measure whether a language model is truthful",
                    "lmeval_task": "truthfulqa_mc",
                    "categories": ["knowledge", "truthfulness"],
                    "languages": ["english"],
                },
                provider_id="lmeval",
                provider_resource_id="truthfulqa_mc",
            ),
            # ARC benchmark (Easy)
            Benchmark(
                identifier="lmeval:arc_easy",
                dataset_id="arc_easy",
                scoring_functions=["basic::accuracy"],
                metadata={
                    "description": "AI2 Reasoning Challenge (Easy) - Multiple-choice questions from science exams",
                    "lmeval_task": "arc_easy",
                    "categories": ["knowledge", "reasoning", "science"],
                    "languages": ["english"],
                },
                provider_id="lmeval",
                provider_resource_id="arc_easy",
            ),
            # ARC benchmark (Challenge)
            Benchmark(
                identifier="lmeval:arc_challenge",
                dataset_id="arc_challenge",
                scoring_functions=["basic::accuracy"],
                metadata={
                    "description": "AI2 Reasoning Challenge (Challenge) - Difficult multiple-choice questions from science exams",
                    "lmeval_task": "arc_challenge",
                    "categories": ["knowledge", "reasoning", "science"],
                    "languages": ["english"],
                },
                provider_id="lmeval",
                provider_resource_id="arc_challenge",
            ),
            # WinoGrande benchmark
            Benchmark(
                identifier="lmeval:winogrande",
                dataset_id="winogrande",
                scoring_functions=["basic::accuracy"],
                metadata={
                    "description": "WinoGrande - A large-scale dataset for coreference resolution",
                    "lmeval_task": "winogrande",
                    "categories": ["reasoning", "coreference"],
                    "languages": ["english"],
                },
                provider_id="lmeval",
                provider_resource_id="winogrande",
            ),
            # LAMBADA benchmark
            Benchmark(
                identifier="lmeval:lambada",
                dataset_id="lambada",
                scoring_functions=["basic::accuracy"],
                metadata={
                    "description": "LAMBADA - A dataset for testing models' ability to understand long-range dependencies",
                    "lmeval_task": "lambada",
                    "categories": ["comprehension", "long-context"],
                    "languages": ["english"],
                },
                provider_id="lmeval",
                provider_resource_id="lambada",
            ),
            # ToxiGen benchmark
            Benchmark(
                identifier="lmeval:toxigen",
                dataset_id="toxigen",
                scoring_functions=["basic::toxicity"],
                metadata={
                    "description": "ToxiGen - A benchmark for evaluating language models on toxic content generation",
                    "lmeval_task": "toxigen",
                    "categories": ["safety", "toxicity"],
                    "languages": ["english"],
                },
                provider_id="lmeval",
                provider_resource_id="toxigen",
            ),
            # BIG-bench benchmark (Logical Deduction)
            Benchmark(
                identifier="lmeval:bigbench_logical_deduction",
                dataset_id="bigbench_logical_deduction",
                scoring_functions=["basic::accuracy"],
                metadata={
                    "description": "BIG-bench Logical Deduction - Tests logical reasoning capabilities",
                    "lmeval_task": "bigbench_logical_deduction",
                    "categories": ["reasoning", "logic"],
                    "languages": ["english"],
                },
                provider_id="lmeval",
                provider_resource_id="bigbench_logical_deduction",
            ),
            # BIG-bench benchmark (Causal Judgment)
            Benchmark(
                identifier="lmeval:bigbench_causal_judgment",
                dataset_id="bigbench_causal_judgment",
                scoring_functions=["basic::accuracy"],
                metadata={
                    "description": "BIG-bench Causal Judgment - Tests causal reasoning capabilities",
                    "lmeval_task": "bigbench_causal_judgment",
                    "categories": ["reasoning", "causality"],
                    "languages": ["english"],
                },
                provider_id="lmeval",
                provider_resource_id="bigbench_causal_judgment",
            ),
            # BIG-bench benchmark (Date Understanding)
            Benchmark(
                identifier="lmeval:bigbench_date_understanding",
                dataset_id="bigbench_date_understanding",
                scoring_functions=["basic::accuracy"],
                metadata={
                    "description": "BIG-bench Date Understanding - Tests temporal reasoning capabilities",
                    "lmeval_task": "bigbench_date_understanding",
                    "categories": ["reasoning", "temporal"],
                    "languages": ["english"],
                },
                provider_id="lmeval",
                provider_resource_id="bigbench_date_understanding",
            ),
        ]

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

    def group_benchmarks_by_category(self) -> Dict[str, List[Benchmark]]:
        """Group benchmarks by category.

        Returns:
            Dict[str, List[Benchmark]]: Dictionary mapping category names to lists of benchmarks
        """
        categories: Dict[str, List[Benchmark]] = {}

        for benchmark in self.benchmarks.values():
            if "categories" in benchmark.metadata:
                for category in benchmark.metadata["categories"]:
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(benchmark)

        return categories

    def get_benchmarks_by_language(self, language: str) -> List[Benchmark]:
        """Get benchmarks for a specific language.

        Args:
            language: The language to filter by (e.g., "english", "python")

        Returns:
            List[Benchmark]: List of benchmarks for the specified language
        """
        result = []

        for benchmark in self.benchmarks.values():
            if "languages" in benchmark.metadata:
                if language.lower() in [lang.lower() for lang in benchmark.metadata["languages"]]:
                    result.append(benchmark)

        return result

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
        """Run evaluation for the given benchmark and configuration.

        Args:
            benchmark_id: The benchmark identifier
            benchmark_config: Configuration for the evaluation task

        Returns:
            Job: Job identifier for tracking the evaluation
        """
        # Check if we're using K8s mode
        if self.use_k8s:
            # Ensure task_config is a BenchmarkConfig
            if not isinstance(benchmark_config, BenchmarkConfig):
                raise LMEvalConfigError("K8s mode requires BenchmarkConfig")

            # Generate K8s CR from the benchmark config
            cr = self._create_lmeval_cr(benchmark_id, benchmark_config)
            logger.info(f"Generated LMEval CR for benchmark {benchmark_id}: {cr}")

            # Create job ID
            _job_id = len(self._jobs)
            _job = Job(job_id=f"lmeval-job-{_job_id}")
            self._jobs.append(_job)

            # Deploy the CR to Kubernetes
            try:
                # Define the CR group, version, plural
                group = "trustyai.opendatahub.io"
                version = "v1alpha1"
                plural = "lmevaljobs"

                # Create the CR in the cluster
                response = self._k8s_custom_api.create_namespaced_custom_object(
                    group=group, version=version, namespace=self._namespace, plural=plural, body=cr
                )

                logger.info(f"Successfully deployed LMEval CR to Kubernetes: {response['metadata']['name']}")

                # Store the K8s resource name in the job for later reference
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
            # For now, return a placeholder result with the correct structure
            from llama_stack.apis.scoring import ScoringResult

            # Create placeholder generations for each input row
            generations = []
            for row in input_rows:
                # Create a placeholder generation that includes the input row fields
                # and adds a generated_answer field
                generation = {**row, "generated_answer": "Placeholder answer from LMEval"}
                generations.append(generation)

            # Create placeholder scoring results for each scoring function
            scores = {}
            for scoring_fn in scoring_functions:
                # Create a placeholder scoring result with a score of 0.5 for each row
                score_rows = [{"score": 0.5} for _ in input_rows]
                scores[scoring_fn] = ScoringResult(aggregated_results={"accuracy": 0.5}, score_rows=score_rows)

            # Return with the correct structure
            return EvaluateResponse(generations=generations, scores=scores)
        else:
            raise NotImplementedError("Non-K8s evaluation not implemented yet")

    def _create_lmeval_cr(self, benchmark_id: str, task_config: BenchmarkConfig) -> dict:
        """Create LMEval Custom Resource from Llama Stack BenchmarkConfig.

        Args:
            benchmark_id: The benchmark identifier
            task_config: Configuration for the evaluation task

        Returns:
            dict: LMEval CR specification
        """
        # Extract model information from eval_candidate
        eval_candidate = task_config.eval_candidate
        if eval_candidate.type != "model":
            raise LMEvalConfigError("LMEval only supports model candidates")

        # Extract model name and sampling parameters
        model_name = eval_candidate.model
        sampling_params = eval_candidate.sampling_params

        # Create model args from the configuration
        model_args = [
            {"name": "model", "value": model_name},
        ]

        # Add base_url if available in config
        if hasattr(self._config, "base_url") and self._config.base_url:
            model_args.append({"name": "base_url", "value": self._config.base_url})

        # Add default parameters
        model_args.extend(
            [
                {"name": "num_concurrent", "value": "1"},
                {"name": "max_retries", "value": "3"},
                {"name": "tokenized_requests", "value": "False"},
            ]
        )

        # If there's a tokenizer specified, add it
        if hasattr(eval_candidate, "tokenizer"):
            model_args.append({"name": "tokenizer", "value": eval_candidate.tokenizer})

        # Extract environment variables if available
        env_vars = []
        if hasattr(task_config, "env_vars") and task_config.env_vars:
            env_vars = task_config.env_vars

        # Extract the actual task name from the benchmark_id
        # If benchmark_id is in format "lmeval::task_name", extract just "task_name"
        task_name = benchmark_id
        if "::" in benchmark_id:
            _, task_name = benchmark_id.split("::", 1)

        # Generate a unique job ID
        job_id = len(self._jobs)

        # Create the CR
        cr = {
            "apiVersion": "trustyai.opendatahub.io/v1alpha1",
            "kind": "LMEvalJob",
            "metadata": {
                "name": f"lmeval-llama-stack-job-{job_id}",
            },
            "spec": {
                "model": "local-completions",  # Default model type
                "taskList": {
                    "taskNames": [task_name]  # Use just the task name, not the full benchmark_id
                },
                "logSamples": True,
                "batchSize": 1,
                "modelArgs": model_args,
            },
        }

        # Add pod configuration if env vars are present
        if env_vars:
            cr["spec"]["pod"] = {"container": {"env": env_vars}}

        return cr

    async def job_status(self, benchmark_id: str, job_id: str) -> Optional[JobStatus]:
        """Get the status of a running evaluation job.

        Args:
            benchmark_id: The benchmark identifier
            job_id: The job identifier

        Returns:
            JobStatus: Current status of the job
        """
        if self.use_k8s:
            # Find the job in our list
            job = next((j for j in self._jobs if j.job_id == job_id), None)
            if not job:
                logger.warning(f"Job {job_id} not found")
                return None

            try:
                # Get the K8s resource name from job metadata
                k8s_name = job.metadata.get("k8s_name") if hasattr(job, "metadata") else None
                if not k8s_name:
                    logger.warning(f"No K8s resource name found for job {job_id}")
                    return JobStatus.unknown

                # Get the CR status from Kubernetes
                group = "trustyai.opendatahub.io"
                version = "v1alpha1"
                plural = "lmevaljobs"

                cr = self._k8s_custom_api.get_namespaced_custom_object(
                    group=group, version=version, namespace=self._namespace, plural=plural, name=k8s_name
                )

                # Map the CR status to JobStatus
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
            # Find the job in our list
            job = next((j for j in self._jobs if j.job_id == job_id), None)
            if not job:
                logger.warning(f"Job {job_id} not found")
                return

            try:
                # Get the K8s resource name from job metadata
                k8s_name = job.metadata.get("k8s_name") if hasattr(job, "metadata") else None
                if not k8s_name:
                    logger.warning(f"No K8s resource name found for job {job_id}")
                    return

                # Delete the CR from Kubernetes
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
            benchmark_id: The benchmark identifier
            job_id: The job identifier

        Returns:
            EvaluateResponse: Results of the evaluation
        """
        if self.use_k8s:
            # Find the job in our list
            job = next((j for j in self._jobs if j.job_id == job_id), None)
            if not job:
                logger.warning(f"Job {job_id} not found")
                return EvaluateResponse(generations=[], scores={})

            try:
                # Get the K8s resource name from job metadata
                k8s_name = job.metadata.get("k8s_name") if hasattr(job, "metadata") else None
                if not k8s_name:
                    logger.warning(f"No K8s resource name found for job {job_id}")
                    return EvaluateResponse(generations=[], scores={})

                # Get the CR from Kubernetes
                group = "trustyai.opendatahub.io"
                version = "v1alpha1"
                plural = "lmevaljobs"

                cr = self._k8s_custom_api.get_namespaced_custom_object(
                    group=group, version=version, namespace=self._namespace, plural=plural, name=k8s_name
                )

                # Check if the job is completed
                status = cr.get("status", {}).get("state", "unknown")
                if status != "Completed":
                    logger.warning(f"Job {job_id} is not completed yet (status: {status})")
                    return EvaluateResponse(generations=[], scores={})

                # Extract results from the CR
                results = cr.get("status", {}).get("results", {})

                # Convert results to EvaluateResponse format
                from llama_stack.apis.scoring import ScoringResult

                # Extract generations if available
                generations = []
                if "samples" in results:
                    for sample in results["samples"]:
                        generation = {"generated_answer": sample.get("output", "")}
                        if "input" in sample:
                            generation["input"] = sample["input"]
                        generations.append(generation)

                # Extract scores if available
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
                # Return empty results on error
                return EvaluateResponse(generations=[], scores={})
        else:
            raise NotImplementedError("Non-K8s evaluation not implemented yet")

    async def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        logger.info("Shutting down LMEval provider")
        if self._k8s_client:
            self._k8s_client.close()
            logger.info("Closed Kubernetes client connection")
