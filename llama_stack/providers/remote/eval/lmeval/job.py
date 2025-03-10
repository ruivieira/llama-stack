from __future__ import annotations

import logging
from typing import Optional

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

    async def initialize(self):
        logger.info("Initializing Base LMEval")

    async def run_eval(
        self,
        benchmark_id: str,
        task_config: BenchmarkConfig,
    ) -> Job:
        """Run evaluation for the given benchmark and configuration.

        Args:
            benchmark_id: The benchmark identifier
            task_config: Configuration for the evaluation task

        Returns:
            Job: Job identifier for tracking the evaluation
        """
        # Check if we're using K8s mode
        if self.use_k8s:
            # Ensure task_config is a BenchmarkConfig
            if not isinstance(task_config, BenchmarkConfig):
                raise LMEvalConfigError("K8s mode requires BenchmarkConfig")

            # Generate K8s CR from the benchmark config
            cr = self._create_lmeval_cr(benchmark_id, task_config)

            # TODO: Submit the CR to K8s (implementation would depend on K8s client)
            logger.info(f"Generated LMEval CR for benchmark {benchmark_id}: {cr}")

            # Return job ID (for now just use a placeholder)
            return Job(job_id="lmeval-job-1")
        else:
            # TODO: Handle non-K8s evaluation
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

        # Create the CR
        cr = {
            "apiVersion": "trustyai.opendatahub.io/v1alpha1",
            "kind": "LMEvalJob",
            "metadata": {
                # TODO: Add random suffix to avoid collisions
                "name": f"lmeval-{benchmark_id}".lower().replace(":", "-"),
            },
            "spec": {
                "model": "local-completions",  # Default model type
                "taskList": {
                    "taskNames": [benchmark_id]  # Use benchmark_id as task name
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
            # TODO: Implement K8s job status checking
            # For now, return a placeholder status
            return JobStatus.in_progress
        else:
            raise NotImplementedError("Non-K8s evaluation not implemented yet")

    async def job_cancel(self, benchmark_id: str, job_id: str) -> None:
        """Cancel a running evaluation job.

        Args:
            benchmark_id: The benchmark identifier
            job_id: The job identifier
        """
        if self.use_k8s:
            # TODO: Implement K8s job cancellation
            logger.info(f"Cancelling job {job_id} for benchmark {benchmark_id}")
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
            # TODO: Implement K8s job result retrieval
            # For now, return a placeholder result with the correct structure
            from llama_stack.apis.scoring import ScoringResult

            # Create a placeholder scoring result
            scoring_result = ScoringResult(
                aggregated_results={"accuracy": 0.75, "f1_score": 0.8}, score_rows=[{"score": 0.8}]
            )

            # Return with the correct structure
            return EvaluateResponse(
                generations=[{"generated_answer": "Placeholder answer"}], scores={"default_scorer": scoring_result}
            )
        else:
            raise NotImplementedError("Non-K8s evaluation not implemented yet")
