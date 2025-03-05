from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Optional

from llama_stack.apis.common.job_types import Job, JobStatus
from llama_stack.apis.eval import Eval, BenchmarkConfig, EvaluateResponse
from llama_stack.providers.datatypes import BenchmarksProtocolPrivate

# Configure logging
logger = logging.getLogger(__name__)


# Custom exceptions
class LMEvalError(Exception):
    """Base exception for LMEval errors"""

    pass


class LMEvalConfigError(LMEvalError):
    """Configuration related errors"""

    pass


class LMEvalValidationError(LMEvalError):
    """Validation related errors"""

    pass

class BaseLMEval(Eval, BenchmarksProtocolPrivate, ABC):
    async def initialize(self):
        print("Initializing Base LMEval")

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
        # Extract model information
        eval_candidate = task_config.eval_candidate
        model_name = eval_candidate.model

        # If using K8s, create a CR for this evaluation
        if hasattr(self, "k8s_config") and self.k8s_config:
            # Create CR spec
            cr_spec = {
                "apiVersion": "trustyai.opendatahub.io/v1alpha1",
                "kind": "LMEvalJob",
                "metadata": {"name": f"evaljob-{model_name.lower()}-{benchmark_id}"},
                "spec": {
                    "model": model_name,
                    "modelArgs": [],
                    "taskList": {"taskNames": [benchmark_id]},
                    "logSamples": True,
                    "pod": {"container": {"env": []}},
                },
            }

            # Add model args if available
            if hasattr(eval_candidate, "pretrained"):
                cr_spec["spec"]["modelArgs"].append({"name": "pretrained", "value": eval_candidate.pretrained})

            # Add environment variables if available
            if hasattr(task_config, "env_vars") and task_config.env_vars:
                cr_spec["spec"]["pod"]["container"]["env"] = task_config.env_vars

            # TODO: Submit CR to Kubernetes


            # Return job ID
            return Job(job_id=f"k8s-{cr_spec['metadata']['name']}")

        # TODO: Reserved for non-K8s evaluations

    async def job_status(self, benchmark_id: str, job_id: str) -> Optional[JobStatus]:
        pass

    async def job_cancel(self, benchmark_id: str, job_id: str) -> None:
        pass

    async def job_result(self, benchmark_id: str, job_id: str) -> EvaluateResponse:
        pass
