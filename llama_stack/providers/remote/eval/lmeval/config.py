from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llama_stack.apis.eval import BenchmarkConfig, EvalCandidate
from llama_stack.providers.remote.eval.lmeval.errors import LMEvalConfigError
from llama_stack.schema_utils import json_schema_type


@json_schema_type
@dataclass
class LMEvalBenchmarkConfig(BenchmarkConfig):
    """Configuration for LMEval benchmark that extends the base BenchmarkConfig"""

    # K8s specific configuration
    model: str
    eval_candidate: EvalCandidate

    def __post_init__(self):
        """Validate configuration"""
        super().__post_init__()

        if not self.model:
            raise ValueError("model must be provided")

    def to_k8s_cr(self) -> Dict[str, Any]:
        """Convert configuration to Kubernetes CR format"""
        cr = {
            "apiVersion": "trustyai.opendatahub.io/v1alpha1",
            "kind": "LMEvalJob",
            "metadata": {"name": f"evaljob-{self.model.lower()}-1"},
            "spec": {
                "model": self.model,
                "modelArgs": "",
                "taskList": "",
                "logSamples": "",
                "pod": {"container": {"env": ""}},
            },
        }

        return cr

    @classmethod
    def from_k8s_cr(cls, cr: Dict[str, Any]) -> "LMEvalBenchmarkConfig":
        """Create configuration from Kubernetes CR"""
        spec = cr.get("spec", {})

        # Extract values from CR
        model = spec.get("model", "hf")
        model_args = spec.get("modelArgs", [])
        task_list = spec.get("taskList", {"taskNames": ["unfair_tos"]})
        log_samples = spec.get("logSamples", True)
        namespace = cr.get("metadata", {}).get("namespace", "default")

        # Extract environment variables
        env_vars = []
        if "pod" in spec and "container" in spec["pod"]:
            env_vars = spec["pod"]["container"].get("env", [])

        # Create config with K8s parameters
        return cls(
            model=model,
            model_args=model_args,
            task_list=task_list,
            log_samples=log_samples,
            namespace=namespace,
            env_vars=env_vars,
            # Add required BenchmarkConfig fields here
            eval_candidate=None,
            scoring_params={},
        )


@json_schema_type
@dataclass
class K8sLMEvalConfig:
    """Configuration for Kubernetes LMEvalJob CR"""

    model: str
    model_args: Optional[List[Dict[str, str]]] = field(default_factory=list)
    task_list: Optional[Dict[str, List[str]]] = None
    log_samples: bool = True
    namespace: str = "default"
    env_vars: Optional[List[Dict[str, str]]] = None

    def __post_init__(self):
        """Validate configuration"""
        if not self.task_list or not self.task_list.get("taskNames"):
            raise ValueError("taskList.taskNames must be provided")

        if not self.model:
            raise ValueError("model must be provided")


@json_schema_type
@dataclass
class LMEvalEvalProviderConfig:
    """Configuration for the LMEval Provider"""

    use_k8s: bool = True
    base_url: str = "/v1/completions"

    def __post_init__(self):
        """Validate configuration"""
        if not isinstance(self.use_k8s, bool):
            raise LMEvalConfigError("use_k8s must be a boolean")
        if self.use_k8s is False:
            raise LMEvalConfigError("Only Kubernetes LMEval backend is supported at the moment")


__all__ = ["LMEvalBenchmarkConfig", "K8sLMEvalConfig", "LMEvalEvalProviderConfig"]
