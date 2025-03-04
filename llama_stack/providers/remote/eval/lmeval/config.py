from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from llama_stack.schema_utils import json_schema_type


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

    eval_config: Dict[str, Any]
    k8s_config: Optional[K8sLMEvalConfig] = None

    def __post_init__(self):
        """Convert configuration dictionaries to proper config objects and validate"""
        # Check if we're using K8s mode
        if "use_k8s" in self.eval_config and self.eval_config["use_k8s"]:
            # Create K8s config if not already present
            if not self.k8s_config:
                # Extract K8s config from eval_config
                model = self.eval_config.get("model", "hf")
                task_names = self.eval_config.get("task_names", ["unfair_tos"])
                namespace = self.eval_config.get("k8s_namespace", "default")
                log_samples = self.eval_config.get("log_samples", True)

                # Extract model args
                model_args = []
                if "model_args" in self.eval_config:
                    model_args = self.eval_config["model_args"]
                elif "pretrained" in self.eval_config:
                    model_args = [{"name": "pretrained", "value": self.eval_config["pretrained"]}]

                # Extract environment variables
                env_vars = []
                if "env_vars" in self.eval_config:
                    env_vars = self.eval_config["env_vars"]
                elif "hf_token" in self.eval_config:
                    env_vars = [{"name": "HF_TOKEN", "value": self.eval_config["hf_token"]}]

                self.k8s_config = K8sLMEvalConfig(
                    model=model,
                    model_args=model_args,
                    task_list={"taskNames": task_names},
                    log_samples=log_samples,
                    namespace=namespace,
                    env_vars=env_vars,
                )

            return

    def to_k8s_cr(self) -> Dict[str, Any]:
        """Convert configuration to Kubernetes CR format"""
        if not self.k8s_config:
            raise ValueError("K8s configuration is not set")

        cr = {
            "apiVersion": "trustyai.opendatahub.io/v1alpha1",
            "kind": "LMEvalJob",
            "metadata": {
                "name": f"evaljob-{self.k8s_config.model.lower()}-{hash(str(self.k8s_config.task_list)) % 10000:04d}"
            },
            "spec": {
                "model": self.k8s_config.model,
                "modelArgs": self.k8s_config.model_args,
                "taskList": self.k8s_config.task_list,
                "logSamples": self.k8s_config.log_samples,
                "pod": {"container": {"env": self.k8s_config.env_vars or []}},
            },
        }

        return cr

    @classmethod
    def from_k8s_cr(cls, cr: Dict[str, Any]) -> "LMEvalEvalProviderConfig":
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

        # Create K8s config
        k8s_config = K8sLMEvalConfig(
            model=model,
            model_args=model_args,
            task_list=task_list,
            log_samples=log_samples,
            namespace=namespace,
            env_vars=env_vars,
        )

        # Create provider config with K8s mode enabled
        return cls(eval_config={"use_k8s": True}, k8s_config=k8s_config)