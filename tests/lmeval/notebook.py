# %%
from __future__ import annotations

import logging

from llama_stack.apis.eval import BenchmarkConfig, ModelCandidate
from llama_stack.apis.inference import SamplingParams, SystemMessage
from llama_stack.providers.remote.eval.lmeval import LMEvalEvalProviderConfig, get_adapter_impl

# Configure logging with better formatting
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
# %%
config = LMEvalEvalProviderConfig(use_k8s=True)

# Initialize LMEval
lmeval = await get_adapter_impl(config)

# Create benchmark config
benchmark_config = BenchmarkConfig(
    eval_candidate=ModelCandidate(
        type="model",
        model="granite",
        sampling_params=SamplingParams(
            temperature=0.0,
            top_p=0.95,
            max_tokens=1024,
        ),
        system_message=SystemMessage(role="system", content="You are a helpful assistant."),
    ),
    env_vars=[{"name": "OPENAI_API_KEY", "valueFrom": {"secretKeyRef": {"name": "api-key-secret", "key": "token"}}}],
)

# Run evaluation
job = await lmeval.run_eval("mmlu", benchmark_config)
print(f"Started job: {job.job_id}")

# Get job status
status = await lmeval.job_status("mmlu", job.job_id)
print(f"Job status: {status}")

# Get job results
results = await lmeval.job_result("mmlu", job.job_id)
print(f"Job results: {results}")
