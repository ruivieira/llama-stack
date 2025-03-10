from unittest.mock import patch

import pytest

from llama_stack.apis.eval import BenchmarkConfig, ModelCandidate
from llama_stack.apis.inference import SamplingParams
from llama_stack.providers.remote.eval.lmeval.config import LMEvalEvalProviderConfig
from llama_stack.providers.remote.eval.lmeval.errors import LMEvalConfigError
from llama_stack.providers.remote.eval.lmeval.job import LMEval


class TestLMEvalEvalProviderConfig:
    def test_use_k8s_default(self):
        """Test that use_k8s defaults to False"""
        config = LMEvalEvalProviderConfig()
        assert config.use_k8s is True

    def test_use_k8s_true(self):
        """Test that use_k8s can be set to True"""
        config = LMEvalEvalProviderConfig(use_k8s=True)
        assert config.use_k8s is True

    def test_use_k8s_validation(self):
        """Test that use_k8s must be a boolean"""
        with pytest.raises(LMEvalConfigError, match="use_k8s must be a boolean"):
            LMEvalEvalProviderConfig(use_k8s="True")

    def test_use_k8s_required(self):
        """Test that use_k8s must be True for now"""
        with pytest.raises(LMEvalConfigError, match="Only Kubernetes LMEval backend is supported at the moment"):
            config = LMEvalEvalProviderConfig(use_k8s=False)
            config.__post_init__()


class TestLMEvalJob:
    @pytest.fixture
    def lmeval_instance(self):
        """Create a basic LMEval instance for testing"""
        config = LMEvalEvalProviderConfig(use_k8s=True)
        instance = LMEval(config=config)
        instance.use_k8s = True
        return instance

    @pytest.mark.asyncio
    async def test_create_lmeval_cr_with_builtin_task(self, lmeval_instance):
        """Test that builtin task names are correctly used as benchmark_id"""
        builtin_tasks = ["mmlu", "hellaswag", "arc_easy", "arc_challenge", "truthfulqa"]

        benchmark_config = BenchmarkConfig(
            eval_candidate=ModelCandidate(
                type="model",
                model="test-model",
                sampling_params=SamplingParams(max_tokens=100),
            )
        )

        for task in builtin_tasks:
            cr = lmeval_instance._create_lmeval_cr(task, benchmark_config)

            assert cr["spec"]["taskList"]["taskNames"] == [task]
            assert task.lower() in cr["metadata"]["name"]

    @pytest.mark.asyncio
    async def test_run_eval_with_builtin_task(self, lmeval_instance):
        """Test that run_eval works with builtin task names"""
        benchmark_config = BenchmarkConfig(
            eval_candidate=ModelCandidate(
                type="model",
                model="test-model",
                sampling_params=SamplingParams(max_tokens=100),
            )
        )

        with patch.object(lmeval_instance, "_create_lmeval_cr") as mock_create_cr:
            mock_create_cr.return_value = {"mock": "cr"}
            job = await lmeval_instance.run_eval("mmlu", benchmark_config)
            mock_create_cr.assert_called_once_with("mmlu", benchmark_config)
            assert job.job_id is not None

    @pytest.mark.asyncio
    async def test_run_eval_with_custom_task(self, lmeval_instance):
        """Test that run_eval works with custom task names"""
        benchmark_config = BenchmarkConfig(
            eval_candidate=ModelCandidate(
                type="model",
                model="test-model",
                sampling_params=SamplingParams(max_tokens=100),
            )
        )

        with patch.object(lmeval_instance, "_create_lmeval_cr") as mock_create_cr:
            mock_create_cr.return_value = {"mock": "cr"}

            job = await lmeval_instance.run_eval("custom_benchmark", benchmark_config)

            mock_create_cr.assert_called_once_with("custom_benchmark", benchmark_config)

            assert job.job_id is not None

    @pytest.mark.asyncio
    async def test_model_args_in_cr(self, lmeval_instance):
        """Test that model args are correctly included in the CR"""
        benchmark_config = BenchmarkConfig(
            eval_candidate=ModelCandidate(
                type="model",
                model="openai/gpt-4",
                sampling_params=SamplingParams(
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=2048,
                ),
            )
        )

        cr = lmeval_instance._create_lmeval_cr("mmlu", benchmark_config)

        model_args = cr["spec"]["modelArgs"]
        model_name_arg = next((arg for arg in model_args if arg["name"] == "model"), None)
        assert model_name_arg is not None
        assert model_name_arg["value"] == "openai/gpt-4"
