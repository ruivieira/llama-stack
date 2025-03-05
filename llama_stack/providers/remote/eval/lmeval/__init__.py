import logging
from typing import Any, Dict, Union

from llama_stack.apis.eval import Eval
from llama_stack.providers.remote.eval.lmeval.config import LMEvalEvalProviderConfig

from llama_stack.providers.remote.eval.lmeval.job import LMEval

# Set up logging
logger = logging.getLogger(__name__)

async def get_adapter_impl(
        config: LMEvalEvalProviderConfig,
        _deps: Dict[str, Any] = None,
) -> LMEval:
    """Get appropriate LMEval implementation(s) based on config type.

    Args:
        config: LMEval configuration instance
        _deps: Optional dependencies for testing/injection

    Returns:
        Configured LMEval implementation

    Raises:
        Exception: If configuration is invalid
    """
    try:
        return LMEval(config=config)
    except Exception as e:
        raise Exception(
            f"Failed to create detector implementation: {str(e)}"
        ) from e


__all__ = [
    # Factory methods
    "get_adapter_impl",
    # Configurations
    "LMEvalEvalProviderConfig"
]