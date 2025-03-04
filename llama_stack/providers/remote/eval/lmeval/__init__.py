import logging
from typing import Dict, Optional

from llama_stack.apis.eval import Eval
from llama_stack.distribution.datatypes import Api, ProviderSpec
from llama_stack.providers.remote.eval.lmeval.config import LMEvalEvalProviderConfig
from llama_stack.apis.inference import Inference
from llama_stack.providers.remote.eval.lmeval.eval import LMEval

# Set up logging
logger = logging.getLogger(__name__)

async def get_adapter_impl(
        config: LMEvalEvalProviderConfig,
        deps: Optional[Dict[Api, ProviderSpec]] = None,
) -> LMEval:
    """Get an LMEval implementation from the configuration.

    Args:
        config: LMEval configuration
        deps: Optional dependencies for testing/injection

    Returns:
        Configured LMEval implementation

    Raises:
        Exception: If configuration is invalid
    """
    try:
        if deps is None:
            deps = {}
        
        
        # Extract base_url from config if available
        base_url = None
        if hasattr(config, 'model_args') and config.model_args:
            for arg in config.model_args:
                if arg.get('name') == 'base_url':
                    base_url = arg.get('value')
                    logger.info(f"Using base_url from config: {base_url}")
                    break
        
        return LMEval(config=config)
    except Exception as e:
        raise Exception(
            f"Failed to create LMEval implementation: {str(e)}"
        ) from e


__all__ = [
    # Factory methods
    "get_adapter_impl",
    # Configurations
    "LMEvalEvalProviderConfig"
]