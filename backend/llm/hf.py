import logging
from builtins import Exception

import torch
from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


class ModelError(Exception):
    """Raised when there's an error with the model operations"""

    pass


# Hugging Face Transformers pipeline
def load_local_model(
    model_id: str = "meta-llama/Llama-3.2-3B",
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> pipeline:
    try:
        logger.info("Loading local model...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        llama_pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        logger.info("Local model loaded successfully")
        return llama_pipe, tokenizer
    except Exception as e:
        logger.error(f"Failed to load local model: {str(e)}")
        raise ModelError(f"Failed to load local model: {str(e)}")
