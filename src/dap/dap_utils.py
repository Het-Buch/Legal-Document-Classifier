# src/dap/dap_utils.py
"""
Helpers for Domain-Adaptive Pretraining (DAP)
- check_transformers_installed()
- prepare_tokenizer_and_model()
"""
import logging

logger = logging.getLogger(__name__)

def check_transformers_installed():
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False

def prepare_tokenizer_and_model(model_name_or_path: str = "nlpaueb/legal-bert-base-uncased", use_fast: bool = True):
    """
    Returns (tokenizer, model) ready for MLM (AutoTokenizer, AutoModelForMaskedLM)
    Raises informative exceptions if transformers not available.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForMaskedLM
    except Exception as e:
        raise RuntimeError("transformers not installed. Install transformers and torch to run DAP.") from e

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast)
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
    logger.info("Loaded tokenizer and model from %s", model_name_or_path)
    return tokenizer, model
