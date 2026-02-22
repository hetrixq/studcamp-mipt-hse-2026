from __future__ import annotations
import logging
import torch


def pick_device(requested: str) -> str:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    device = requested
    if device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available; falling back to cpu")
        return "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        logging.warning("MPS not available; falling back to cpu")
        return "cpu"
    return device
