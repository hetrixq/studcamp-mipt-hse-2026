from __future__ import annotations

import logging
import warnings


class _DropOnlyWarningsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno != logging.WARNING


def _silence_third_party_loggers() -> None:
    for name in (
        "speechbrain",
        "huggingface_hub",
        "httpx",
        "urllib3",
        "torchaudio",
        "torch",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)

    try:
        import huggingface_hub.utils.logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:
        pass


def configure_logging(*, log_level: str, show_warnings: bool, verbose: bool) -> None:
    """
    Default behavior:
      - no Python warnings
      - no logging WARNING
      - third-party loggers silenced
    Enable them explicitly with flags.
    """
    if show_warnings:
        warnings.simplefilter("default")
    else:
        warnings.filterwarnings("ignore")

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(message)s",
    )

    if not show_warnings:
        root = logging.getLogger()
        for handler in root.handlers:
            handler.addFilter(_DropOnlyWarningsFilter())

    if not verbose:
        _silence_third_party_loggers()
