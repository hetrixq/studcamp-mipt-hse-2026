from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DeviceName = Literal["auto", "cpu", "cuda", "mps"]


@dataclass(frozen=True)
class RunConfig:
    data_dir: Path
    out_path: Path
    clusters: int = 50
    seed: int = 42

    trim_silence: bool = False
    trim_db: float = 35.0
    min_speech_ms: int = 250

    chunk_sec: float = 3.0
    hop_sec: float = 2.0

    batch_size: int = 8

    device: DeviceName = "auto"
