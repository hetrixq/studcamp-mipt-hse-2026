from __future__ import annotations

from pathlib import Path


def list_wavs(data_dir: Path) -> list[Path]:
    wavs = [p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav"]
    wavs.sort(key=lambda p: p.name)
    return wavs

