from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import RunConfig
from .logging_setup import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Speaker clustering -> submission.csv")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Folder with .wav files"
    )
    parser.add_argument(
        "--out", type=str, default="submission.csv", help="Output CSV path"
    )
    parser.add_argument(
        "--clusters", type=int, default=50, help="Number of speaker clusters (M)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument(
        "--trim_silence", type=int, default=0, help="1 to enable energy-based trimming"
    )
    parser.add_argument(
        "--trim_db",
        type=float,
        default=35.0,
        help="Trim threshold: keep within top_db of max",
    )
    parser.add_argument(
        "--min_speech_ms",
        type=int,
        default=250,
        help="Minimum speech length after trimming",
    )

    parser.add_argument(
        "--chunk_sec",
        type=float,
        default=3.0,
        help="Chunk length (sec) for embedding averaging",
    )
    parser.add_argument(
        "--hop_sec",
        type=float,
        default=2.0,
        help="Hop length (sec) for embedding averaging",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for model forward (files/chunks); try 8-16 on MPS",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="auto/cpu/cuda/mps",
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO", help="DEBUG/INFO/WARNING/ERROR"
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="1 to show noisy third-party logs (HF/SpeechBrain/etc.)",
    )
    parser.add_argument(
        "--warnings",
        type=int,
        default=0,
        help="1 to show warnings (default: hidden)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(
        log_level=args.log_level,
        show_warnings=(args.warnings == 1),
        verbose=(args.verbose == 1),
    )

    cfg = RunConfig(
        data_dir=Path(args.data_dir),
        out_path=Path(args.out),
        clusters=args.clusters,
        seed=args.seed,
        trim_silence=(args.trim_silence == 1),
        trim_db=args.trim_db,
        min_speech_ms=args.min_speech_ms,
        chunk_sec=args.chunk_sec,
        hop_sec=args.hop_sec,
        batch_size=args.batch_size,
        device=args.device,
    )

    from .pipeline import run_pipeline

    run_pipeline(cfg)
    return 0
