from __future__ import annotations

from typing import Tuple

import torch
import torchaudio


def to_mono(wav: torch.Tensor) -> torch.Tensor:
    # wav: [channels, time] or [time]
    if wav.dim() == 1:
        return wav.unsqueeze(0)
    if wav.size(0) == 1:
        return wav
    return wav.mean(dim=0, keepdim=True)


def resample_if_needed(
    wav: torch.Tensor, sr: int, target_sr: int = 16000
) -> Tuple[torch.Tensor, int]:
    if sr == target_sr:
        return wav, sr
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
    return resampler(wav), target_sr


def trim_silence_energy(
    wav: torch.Tensor,
    sr: int,
    *,
    top_db: float = 35.0,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    min_speech_ms: int = 250,
) -> torch.Tensor:
    """
    Simple energy-based trimming:
    - Compute RMS dB per frame
    - Keep frames within top_db of max dB
    - Return contiguous region covering those frames (with a little padding)
    """
    wav = to_mono(wav)
    if wav.numel() == 0:
        return wav

    frame_len = int(sr * frame_ms / 1000.0)
    hop_len = int(sr * hop_ms / 1000.0)
    if frame_len <= 0 or hop_len <= 0 or wav.size(1) < frame_len:
        return wav

    xs = wav.squeeze(0)
    frames = xs.unfold(
        dimension=0, size=frame_len, step=hop_len
    )  # [n_frames, frame_len]
    rms = torch.sqrt(torch.mean(frames * frames, dim=1) + 1e-12)
    db = 20.0 * torch.log10(rms + 1e-12)

    max_db = torch.max(db)
    keep = db >= (max_db - top_db)
    if keep.sum().item() == 0:
        return wav

    idx = torch.where(keep)[0]
    start_frame = int(idx.min().item())
    end_frame = int(idx.max().item())

    start = start_frame * hop_len
    end = end_frame * hop_len + frame_len

    pad = int(0.05 * sr)  # 50ms
    start = max(0, start - pad)
    end = min(wav.size(1), end + pad)

    trimmed = wav[:, start:end]
    min_len = int(sr * (min_speech_ms / 1000.0))
    if trimmed.size(1) < min_len:
        return wav

    return trimmed
