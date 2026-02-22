from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .audio import to_mono
from .speechbrain_compat import patch_speechbrain_mps_device_type


def load_ecapa_voxceleb_classifier(*, device: str) -> Any:
    """
    Lazily imports SpeechBrain and loads the ECAPA-TDNN (VoxCeleb) encoder.
    NOTE: first run downloads weights (needs internet once).
    """
    patch_speechbrain_mps_device_type()
    from speechbrain.pretrained import EncoderClassifier

    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )


def _iter_chunks(
    wav_16k: torch.Tensor,
    *,
    chunk_sec: float,
    hop_sec: float,
) -> list[torch.Tensor]:
    sr = 16000
    wav_16k = to_mono(wav_16k)
    t = wav_16k.size(1)

    chunk = int(chunk_sec * sr)
    hop = int(hop_sec * sr)

    if t <= chunk:
        return [wav_16k]

    segments: list[torch.Tensor] = []
    for start in range(0, max(1, t - chunk + 1), hop):
        end = min(t, start + chunk)
        segment = wav_16k[:, start:end]
        if segment.size(1) < int(0.8 * chunk):
            continue
        segments.append(segment)

    return segments or [wav_16k]


def _pad_batch(segments: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      wavs: [B, T]
      wav_lens: [B] relative lengths in (0, 1]
    """
    if not segments:
        raise ValueError("No segments to pad")

    lengths = torch.tensor([s.size(1) for s in segments], dtype=torch.long)
    max_len = int(lengths.max().item())
    batch = torch.zeros((len(segments), max_len), dtype=segments[0].dtype)
    for i, seg in enumerate(segments):
        batch[i, : seg.size(1)] = seg.squeeze(0)
    wav_lens = lengths.to(torch.float32) / float(max_len)
    return batch, wav_lens


def embed_batch(
    classifier: Any,
    segments: list[torch.Tensor],
    *,
    device: str,
) -> np.ndarray:
    """
    segments: list of [1, T] tensors at 16kHz.
    returns: [B, D] numpy array.
    """
    wavs, wav_lens = _pad_batch(segments)
    wavs = wavs.to(device)
    wav_lens = wav_lens.to(device)

    with torch.inference_mode():
        try:
            embs = classifier.encode_batch(wavs, wav_lens=wav_lens)
        except TypeError:
            embs = classifier.encode_batch(wavs, wav_lens)

    embs = embs.detach().cpu()
    if embs.dim() == 3 and embs.size(1) == 1:
        embs = embs[:, 0, :]
    return embs.numpy()


def chunk_and_average_embedding(
    classifier: Any,
    wav_16k: torch.Tensor,
    *,
    device: str,
    chunk_sec: float = 3.0,
    hop_sec: float = 2.0,
) -> np.ndarray:
    segments = _iter_chunks(wav_16k, chunk_sec=chunk_sec, hop_sec=hop_sec)
    embs = embed_batch(classifier, segments, device=device)
    return np.mean(embs, axis=0)


def batch_chunk_and_average_embeddings(
    classifier: Any,
    wavs_16k: list[torch.Tensor],
    *,
    device: str,
    chunk_sec: float = 3.0,
    hop_sec: float = 2.0,
    batch_size: int = 8,
) -> list[np.ndarray]:
    """
    Computes one embedding per wav by averaging embeddings over overlapping chunks.
    Uses batching for better device utilization (esp. MPS/CUDA).
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1")

    per_file_segments: list[list[torch.Tensor]] = [
        _iter_chunks(w, chunk_sec=chunk_sec, hop_sec=hop_sec) for w in wavs_16k
    ]

    flat_segments: list[torch.Tensor] = []
    owner: list[int] = []
    for file_idx, segs in enumerate(per_file_segments):
        for s in segs:
            flat_segments.append(s)
            owner.append(file_idx)

    if not flat_segments:
        return []

    sums: list[np.ndarray | None] = [None] * len(wavs_16k)
    counts: list[int] = [0] * len(wavs_16k)

    for i in range(0, len(flat_segments), batch_size):
        seg_batch = flat_segments[i : i + batch_size]
        owners_batch = owner[i : i + batch_size]
        embs = embed_batch(classifier, seg_batch, device=device)  # [B, D]

        for j, file_idx in enumerate(owners_batch):
            if sums[file_idx] is None:
                sums[file_idx] = embs[j].astype(np.float64)
            else:
                sums[file_idx] = sums[file_idx] + embs[j]
            counts[file_idx] += 1

    out: list[np.ndarray] = []
    for s, c in zip(sums, counts):
        if s is None or c == 0:
            raise RuntimeError("Failed to compute embeddings for at least one input file")
        out.append((s / float(c)).astype(np.float32))
    return out
