from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.preprocessing import normalize
from tqdm import tqdm

from .audio import resample_if_needed, to_mono, trim_silence_energy
from .clustering import agglomerative_cosine_average
from .config import RunConfig
from .device import pick_device
from .embedding import batch_chunk_and_average_embeddings, load_ecapa_voxceleb_classifier
from .fs import list_wavs
from .seed import set_seeds


def run_pipeline(cfg: RunConfig) -> None:
    set_seeds(cfg.seed)
    device = pick_device(cfg.device)

    wav_paths = list_wavs(cfg.data_dir)
    if not wav_paths:
        raise RuntimeError(f"No .wav files found in: {cfg.data_dir}")

    classifier = load_ecapa_voxceleb_classifier(device=device)

    embeddings: list[np.ndarray] = []
    filenames: list[str] = []

    for i in tqdm(range(0, len(wav_paths), cfg.batch_size), desc="Extract embeddings"):
        batch_paths = wav_paths[i : i + cfg.batch_size]
        batch_wavs: list[torch.Tensor] = []
        batch_names: list[str] = []

        for path in batch_paths:
            wav, sr = torchaudio.load(str(path))
            wav = to_mono(wav)

            if cfg.trim_silence:
                wav = trim_silence_energy(
                    wav,
                    sr,
                    top_db=cfg.trim_db,
                    min_speech_ms=cfg.min_speech_ms,
                )

            wav, _ = resample_if_needed(wav, sr, 16000)
            batch_wavs.append(wav)
            batch_names.append(path.name)

        batch_embs = batch_chunk_and_average_embeddings(
            classifier,
            batch_wavs,
            device=device,
            chunk_sec=cfg.chunk_sec,
            hop_sec=cfg.hop_sec,
            batch_size=cfg.batch_size,
        )
        embeddings.extend(batch_embs)
        filenames.extend(batch_names)

    x = np.vstack(embeddings).astype(np.float32)
    x = normalize(x, norm="l2")

    labels = agglomerative_cosine_average(x, n_clusters=cfg.clusters)

    df = pd.DataFrame({"filename": filenames, "speaker_id": labels})
    df.to_csv(cfg.out_path, index=False)

    logging.info("Saved: %s", cfg.out_path)
    logging.info("%s", df.head(10).to_string(index=False))
