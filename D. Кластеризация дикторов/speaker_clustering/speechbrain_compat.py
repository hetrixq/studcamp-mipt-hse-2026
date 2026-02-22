from __future__ import annotations

_PATCHED = False


def patch_speechbrain_mps_device_type() -> None:
    """
    SpeechBrain 1.x: Pretrained.__init__ sets device_type only for cpu/cuda.
    On macOS MPS it can crash with:
      AttributeError: 'EncoderClassifier' object has no attribute 'device_type'
    This patch extends the logic to also handle 'mps'.
    """
    global _PATCHED
    if _PATCHED:
        return

    import torch
    import speechbrain.inference.interfaces as sb_interfaces
    from speechbrain.dataio.preprocess import AudioNormalizer
    from speechbrain.utils.autocast import AMPConfig, TorchAutocast
    from speechbrain.utils.distributed import infer_device
    from speechbrain.utils.run_opts import RunOptions
    from types import SimpleNamespace

    logger = sb_interfaces.logger

    def patched_init(
        self, modules=None, hparams=None, run_opts=None, freeze_params=True
    ):
        torch.nn.Module.__init__(self)

        if isinstance(run_opts, dict):
            run_opts = RunOptions.from_dictionary(run_opts)

        self.run_opt_defaults = RunOptions()
        for arg, default in self.run_opt_defaults.as_dict().items():
            if run_opts is not None and arg in run_opts.overridden_args:
                setattr(self, arg, run_opts[arg])
            elif hparams is not None and arg in hparams:
                setattr(self, arg, hparams[arg])
            else:
                setattr(self, arg, default)

        if self.device is None:
            self.device = infer_device()

        if self.device == "cpu":
            self.device_type = "cpu"
        elif "cuda" in str(self.device):
            self.device_type = "cuda"
            try:
                _, device_index = str(self.device).split(":")
                torch.cuda.set_device(int(device_index))
            except (ValueError, IndexError, TypeError) as e:
                logger.warning(
                    f"Could not parse CUDA device string '{self.device}': {e}. Falling back to device 0."
                )
                torch.cuda.set_device(0)
        elif "mps" in str(self.device):
            self.device_type = "mps"
        else:
            self.device_type = "cpu"
            self.device = "cpu"

        precision_dtype = AMPConfig.from_name(self.precision).dtype
        self.inference_ctx = TorchAutocast(
            device_type=self.device_type, dtype=precision_dtype
        )

        self.mods = torch.nn.ModuleDict(modules)
        for module in self.mods.values():
            if module is not None:
                module.to(self.device)

        if self.HPARAMS_NEEDED and hparams is None:
            raise ValueError("Need to provide hparams dict.")
        if hparams is not None:
            for hp in self.HPARAMS_NEEDED:
                if hp not in hparams:
                    raise ValueError(f"Need hparams['{hp}']")
            self.hparams = SimpleNamespace(**hparams)

        self._prepare_modules(freeze_params)
        if hparams is None:
            self.audio_normalizer = AudioNormalizer()
        else:
            self.audio_normalizer = hparams.get("audio_normalizer", AudioNormalizer())

    sb_interfaces.Pretrained.__init__ = patched_init
    _PATCHED = True
