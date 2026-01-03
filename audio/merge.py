from pathlib import Path
import soundfile as sf
import numpy as np


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr[:, None]
    return arr


def concat_wavs(wav_paths, out_path, crossfade_ms: int = 500, gap_ms: int = 200, target_sr: int | None = None):
    """Concatenate WAV files with optional short crossfade or gap using soundfile + numpy.

    - Ensures all files use the same sample rate (resamples if `target_sr` provided),
      otherwise requires all inputs to share the same sample rate.
    - `crossfade_ms` is the crossfade duration in milliseconds between consecutive scenes.
    - `gap_ms` inserts a short silence gap (ms) between scenes. If `gap_ms` > 0,
      crossfade is disabled and a silence of `gap_ms` is inserted between clips.
    """
    if not wav_paths:
        raise ValueError("No wav paths provided")

    combined = None
    sr = None

    for p in wav_paths:
        data, this_sr = sf.read(p, always_2d=True)
        data = data.astype(np.float32)

        if sr is None:
            sr = this_sr
        if target_sr is not None and this_sr != target_sr:
            # simple linear resample
            import math

            ratio = target_sr / this_sr
            new_len = int(math.ceil(data.shape[0] * ratio))
            x_old = np.linspace(0, 1, data.shape[0])
            x_new = np.linspace(0, 1, new_len)
            data = np.stack([np.interp(x_new, x_old, data[:, ch]) for ch in range(data.shape[1])], axis=1)
            this_sr = target_sr
        elif target_sr is None and this_sr != sr:
            raise RuntimeError(f"Sample rate mismatch: {this_sr} != {sr}. Provide target_sr to resample.")

        data = _ensure_2d(data)

        if combined is None:
            combined = data
        else:
            # either insert short silence gap (preferred) or apply crossfade
            if gap_ms and gap_ms > 0:
                gap_samples = int(sr * (gap_ms / 1000.0))
                if gap_samples > 0:
                    silence = np.zeros((gap_samples, data.shape[1]), dtype=np.float32)
                    combined = np.concatenate([combined, silence, data], axis=0)
                else:
                    combined = np.concatenate([combined, data], axis=0)
            else:
                # apply crossfade
                crossfade_samples = int(sr * (crossfade_ms / 1000.0))
                if crossfade_samples <= 0:
                    combined = np.concatenate([combined, data], axis=0)
                else:
                    cf = min(crossfade_samples, combined.shape[0], data.shape[0])
                    if cf == 0:
                        combined = np.concatenate([combined, data], axis=0)
                    else:
                        a_tail = combined[-cf:]
                        b_head = data[:cf]
                        ramp = np.linspace(1.0, 0.0, cf, dtype=np.float32)[:, None]
                        blended = a_tail * ramp + b_head * (1.0 - ramp)
                        combined = np.concatenate([combined[:-cf], blended, data[cf:]], axis=0)

    # write out
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), combined, sr)
    return str(out_path)
