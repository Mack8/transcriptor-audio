import wave

import numpy as np
import pytest

from transcriptor.audio import AUDIO_PROFILES, SAMPLE_RATE, load_audio


@pytest.fixture
def quiet_noisy_wav(tmp_path):
    """Tono de 'voz' muy bajo (44,1 kHz estéreo) + ruido grave: simula un aula."""
    sr, n = 44100, 44100 * 3
    t = np.arange(n) / sr
    x = 0.02 * np.sin(2 * np.pi * 440 * t) + 0.02 * np.sin(2 * np.pi * 40 * t)
    pcm = (np.stack([x, x], 1) * 32767).astype("<i2")
    path = tmp_path / "aula.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


@pytest.mark.parametrize("profile", list(AUDIO_PROFILES))
def test_load_audio_mono_16k(quiet_noisy_wav, profile):
    a = load_audio(quiet_noisy_wav, profile)
    assert a.dtype == np.float32 and a.ndim == 1
    assert abs(len(a) / SAMPLE_RATE - 3) < 0.1


def test_normaliza_volumen(quiet_noisy_wav):
    base = np.abs(load_audio(quiet_noisy_wav, "Sin procesar")).max()
    norm = np.abs(load_audio(quiet_noisy_wav, "Normalizar volumen")).max()
    assert norm > base * 3
