"""Núcleo de transcripción: carga de modelo (GPU con fallback a CPU) y generación de segmentos."""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .audio import DEFAULT_AUDIO, DEFAULT_VAD, VAD_PROFILES, load_audio
from .cuda import setup_cuda_libs
from .export import Segment

MODELS = ["tiny", "base", "small", "medium", "large-v3"]
DEFAULT_MODEL = "large-v3"
BATCH_SIZE = 16
LANGUAGES = {"Autodetectar": None, "Español": "es", "English": "en"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma",
              ".mp4", ".mkv", ".webm", ".mov", ".avi"}

ProgressFn = Callable[[float], None]  # fracción 0..1


@dataclass
class Result:
    segments: list[Segment]
    language: str
    language_probability: float
    duration: float          # segundos de audio
    elapsed: float           # segundos de procesamiento
    device: str
    model: str

    def meta(self, audio_profile: str = "", vad_profile: str = "") -> dict[str, str]:
        """Datos de la cabecera del documento exportado."""
        m, sec = divmod(round(self.duration), 60)
        h, m = divmod(m, 60)
        info = {"Fecha": time.strftime("%Y-%m-%d %H:%M"),
                "Duración": f"{h}:{m:02d}:{sec:02d}",
                "Idioma": f"{self.language} ({self.language_probability:.0%})",
                "Modelo": f"{self.model} ({self.device})"}
        if audio_profile or vad_profile:
            info["Audio / VAD"] = f"{audio_profile or '-'} / {vad_profile or '-'}"
        return info


def cuda_available() -> bool:
    setup_cuda_libs()
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


class Transcriber:
    """Mantiene un modelo cargado en memoria y lo reutiliza entre archivos."""

    def __init__(self) -> None:
        self._model = None
        self._batched = None
        self._key: Optional[tuple] = None
        self.device = "cpu"

    def unload(self) -> None:
        """Libera el modelo (y VRAM), p. ej. antes de usar el LLM de resúmenes."""
        self._model = self._batched = self._key = None

    def _load(self, model_name: str, force_cpu: bool):
        setup_cuda_libs()
        from faster_whisper import WhisperModel

        use_gpu = not force_cpu and cuda_available()
        key = (model_name, use_gpu)
        if self._model is not None and self._key == key:
            return self._model
        self._model = self._batched = None  # libera el anterior antes de cargar otro
        if use_gpu:
            try:
                self._model = WhisperModel(model_name, device="cuda", compute_type="float16")
                self.device = "cuda"
            except Exception as exc:  # p. ej. cuDNN ausente o VRAM insuficiente
                print(f"[aviso] GPU no utilizable ({exc}); usando CPU.")
                use_gpu = False
        if self._model is None:
            self._model = WhisperModel(model_name, device="cpu", compute_type="int8")
            self.device = "cpu"
        self._key = (model_name, self.device == "cuda")
        if self.device == "cuda":
            from faster_whisper import BatchedInferencePipeline
            self._batched = BatchedInferencePipeline(self._model)
        return self._model

    def transcribe(
        self,
        path: str | Path,
        model_name: str = DEFAULT_MODEL,
        language: Optional[str] = None,
        force_cpu: bool = False,
        on_progress: Optional[ProgressFn] = None,
        audio_profile: str = DEFAULT_AUDIO,
        vad_profile: str = DEFAULT_VAD,
        context: str = "",
        word_confidence: bool = False,
    ) -> Result:
        model = self._load(model_name, force_cpu)
        t0 = time.perf_counter()
        # Sin filtros se deja que faster-whisper decodifique; con filtros se pasa el array ya limpio.
        audio = str(path) if audio_profile == DEFAULT_AUDIO else load_audio(path, audio_profile)
        vad = VAD_PROFILES[vad_profile]
        vad_kw = {"vad_filter": vad is not None, "vad_parameters": vad or None}
        extra = {"initial_prompt": context or None, "word_timestamps": word_confidence}
        if self._batched is not None:  # GPU: inferencia por lotes, mucho más rápida
            if vad is None:  # el modo por lotes trocea con VAD; sin él usa ventanas fijas
                vad_kw = {"vad_filter": False}
            seg_iter, info = self._batched.transcribe(
                audio, language=language, beam_size=5, batch_size=BATCH_SIZE, **vad_kw, **extra)
        else:
            seg_iter, info = model.transcribe(
                audio, language=language, beam_size=5, **vad_kw, **extra)
        segments: list[Segment] = []
        for s in seg_iter:  # generador perezoso: aquí ocurre la transcripción real
            words = tuple((w.word, w.probability) for w in s.words) if word_confidence and s.words else ()
            segments.append(Segment(s.start, s.end, s.text, words))
            if on_progress and info.duration:
                on_progress(min(s.end / info.duration, 1.0))
        if on_progress:
            on_progress(1.0)
        return Result(
            segments=segments,
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            elapsed=time.perf_counter() - t0,
            device=self.device,
            model=model_name,
        )
