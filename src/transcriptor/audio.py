"""Preprocesado de audio (filtros ffmpeg vía PyAV) y perfiles de VAD."""
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000

# Cadenas de filtros ffmpeg. loudnorm iguala el volumen por tramos (voces lejanas/bajas);
# afftdn resta ruido de fondo estacionario; highpass quita retumbe y ventilación.
# No se usan dynaudnorm/speechnorm: en el build de PyAV actual pasan el audio sin cambios.
AUDIO_PROFILES: dict[str, str] = {
    "Sin procesar": "",
    "Normalizar volumen": "loudnorm=I=-18:TP=-2:LRA=7",
    "Aula (ruido + volumen)": "highpass=f=90,afftdn=nr=12:nf=-30,loudnorm=I=-18:TP=-2:LRA=7",
}
DEFAULT_AUDIO = "Sin procesar"

# Parámetros de Silero VAD. En aula la voz del docente llega baja y con eco: un umbral
# menor evita perder frases, y pausas de silencio más cortas dan segmentos más limpios.
VAD_PROFILES: dict[str, dict] = {
    "Estándar": {},
    "Aula (sensible)": {"threshold": 0.35, "min_silence_duration_ms": 700,
                        "min_speech_duration_ms": 200, "speech_pad_ms": 300},
    "Sin VAD": None,
}
DEFAULT_VAD = "Estándar"


def load_audio(path: str | Path, profile: str = DEFAULT_AUDIO) -> np.ndarray:
    """Decodifica a mono float32 a 16 kHz aplicando el perfil de filtros indicado."""
    import av
    from av.filter import Graph

    filters = AUDIO_PROFILES[profile]
    chunks: list[np.ndarray] = []

    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError("el archivo no contiene pista de audio")
        graph = Graph()
        src = graph.add_abuffer(template=stream)
        chain = [src]
        if filters:
            for spec in filters.split(","):
                name, _, args = spec.partition("=")
                chain.append(graph.add(name, args or None))
        chain.append(graph.add("aformat", f"sample_fmts=flt:sample_rates={SAMPLE_RATE}:channel_layouts=mono"))
        sink = graph.add("abuffersink")
        chain.append(sink)
        for a, b in zip(chain, chain[1:]):
            a.link_to(b)
        graph.configure()

        def drain() -> None:
            while True:
                try:
                    frame = sink.pull()
                except (av.error.BlockingIOError, av.error.EOFError):
                    return
                chunks.append(frame.to_ndarray().reshape(-1))

        for packet in container.demux(stream):
            for frame in packet.decode():
                src.push(frame)
                drain()
        src.push(None)  # vacía los filtros con memoria (loudnorm, afftdn)
        drain()

    if not chunks:
        raise ValueError("no se pudo decodificar audio")
    return np.concatenate(chunks).astype(np.float32)
