"""Resumen/anotaciones con un LLM local (Ollama). Nada sale del equipo."""
import json
import os
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.environ.get("TRANSCRIPTOR_LLM", "qwen3:14b")
CHUNK_CHARS = 12_000  # ~3k tokens; los textos más largos se resumen por partes
NUM_CTX = 16_384

MODES = {
    "Resumen": "Escribe un resumen claro y conciso (uno a tres párrafos).",
    "Puntos clave": "Extrae los puntos clave como lista con viñetas, del más al menos importante.",
    "Tareas y decisiones": (
        "Extrae dos listas: 'Decisiones' y 'Tareas' (con responsable y fecha solo si se "
        "mencionan explícitamente). Si no hay ninguna, escribe 'Ninguna mencionada'."
    ),
    "Acta de reunión": (
        "Redacta un acta con las secciones: Tema, Resumen, Puntos tratados, Decisiones, "
        "Tareas pendientes. Omite las secciones sin contenido."
    ),
}

_SYSTEM = (
    "Eres un asistente que trabaja sobre la transcripción automática de un audio. "
    "Responde en el mismo idioma de la transcripción. Usa SOLO lo que dice el texto: "
    "no inventes nombres, cifras, fechas ni acuerdos. Si algo es ilegible o dudoso por "
    "errores de transcripción, dilo en vez de adivinar."
)


class LLMError(RuntimeError):
    pass


def _chat(prompt: str) -> str:
    body = {
        "model": LLM_MODEL,
        "stream": False,
        "think": False,  # qwen3: sin cadena de pensamiento, respuesta directa
        "options": {"num_ctx": NUM_CTX, "temperature": 0.2},
        "messages": [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        OLLAMA_URL + "/api/chat", json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)["message"]["content"].strip()
    except OSError as exc:
        raise LLMError(
            f"No se pudo contactar a Ollama en {OLLAMA_URL} ({exc}). "
            f"¿Está corriendo y existe el modelo '{LLM_MODEL}'?"
        ) from exc


def split_chunks(text: str, size: int | None = None) -> list[str]:
    """Parte por líneas (segmentos) sin cortar frases a la mitad."""
    size = size or CHUNK_CHARS
    chunks, cur = [], ""
    for line in text.splitlines():
        while len(line) > size:  # línea gigante: corte duro
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(line[:size])
            line = line[size:]
        if len(cur) + len(line) + 1 > size and cur:
            chunks.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        chunks.append(cur)
    return chunks


def summarize(text: str, mode: str = "Resumen", on_step=None) -> str:
    if not text.strip():
        raise LLMError("No hay texto para resumir.")
    instruction = MODES[mode]
    chunks = split_chunks(text)
    if len(chunks) == 1:
        return _chat(f"{instruction}\n\nTranscripción:\n{chunks[0]}")
    partials = []
    for i, ch in enumerate(chunks, 1):
        if on_step:
            on_step(i, len(chunks))
        partials.append(_chat(
            f"Esta es la parte {i} de {len(chunks)} de una transcripción. Resume fielmente "
            f"todos sus datos relevantes (temas, decisiones, tareas, nombres).\n\n{ch}"))
    joined = "\n\n".join(f"[Parte {i}]\n{p}" for i, p in enumerate(partials, 1))
    return _chat(f"{instruction}\n\nNotas parciales de una transcripción larga:\n{joined}")
