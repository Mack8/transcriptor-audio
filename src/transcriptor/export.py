"""Exportación de segmentos a .txt y .srt."""
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: tuple[tuple[str, float], ...] = ()   # (palabra, probabilidad) si se pidió confianza


def format_timestamp(seconds: float) -> str:
    """Formato SRT: HH:MM:SS,mmm."""
    ms = round(max(seconds, 0.0) * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_txt(segments: list[Segment]) -> str:
    return "\n".join(s.text.strip() for s in segments if s.text.strip()) + "\n"


PARAGRAPH_CHARS = 550     # longitud objetivo de un párrafo (se corta al final de una oración)
PAUSE_BREAK = 2.5         # segundos de silencio entre segmentos que fuerzan párrafo nuevo
_SENT_END = re.compile(r"(?<=[.?!…])\s+")
_PUNCT = re.compile(r"[^\w]+", re.UNICODE)


def format_clock(seconds: float) -> str:
    """mm:ss (o h:mm:ss a partir de la hora) para las marcas de los párrafos."""
    t = int(max(seconds, 0))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def collapse_repetitions(text: str, min_repeats: int = 3) -> str:
    """Whisper a veces entra en bucle y repite la misma frase. Deja una sola copia de
    cualquier tramo (2-15 palabras) repetido `min_repeats` veces o más seguidas."""
    words = text.split()
    norm = [_PUNCT.sub("", w).lower() for w in words]
    out: list[str] = []
    i = 0
    while i < len(words):
        for k in range(2, 16):
            gram = norm[i:i + k]
            if len(gram) < k or not any(gram):
                continue
            reps = 1
            while norm[i + reps * k:i + (reps + 1) * k] == gram:
                reps += 1
            if reps >= min_repeats:
                out.extend(words[i:i + k])
                i += reps * k
                break
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


def _sentences(segments: list[Segment]) -> list[tuple[float, str, bool]]:
    """Oraciones con su hora de inicio (interpolada dentro del segmento) y si van tras una pausa."""
    text, anchors, gaps = "", [], set()
    prev_end = None
    for seg in segments:
        t = seg.text.strip()
        if not t:
            continue
        if prev_end is not None and seg.start - prev_end >= PAUSE_BREAK:
            gaps.add(len(text) + 1)
        anchors.append((len(text) + (1 if text else 0), seg))
        text = f"{text} {t}" if text else t
        prev_end = seg.end
    out, pos, prev_idx = [], 0, -1
    for part in _SENT_END.split(text):
        idx = text.index(part, pos)
        pos = idx + len(part)
        seg_start, seg = next((a for a in reversed(anchors) if a[0] <= idx), anchors[0])
        span = max(len(seg.text.strip()), 1)
        frac = min(max((idx - seg_start) / span, 0), 1)
        out.append((seg.start + frac * (seg.end - seg.start), part, any(prev_idx < g <= idx for g in gaps)))
        prev_idx = idx
    return out


DOUBT_THRESHOLD = 0.25


def mark_doubtful(segments: list[Segment], threshold: float = DOUBT_THRESHOLD) -> list[Segment]:
    """Copia de los segmentos con ⟦palabra?⟧ en las palabras de baja probabilidad.
    Ignora palabras de <4 letras (artículos, preposiciones), que casi siempre dudan sin importar."""
    out = []
    for seg in segments:
        if not seg.words:
            out.append(seg)
            continue
        parts = []
        for word, prob in seg.words:
            core = word.strip().strip(".,;:¿?¡!")
            parts.append(f" ⟦{word.strip()}?⟧" if prob < threshold and len(core) >= 4 and word.startswith(" ")
                         else word)
        out.append(Segment(seg.start, seg.end, "".join(parts).strip()))
    return out


def to_document(segments: list[Segment], title: str = "", meta: dict[str, str] | None = None) -> str:
    """Transcripción legible: cabecera, párrafos separados y marca [mm:ss] al inicio de cada uno."""
    if not any(s.text.strip() for s in segments):
        return ""
    paragraphs: list[tuple[float, str]] = []
    cur_t, cur = 0.0, ""
    for start, sentence, after_gap in _sentences(segments):
        if cur and (len(cur) >= PARAGRAPH_CHARS or after_gap):
            paragraphs.append((cur_t, cur))
            cur = ""
        if not cur:
            cur_t = start
        cur = f"{cur} {sentence}" if cur else sentence
    if cur:
        paragraphs.append((cur_t, cur))
    lines = []
    if title or meta:
        lines.append(f"TRANSCRIPCIÓN{': ' + title if title else ''}")
        lines += [f"{k}: {v}" for k, v in (meta or {}).items()]
        lines += ["=" * 60, ""]
    for t, para in paragraphs:
        lines += [f"[{format_clock(t)}]", collapse_repetitions(para), ""]
    return "\n".join(lines)


def to_srt(segments: list[Segment]) -> str:
    blocks = []
    n = 0
    for s in segments:
        text = s.text.strip()
        if not text:
            continue
        n += 1
        blocks.append(f"{n}\n{format_timestamp(s.start)} --> {format_timestamp(s.end)}\n{text}\n")
    return "\n".join(blocks)


def write_exports(segments: list[Segment], base_path: Path, title: str = "",
                  meta: dict[str, str] | None = None) -> tuple[Path, Path]:
    """Escribe <base>.txt (documento ordenado) y <base>.srt (UTF-8) y devuelve sus rutas."""
    txt, srt = base_path.with_suffix(".txt"), base_path.with_suffix(".srt")
    txt.write_text(to_document(segments, title, meta), encoding="utf-8")
    srt.write_text(to_srt(segments), encoding="utf-8")
    return txt, srt
