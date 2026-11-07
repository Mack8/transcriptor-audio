"""Glosario global de términos (nombres, siglas, jerga) que se pasa a Whisper como contexto."""
import re
from pathlib import Path

from .history import data_dir

# Whisper solo lee ~224 tokens del prompt inicial; en español ~500 caracteres es un tope seguro.
PROMPT_CHARS = 500
_MARKS = re.compile(r"[⟦⟧?¿!¡]")


def glossary_path() -> Path:
    return data_dir() / "glosario.txt"


def parse(text: str) -> list[str]:
    """Un término por línea; ignora vacías y comentarios (#) y duplicados (sin distinguir mayúsculas)."""
    seen, terms = set(), []
    for line in text.splitlines():
        term = line.strip()
        if term and not term.startswith("#") and term.lower() not in seen:
            seen.add(term.lower())
            terms.append(term)
    return terms


def load(path: Path | None = None) -> str:
    p = path or glossary_path()
    return p.read_text(encoding="utf-8") if p.exists() else ""


def save(text: str, path: Path | None = None) -> None:
    (path or glossary_path()).write_text("\n".join(parse(text)) + "\n", encoding="utf-8")


def clean_term(raw: str) -> str:
    """Limpia una selección de la transcripción: quita marcas ⟦?⟧, puntuación de los bordes y espacios."""
    return _MARKS.sub("", raw).strip(" \t\n.,;:()[]\"'«»")


def add_term(text: str, raw: str) -> tuple[str, str]:
    """Añade un término al texto del glosario. Devuelve (texto nuevo, mensaje para el usuario)."""
    term = clean_term(raw)
    if not term:
        return text, "Selecciona primero una palabra o frase en la transcripción."
    if len(term) > 80:
        return text, "La selección es demasiado larga para un término de glosario."
    if term.lower() in {t.lower() for t in parse(text)}:
        return text, f"«{term}» ya está en el glosario."
    return (text.rstrip("\n") + "\n" if text.strip() else "") + term + "\n", f"«{term}» añadido al glosario."


def build_prompt(context: str = "", glossary_text: str = "", limit: int = PROMPT_CHARS) -> tuple[str, int]:
    """Prompt inicial para Whisper y cuántos términos del glosario no cupieron.
    El contexto de la clase va primero; los términos más recientes tienen prioridad."""
    prompt = f"{context.strip().rstrip('.')}." if context.strip() else ""
    terms = parse(glossary_text)[::-1]
    used: list[str] = []
    for term in terms:
        head = f"{prompt} Vocabulario: " if prompt else "Vocabulario: "
        if len(head + ", ".join(used + [term])) > limit:
            break
        used.append(term)
    if used:
        prompt = (f"{prompt} " if prompt else "") + "Vocabulario: " + ", ".join(used) + "."
    return prompt[:limit + 20], len(terms) - len(used)
