import pytest

from transcriptor import summarize as sm


def test_split_chunks_respects_size_and_keeps_text():
    text = "\n".join(f"linea {i} " + "x" * 50 for i in range(200))
    chunks = sm.split_chunks(text, 1000)
    assert all(len(c) <= 1001 for c in chunks) and len(chunks) > 1
    assert "".join(chunks).replace("\n", "") == text.replace("\n", "")


def test_split_chunks_giant_line():
    assert all(len(c) <= 100 for c in sm.split_chunks("a" * 350, 100))


def test_single_chunk_one_call(monkeypatch):
    calls = []
    monkeypatch.setattr(sm, "_chat", lambda p: calls.append(p) or "ok")
    assert sm.summarize("hola", "Resumen") == "ok" and len(calls) == 1


def test_long_text_map_reduce(monkeypatch):
    calls = []
    monkeypatch.setattr(sm, "_chat", lambda p: calls.append(p) or "parcial")
    monkeypatch.setattr(sm, "CHUNK_CHARS", 100)
    sm.summarize("\n".join(["y" * 60] * 6), "Puntos clave")
    assert len(calls) > 2 and "Notas parciales" in calls[-1]


def test_empty_text():
    with pytest.raises(sm.LLMError):
        sm.summarize("  ")
