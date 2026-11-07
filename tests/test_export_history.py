from transcriptor.export import Segment, format_timestamp, to_srt, to_txt, write_exports
from transcriptor.history import History

SEGS = [Segment(0.0, 1.5, " Hola mundo "), Segment(3723.456, 3725.0, "Fin"), Segment(4, 5, "  ")]


def test_timestamp():
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(3723.456) == "01:02:03,456"
    assert format_timestamp(59.9996) == "00:01:00,000"


def test_srt_skips_empty_and_numbers_sequentially():
    srt = to_srt(SEGS)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,500\nHola mundo\n\n2\n01:02:03,456")
    assert "3\n" not in srt


def test_txt():
    assert to_txt(SEGS) == "Hola mundo\nFin\n"


def test_write_exports(tmp_path):
    txt, srt = write_exports(SEGS, tmp_path / "audio")
    assert txt.name == "audio.txt" and srt.name == "audio.srt"
    assert "Hola mundo" in txt.read_text(encoding="utf-8")


def test_history_roundtrip(tmp_path):
    h = History(tmp_path / "h.db")
    i = h.add("a.mp3", "small", "es", 12.0, SEGS)
    assert h.list()[0][:3] == (i, h.list()[0][1], "a.mp3")
    assert h.get(i)[0] == "Hola mundo\nFin\n"
    assert h.get(999) is None


def test_collapse_repetitions():
    from transcriptor.export import collapse_repetitions
    loop = "Los miembros son una persona de la Comisión, " * 4 + "y el ministro."
    assert collapse_repetitions(loop) == "Los miembros son una persona de la Comisión, y el ministro."
    assert collapse_repetitions("No, no. Sí, sí.") == "No, no. Sí, sí."  # repetición normal se conserva


def test_document_paragraphs_and_header():
    from transcriptor.export import to_document
    segs = [Segment(0, 30, "Primera idea. " * 20), Segment(30, 60, "Segunda idea."), Segment(70, 90, "Tras pausa.")]
    doc = to_document(segs, "clase.m4a", {"Idioma": "es"})
    assert doc.startswith("TRANSCRIPCIÓN: clase.m4a\nIdioma: es\n")
    assert "[00:00]" in doc and "[01:10]" in doc          # la pausa de 10 s abre párrafo
    assert to_document([Segment(0, 1, " ")]) == ""
