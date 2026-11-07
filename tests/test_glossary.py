from transcriptor import glossary
from transcriptor.export import Segment, mark_doubtful


def test_parse_ignora_vacios_comentarios_y_duplicados():
    assert glossary.parse("CONARE\n\n# nota\nconare\nSINAES\n") == ["CONARE", "SINAES"]


def test_add_term_limpia_y_evita_duplicados():
    txt, msg = glossary.add_term("CONARE\n", " ⟦SINAES?⟧, ")
    assert txt == "CONARE\nSINAES\n" and "añadido" in msg
    assert glossary.add_term(txt, "sinaes")[0] == txt
    assert glossary.add_term(txt, "  ")[0] == txt


def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "g.txt"
    glossary.save("A\nB\nA\n", p)
    assert glossary.load(p) == "A\nB\n"


def test_prompt_prioriza_recientes_y_respeta_limite():
    text = "\n".join(f"Termino{i:03d}" for i in range(200))
    prompt, dropped = glossary.build_prompt("Clase de educación", text, limit=120)
    assert prompt.startswith("Clase de educación. Vocabulario: Termino199")
    assert dropped > 0 and len(prompt) <= 140
    assert glossary.build_prompt("", "") == ("", 0)


def test_mark_doubtful_solo_palabras_largas_y_dudosas():
    seg = Segment(0, 1, " Llegamos al casquete", words=((" Llegamos", 0.9), (" al", 0.05), (" casquete", 0.1)))
    assert mark_doubtful([seg])[0].text == "Llegamos al ⟦casquete?⟧"
    plain = Segment(0, 1, "sin palabras")
    assert mark_doubtful([plain])[0] is plain
