"""Interfaz web local (Gradio). Solo escucha en 127.0.0.1: nada sale del equipo."""
import os
import tempfile
from pathlib import Path

import gradio as gr

from .audio import AUDIO_PROFILES, DEFAULT_AUDIO, DEFAULT_VAD, VAD_PROFILES
from .core import DEFAULT_MODEL, LANGUAGES, MODELS, Transcriber, cuda_available
from .export import mark_doubtful, to_document, to_srt, to_txt
from . import glossary
from .history import History
from .summarize import LLM_MODEL, MODES, LLMError, summarize

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

transcriber = Transcriber()
history = History()
HIST_HEADERS = ["ID", "Fecha", "Archivo", "Modelo", "Idioma", "Duración (s)"]


def _hist_rows():
    return [[i, c, a, m, l or "", round(d or 0)] for i, c, a, m, l, d in history.list()]


def run(files, model, language, force_cpu, audio_profile, vad_profile, context, gloss_text, mark,
        progress=gr.Progress()):
    if not files:
        raise gr.Error("Selecciona al menos un archivo de audio.")
    lang = LANGUAGES[language]
    prompt, _ = glossary.build_prompt(context, gloss_text)
    out_dir = Path(tempfile.mkdtemp(prefix="transcriptor_"))
    texts, exports, status = [], [], []
    n = len(files)
    for i, f in enumerate(files):
        path = Path(f if isinstance(f, str) else f.name)
        try:
            res = transcriber.transcribe(
                path, model, lang, force_cpu,
                audio_profile=audio_profile, vad_profile=vad_profile,
                context=prompt, word_confidence=mark,
                on_progress=lambda x, i=i, name=path.name: progress(
                    (i + x) / n, desc=f"[{i + 1}/{n}] {name}"),
            )
        except Exception as exc:
            status.append(f"❌ {path.name}: {exc}")
            continue
        txt, srt = to_txt(res.segments), to_srt(res.segments)
        doc = to_document(mark_doubtful(res.segments) if mark else res.segments, path.name, res.meta(audio_profile, vad_profile))
        (out_dir / f"{path.stem}.txt").write_text(doc, encoding="utf-8")
        (out_dir / f"{path.stem}.srt").write_text(srt, encoding="utf-8")
        exports += [str(out_dir / f"{path.stem}.txt"), str(out_dir / f"{path.stem}.srt")]
        history.add(path.name, model, res.language, res.duration, res.segments)
        texts.append(f"=== {path.name} ===\n{doc}" if n > 1 else doc)
        status.append(f"✅ {path.name}: {res.duration:.0f}s de audio en {res.elapsed:.1f}s "
                      f"({res.device}, idioma {res.language})")
    return "\n\n".join(texts), exports, "\n".join(status), _hist_rows()


def run_summary(text, mode, progress=gr.Progress()):
    if not text or not text.strip():
        raise gr.Error("Primero transcribe algo (o carga una del historial).")
    transcriber.unload()  # libera VRAM para el LLM
    try:
        return summarize(text, mode, on_step=lambda i, n: progress(
            i / (n + 1), desc=f"Resumiendo parte {i}/{n}"))
    except LLMError as exc:
        raise gr.Error(str(exc))


def save_glossary(text):
    glossary.save(text)
    gr.Info("Glosario guardado.")
    return glossary_info(text)


def glossary_info(text, context=""):
    _, dropped = glossary.build_prompt(context, text)
    n = len(glossary.parse(text))
    msg = f"{n} término{'' if n == 1 else 's'}."
    return msg + (f" ⚠️ Whisper solo usará los más recientes: {dropped} no caben." if dropped else " Caben todos.")


def add_to_glossary(selection, text):
    new_text, msg = glossary.add_term(text, selection or "")
    if new_text != text:
        glossary.save(new_text)
        gr.Info(msg)
    else:
        gr.Warning(msg)
    return new_text, glossary_info(new_text)


def remember_selection(evt: gr.SelectData):
    return evt.value


def load_history(evt: gr.SelectData, rows):
    entry = history.get(int(rows.iloc[evt.index[0], 0]))
    return entry[0] if entry else ""


def build() -> gr.Blocks:
    gpu = cuda_available()
    with gr.Blocks(title="Transcriptor local") as demo:
        gr.Markdown(f"# Transcriptor de audio local\n100 % offline · "
                    f"{'GPU CUDA detectada ✅' if gpu else 'sin GPU: se usará CPU (más lento)'}")
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(label="Arrastra audio/video aquí", file_count="multiple",
                                file_types=["audio", "video"])
                model = gr.Dropdown(MODELS, value=DEFAULT_MODEL, label="Modelo")
                lang = gr.Dropdown(list(LANGUAGES), value="Autodetectar", label="Idioma")
                audio_p = gr.Dropdown(list(AUDIO_PROFILES), value=DEFAULT_AUDIO, label="Audio (ruido/volumen)")
                vad_p = gr.Dropdown(list(VAD_PROFILES), value=DEFAULT_VAD, label="Filtro de voz (VAD)")
                context = gr.Textbox(label="Contexto de esta clase (opcional)", lines=2,
                                     placeholder="Ej.: clase de educación superior en Costa Rica")
                mark = gr.Checkbox(label="Marcar palabras dudosas ⟦?⟧ (más lento)", value=False)
                cpu = gr.Checkbox(label="Forzar CPU", value=False, interactive=gpu)
                btn = gr.Button("Transcribir", variant="primary")
                status = gr.Textbox(label="Estado", lines=4, interactive=False)
            with gr.Column(scale=2):
                out = gr.Textbox(label="Transcripción", lines=18, buttons=["copy"])
                with gr.Row():
                    add_sel = gr.Button("➕ Añadir selección al glosario", scale=2)
                    gr.Markdown("Selecciona con el ratón una palabra en la transcripción y pulsa el botón.")
                dl = gr.File(label="Descargar .txt / .srt", file_count="multiple")
                with gr.Row():
                    mode = gr.Dropdown(list(MODES), value="Resumen", label=f"Anotaciones ({LLM_MODEL})")
                    sbtn = gr.Button("Generar")
                notes = gr.Textbox(label="Anotaciones", lines=10, buttons=["copy"])
        with gr.Accordion("Glosario (nombres, siglas y jerga que Whisper suele fallar)", open=False):
            gloss = gr.Textbox(label="Un término por línea", lines=8, value=glossary.load,
                               placeholder="CONARE\nSINAES\nMinisterio de Educación Pública")
            gloss_info = gr.Markdown(value=lambda: glossary_info(glossary.load()))
            with gr.Row():
                new_term = gr.Textbox(label="Añadir término a mano", scale=3)
                add_btn = gr.Button("Añadir", scale=1)
                save_btn = gr.Button("Guardar cambios", scale=1)
        selection = gr.State("")
        with gr.Accordion("Historial", open=False):
            hist = gr.Dataframe(headers=HIST_HEADERS, value=_hist_rows, interactive=False)
            gr.Markdown("Haz clic en una fila para ver su texto arriba.")
        btn.click(run, [files, model, lang, cpu, audio_p, vad_p, context, gloss, mark],
                  [out, dl, status, hist])
        out.select(remember_selection, None, selection)
        add_sel.click(add_to_glossary, [selection, gloss], [gloss, gloss_info])
        add_btn.click(add_to_glossary, [new_term, gloss], [gloss, gloss_info]).then(lambda: "", None, new_term)
        save_btn.click(save_glossary, gloss, gloss_info)
        sbtn.click(run_summary, [out, mode], notes)
        hist.select(load_history, hist, out)
    return demo


def main() -> None:
    build().queue().launch(server_name="127.0.0.1", inbrowser=True)


if __name__ == "__main__":
    main()
