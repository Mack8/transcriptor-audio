"""CLI: transcriptor audio1.mp3 audio2.wav --model large-v3 --lang es"""
import argparse
import sys
from pathlib import Path

from .audio import AUDIO_PROFILES, DEFAULT_AUDIO, DEFAULT_VAD, VAD_PROFILES
from .core import DEFAULT_MODEL, MODELS, Transcriber
from . import glossary
from .export import mark_doubtful, write_exports


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="transcriptor", description="Transcripción local con Whisper")
    p.add_argument("archivos", nargs="+", type=Path)
    p.add_argument("-m", "--model", choices=MODELS, default=DEFAULT_MODEL)
    p.add_argument("-l", "--lang", default=None, help="es, en... (por defecto: autodetección)")
    p.add_argument("--cpu", action="store_true", help="forzar CPU")
    p.add_argument("--audio", choices=list(AUDIO_PROFILES), default=DEFAULT_AUDIO,
                   help="preprocesado de audio (ruido/volumen)")
    p.add_argument("--vad", choices=list(VAD_PROFILES), default=DEFAULT_VAD, help="perfil del filtro de voz")
    p.add_argument("--contexto", default="", help="tema/contexto de esta grabación (se suma al glosario)")
    p.add_argument("--marcar-dudosas", action="store_true", help="marca ⟦palabra?⟧ las de baja confianza (más lento)")
    p.add_argument("-o", "--out", type=Path, default=None, help="carpeta de salida (por defecto junto al audio)")
    args = p.parse_args(argv)

    prompt, dropped = glossary.build_prompt(args.contexto, glossary.load())
    if dropped:
        print(f"[aviso] {dropped} términos del glosario no caben en el contexto de Whisper.", file=sys.stderr)
    tr = Transcriber()
    failed = 0
    for f in args.archivos:
        if not f.is_file():
            print(f"[error] no existe: {f}", file=sys.stderr)
            failed += 1
            continue
        try:
            def prog(x: float, name=f.name) -> None:
                print(f"\r{name}: {x:6.1%}", end="", flush=True)

            res = tr.transcribe(f, args.model, args.lang, args.cpu, prog, args.audio, args.vad,
                               prompt, args.marcar_dudosas)
        except Exception as exc:  # un archivo malo no detiene la cola
            print(f"\n[error] {f}: {exc}", file=sys.stderr)
            failed += 1
            continue
        out_dir = args.out or f.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        txt, srt = write_exports(
            mark_doubtful(res.segments) if args.marcar_dudosas else res.segments, out_dir / f.stem, f.name, res.meta(args.audio, args.vad))
        rtf = res.duration / res.elapsed if res.elapsed else 0
        print(f"\n  {res.device}/{res.model} idioma={res.language} "
              f"audio={res.duration:.0f}s tiempo={res.elapsed:.1f}s ({rtf:.1f}x tiempo real)")
        print(f"  -> {txt}\n  -> {srt}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
