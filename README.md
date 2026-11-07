# Local audio transcriber / Transcriptor de audio local

**English:** 100% local speech-to-text built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper). No audio or text leaves your machine. It runs on an NVIDIA GPU (CUDA, float16) and falls back to CPU (int8). It ships a local web UI (Gradio, bound to `127.0.0.1`) and a CLI, exports `.txt` and `.srt`, keeps a SQLite history, supports a custom glossary for names and jargon, noise/echo-tolerant audio profiles for classrooms, and optional summaries and meeting minutes through a local LLM (Ollama). Batched GPU inference measured about 98x real time on an RTX 5060 Ti (synthetic clip, see "Rendimiento medido"). The detailed documentation below is in Spanish.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m transcriptor.cli audio.mp3 -m large-v3 -l es -o out/
```

---

Transcribe audio/video a texto con Whisper (`faster-whisper`), 100 % local: ningún
audio ni texto sale del equipo. Usa la GPU NVIDIA (CUDA, float16) y cae a CPU (int8) si
no está disponible. Exporta `.txt` y `.srt`.

## Instalación

**Windows** (destino principal): instala Python 3.11–3.13 y el driver NVIDIA, luego:

```bat
setup.bat
```

**Linux / WSL2** (probado aquí, Python 3.12 con uv):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

cuBLAS y cuDNN llegan como paquetes pip (`nvidia-*-cu12`); no hace falta instalar el
CUDA Toolkit. `ffmpeg` no es necesario: `faster-whisper` decodifica con PyAV.

## Uso

Interfaz web (solo escucha en `127.0.0.1`): `run.bat` o `python -m transcriptor.app`.
Arrastra uno o varios archivos, elige modelo/idioma, pulsa *Transcribir*. Los archivos
se procesan en cola; hay botón de copiar, descarga de `.txt`/`.srt` e historial.

CLI:

```bash
python -m transcriptor.cli reunion.mp3 nota.m4a -m large-v3 -l es -o salida/
```

## Audio con ruido o eco (aulas)

Dos selectores nuevos (UI y CLI: `--audio`, `--vad`):

| Audio | Qué hace |
| --- | --- |
| Sin procesar | Comportamiento anterior (por defecto) |
| Normalizar volumen | `loudnorm`: iguala el volumen por tramos, sube voces lejanas o bajas |
| Aula (ruido + volumen) | Filtro paso alto 90 Hz (ventilación, retumbe) + reducción de ruido `afftdn` + `loudnorm` |

| VAD | Parámetros |
| --- | --- |
| Estándar | Valores por defecto de faster-whisper |
| Aula (sensible) | umbral 0,35, silencio mínimo 700 ms, habla mínima 200 ms, relleno 300 ms |
| Sin VAD | Desactivado (en GPU trocea en ventanas fijas) |

Se procesa en memoria; el original no se toca. Ninguno de los filtros elimina el **eco**
(reverberación): el paso alto y la normalización solo ayudan de forma indirecta. En audio ya
limpio el perfil *Aula* puede degradar levemente el resultado; compara con tu grabación real.

## Glosario y palabras dudosas

Whisper falla en nombres propios, siglas y jerga. El **glosario** es un archivo global
(`~/.transcriptor-audio/glosario.txt`, un término por línea) que se envía a Whisper como
contexto en cada transcripción. Crece con el uso:

- En la UI: seleccionas una palabra en la transcripción y pulsas *➕ Añadir selección al
  glosario*, o lo escribes a mano en el panel *Glosario* (que también permite editarlo).
- El campo *Contexto de esta clase* (CLI: `--contexto`) suma el tema de esa grabación solo para esa ejecución.
- Límite: Whisper lee ~500 caracteres. Si no caben todos, se usan primero los términos más recientes y la UI avisa.
- Mantenlo corto: solo lo que realmente falla. No corrige errores acústicos (frases mal oídas).

*Marcar palabras dudosas* (CLI: `--marcar-dudosas`) escribe `⟦palabra?⟧` en el `.txt` cuando
Whisper tenía poca confianza (probabilidad < 0,25, palabras de 4+ letras). Es más lento.
Whisper también se equivoca con seguridad, así que las marcas no detectan todos los errores.

## Anotaciones con LLM local (opcional)

Tras transcribir, el panel *Anotaciones* genera con `qwen3:14b` (vía Ollama) un
**Resumen**, **Puntos clave**, **Tareas y decisiones** o un **Acta de reunión**. Requiere
Ollama corriendo con el modelo (`ollama pull qwen3:14b`). Todo local.

- Cambia modelo o servidor con `TRANSCRIPTOR_LLM` y `OLLAMA_URL`.
- Textos largos (>12 000 caracteres) se resumen por partes y luego se combinan.
- Antes de resumir se descarga Whisper de la VRAM para dejar sitio al LLM; la siguiente transcripción lo recarga.
- El prompt ordena no inventar datos, pero un LLM puede equivocarse: revisa tareas, fechas y nombres contra la transcripción.

## Datos y límites

- Modelos: se descargan una vez de Hugging Face a `~/.cache/huggingface`; después funciona sin internet.
- Historial: SQLite en `~/.transcriptor-audio/historial.db` (variable `TRANSCRIPTOR_DATA` para cambiarlo). Guarda texto y SRT, **no** el audio.
- Los originales nunca se modifican ni se borran.
- En GPU se usa inferencia por lotes (`BATCH_SIZE=16` en `core.py`; bájalo si falta VRAM).
- Audio ruidoso o con varios idiomas mezclados reduce la precisión; revisa el resultado.
- Sin diarización ni micrófono en vivo (fases futuras del PRD).

## Rendimiento medido (RTX 5060 Ti, WSL2, large-v3 float16)

| Modo | 30 min de audio | Velocidad |
| --- | --- | --- |
| Secuencial | 436 s | 4,1× |
| Por lotes (actual) | 18 s | 98× |

Ojo: el audio de prueba fue un clip de 11 s repetido (sintético, habla densa y sin
pausas). Con habla real será más lento; conviene repetir la medición con una
grabación real de ~1 h antes de dar por cumplido el criterio de <10 min.

## Pruebas

```bash
.venv/bin/python -m pytest
```

Cubren exportación `.txt`/`.srt`, historial y la división de textos largos del resumen (el LLM se simula). La transcripción real se validó a mano con
el clip público `jfk.flac` de openai/whisper (sin datos sensibles).
