"""Hace visibles las bibliotecas CUDA instaladas vía pip (nvidia-cublas-cu12, nvidia-cudnn-cu12).

Debe llamarse antes de importar ctranslate2/faster_whisper.
"""
import ctypes
import glob
import importlib.util
import os
import sys

_done = False


def _nvidia_lib_dirs() -> list[str]:
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return []
    sub = "bin" if sys.platform == "win32" else "lib"
    dirs = []
    for base in spec.submodule_search_locations:
        dirs += glob.glob(os.path.join(base, "*", sub))
    return dirs


def setup_cuda_libs() -> None:
    global _done
    if _done:
        return
    _done = True
    dirs = _nvidia_lib_dirs()
    if sys.platform == "win32":
        for d in dirs:
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    else:
        # LD_LIBRARY_PATH ya no se relee tras el arranque: precargamos las .so.
        for d in dirs:
            for so in sorted(glob.glob(os.path.join(d, "lib*.so*"))):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
