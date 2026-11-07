@echo off
REM Instalación única en Windows. Requiere Python 3.11-3.13 (python.org) y driver NVIDIA actualizado.
python -m venv .venv || exit /b 1
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e . || exit /b 1
echo Listo. Usa run.bat para abrir la interfaz.
