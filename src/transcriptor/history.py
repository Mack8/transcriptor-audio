"""Historial local de transcripciones (SQLite)."""
import os
import sqlite3
from datetime import datetime
from pathlib import Path

from .export import Segment, to_srt, to_txt


def data_dir() -> Path:
    override = os.environ.get("TRANSCRIPTOR_DATA")
    base = Path(override) if override else Path.home() / ".transcriptor-audio"
    base.mkdir(parents=True, exist_ok=True)
    return base


class History:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or data_dir() / "historial.db"
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS transcripciones (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    creado TEXT NOT NULL, archivo TEXT NOT NULL, modelo TEXT NOT NULL,
                    idioma TEXT, duracion REAL, texto TEXT NOT NULL, srt TEXT NOT NULL)"""
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def add(self, archivo: str, modelo: str, idioma: str, duracion: float,
            segments: list[Segment]) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO transcripciones (creado, archivo, modelo, idioma, duracion, texto, srt)"
                " VALUES (?,?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), archivo, modelo, idioma,
                 duracion, to_txt(segments), to_srt(segments)),
            )
            return cur.lastrowid

    def list(self, limit: int = 100) -> list[tuple]:
        with self._conn() as c:
            return c.execute(
                "SELECT id, creado, archivo, modelo, idioma, duracion FROM transcripciones"
                " ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()

    def get(self, entry_id: int) -> tuple[str, str] | None:
        """Devuelve (texto, srt) o None."""
        with self._conn() as c:
            return c.execute(
                "SELECT texto, srt FROM transcripciones WHERE id=?", (entry_id,)
            ).fetchone()
