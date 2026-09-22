"""Safe file writing.

Large generated artefacts get grabbed by real-time virus scanners, search
indexers and editors, especially on Windows, and a half-written 24 MB CSV is
worse than no CSV at all. Everything here writes to a sibling temporary file
first and swaps it into place, retrying briefly if the destination is held.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

#: How long to keep retrying a blocked replace before giving up, in seconds.
DEFAULT_RETRY_SECONDS = 5.0


class FileLockedError(OSError):
    """Raised when a destination stays locked by another process."""

    def __init__(self, path: Path) -> None:
        super().__init__(
            f"cannot write {path}: the file is held open by another process.\n"
            "  Close anything viewing it (Excel, an editor, a notebook), or wait\n"
            "  for the virus scanner to release it, then run the command again."
        )
        self.path = path


def atomic_write(
    path: str | os.PathLike[str],
    write: Callable[[Path], None],
    retry_seconds: float = DEFAULT_RETRY_SECONDS,
) -> Path:
    """Write via a temporary sibling and swap it into place.

    ``write`` is called with the temporary path and must produce the complete
    file. The destination is only touched once the write has succeeded, so a
    crash mid-write leaves the previous version intact.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")

    if temporary.exists():
        temporary.unlink(missing_ok=True)

    write(temporary)

    deadline = time.monotonic() + max(retry_seconds, 0.0)
    while True:
        try:
            os.replace(temporary, destination)
            return destination
        except PermissionError:
            if time.monotonic() >= deadline:
                temporary.unlink(missing_ok=True)
                raise FileLockedError(destination) from None
            time.sleep(0.25)


def write_csv(frame, path: str | os.PathLike[str], **kwargs) -> Path:
    """``DataFrame.to_csv`` through :func:`atomic_write`."""
    kwargs.setdefault("index", False)
    return atomic_write(path, lambda tmp: frame.to_csv(tmp, **kwargs))


def write_parquet(frame, path: str | os.PathLike[str], **kwargs) -> Path:
    """``DataFrame.to_parquet`` through :func:`atomic_write`."""
    kwargs.setdefault("index", False)
    return atomic_write(path, lambda tmp: frame.to_parquet(tmp, **kwargs))


def write_json(payload, path: str | os.PathLike[str], **kwargs) -> Path:
    """``json.dump`` through :func:`atomic_write`."""
    import json

    kwargs.setdefault("indent", 2)
    kwargs.setdefault("default", str)

    def _write(tmp: Path) -> None:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, **kwargs)

    return atomic_write(path, _write)


def write_text(text: str, path: str | os.PathLike[str]) -> Path:
    """Write text through :func:`atomic_write`."""
    return atomic_write(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))
