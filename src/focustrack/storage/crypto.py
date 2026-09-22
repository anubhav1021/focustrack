"""Encryption for the local store.

The agent records a minute-by-minute account of someone's working day. Even as
counts, that is a sensitive record - it shows when they started, when they
stopped, how long they stared at nothing - so it is encrypted at rest with a
key that never leaves the machine.

Fernet (AES-128-CBC with an HMAC) is used per value rather than whole-file
encryption, so the database stays queryable by time and user while the
readings themselves stay unreadable to anything without the key.
"""

from __future__ import annotations

import base64
import json
import os
import stat
from pathlib import Path
from typing import Any

try:
    from cryptography.fernet import Fernet, InvalidToken
    CRYPTO_AVAILABLE = True
except ImportError:                                          # pragma: no cover
    Fernet = None                                            # type: ignore[assignment]
    InvalidToken = Exception                                 # type: ignore[assignment]
    CRYPTO_AVAILABLE = False


class DecryptionError(RuntimeError):
    """Raised when a stored value cannot be decrypted with the current key."""


def generate_key() -> bytes:
    """Create a fresh Fernet key."""
    if not CRYPTO_AVAILABLE:
        raise RuntimeError(
            "the `cryptography` package is required for the encrypted store.\n"
            "  pip install cryptography"
        )
    return Fernet.generate_key()


def load_or_create_key(path: str | os.PathLike[str]) -> bytes:
    """Read the key at ``path``, creating it on first use.

    The file is created with owner-only permissions. On Windows the POSIX mode
    is advisory, so the key file's real protection there is the user profile's
    own ACL - which is why the store is never meant to be synced or shared.
    """
    key_path = Path(path)
    if key_path.exists():
        return key_path.read_bytes().strip()

    key_path.parent.mkdir(parents=True, exist_ok=True)
    key = generate_key()
    key_path.write_bytes(key)
    try:
        key_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except (OSError, NotImplementedError):                   # pragma: no cover
        pass
    return key


class ValueCipher:
    """Encrypts and decrypts the JSON payload of a single stored record."""

    def __init__(self, key: bytes | None, enabled: bool = True) -> None:
        self.enabled = bool(enabled and key is not None and CRYPTO_AVAILABLE)
        self._fernet = Fernet(key) if self.enabled else None

    def encrypt(self, payload: dict[str, Any]) -> bytes:
        raw = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
        if self._fernet is None:
            # Still base64 so the column type is the same either way.
            return base64.b64encode(raw)
        return self._fernet.encrypt(raw)

    def decrypt(self, blob: bytes | str) -> dict[str, Any]:
        data = blob.encode("utf-8") if isinstance(blob, str) else blob
        try:
            raw = base64.b64decode(data) if self._fernet is None else self._fernet.decrypt(data)
        except (InvalidToken, ValueError, TypeError) as exc:
            raise DecryptionError(
                "could not decrypt a stored record - the key does not match "
                "this database. If the key file was lost, the data cannot be "
                "recovered; delete the store to start a fresh one."
            ) from exc
        return json.loads(raw.decode("utf-8"))

    def describe(self) -> str:
        if not CRYPTO_AVAILABLE:
            return "disabled (cryptography not installed)"
        return "Fernet (AES-128-CBC + HMAC)" if self.enabled else "disabled by config"
