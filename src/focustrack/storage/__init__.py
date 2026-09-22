"""Layer 2 - the encrypted local store. Nothing here leaves the machine."""

from focustrack.storage.crypto import (
    CRYPTO_AVAILABLE,
    DecryptionError,
    ValueCipher,
    generate_key,
    load_or_create_key,
)
from focustrack.storage.db import FocusStore, StoreStats

__all__ = [
    "CRYPTO_AVAILABLE",
    "DecryptionError",
    "FocusStore",
    "StoreStats",
    "ValueCipher",
    "generate_key",
    "load_or_create_key",
]
