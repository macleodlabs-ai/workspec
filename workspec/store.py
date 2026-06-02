"""Per-scope persistence for contextual learning.

:class:`ProfileStore` knew one file: ``~/.workspec/voice_profile.json``. The
contextual model needs one file *per scope* — a global profile, plus a profile
per recipient (and, later, per channel/project). :class:`ContextStore` is that
addressing layer.

Layout under ``base_dir`` (``~/.workspec`` by default)::

    voice/<scope>.json        VoiceProfile per scope (reused verbatim)
    contract/<scope>.json     ContractDelta per scope (learned structural overlay)
    capability/<scope>.json   Capability per scope (owner-set manual dial)

Migration (Decision 8, lossless + one-time): the first time the store is asked
for the global voice profile, if a legacy ``voice_profile.json`` exists at the
``base_dir`` root and ``voice/global.json`` does not, the legacy file is moved
into place as the global scope. With no ``--recipient`` everything resolves to
global, so this leaves existing behavior byte-identical.

All three artifact kinds are wired. The capability dial is owner-set only
(Decision 4): there is no code path here that writes a bucket except an explicit
:meth:`save_capability` call behind the ``workspec capability set`` command.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from workspec.capability import Capability
from workspec.context import ContextKey
from workspec.contract import ContractDelta
from workspec.profile import (
    PROFILE_FILENAME,
    ProfileLoadError,
    ProfileStore,
    VoiceProfile,
)

DEFAULT_BASE_DIR = Path.home() / ".workspec"

# Sub-directories, one per learned artifact kind. All three are live: ``voice``
# (style), ``contract`` (structure), ``capability`` (the owner-set manual dial).
_VOICE_DIR = "voice"
_CONTRACT_DIR = "contract"  # learned structural contract overlay (per scope)
_CAPABILITY_DIR = "capability"  # owner-set manual capability dial (per scope)


class ContextStore:
    """Addresses per-scope learning files under one base directory.

    Parameters
    ----------
    base_dir:
        Root of the WorkSpec data dir (default ``~/.workspec``). Per-scope
        voice profiles live under ``<base_dir>/voice/<scope>.json``.
    """

    def __init__(self, base_dir: Path | str = DEFAULT_BASE_DIR) -> None:
        self.base_dir = Path(base_dir)
        self._legacy_path = self.base_dir / PROFILE_FILENAME
        self._migrated = False

    # --- paths ------------------------------------------------------------ #

    @property
    def voice_dir(self) -> Path:
        return self.base_dir / _VOICE_DIR

    @property
    def contract_dir(self) -> Path:
        """Location of the learned structural contract overlay (one file per scope)."""
        return self.base_dir / _CONTRACT_DIR

    @property
    def capability_dir(self) -> Path:
        """Location of the owner-set manual capability dial (one file per scope)."""
        return self.base_dir / _CAPABILITY_DIR

    def voice_path(self, key: ContextKey) -> Path:
        """The voice-profile file backing ``key``'s own scope (no backoff)."""
        return self.voice_dir / f"{key.scope_id}.json"

    def contract_path(self, key: ContextKey) -> Path:
        """The contract-delta file backing ``key``'s own scope (no backoff)."""
        return self.contract_dir / f"{key.scope_id}.json"

    def capability_path(self, key: ContextKey) -> Path:
        """The capability file backing ``key``'s own scope (no backoff)."""
        return self.capability_dir / f"{key.scope_id}.json"

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        """Write ``text`` to ``path`` atomically via a unique temp file.

        Each writer gets its own temp file (``mkstemp`` in ``path``'s directory),
        so concurrent writers of the same scope never interleave into one shared
        temp; the final :func:`os.replace` is atomic. A temp left behind by a
        failed write (e.g. a full disk) is cleaned up rather than orphaned.
        """
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    # --- migration -------------------------------------------------------- #

    def _migrate_legacy_if_needed(self) -> None:
        """Move a legacy ``voice_profile.json`` into the global scope, once.

        Lossless: the file is relocated, not rewritten, so a hand-edited legacy
        profile survives byte-for-byte. Runs at most once per store and only
        when the global scope file does not already exist.
        """
        if self._migrated:
            return
        self._migrated = True
        global_path = self.voice_path(ContextKey())
        if global_path.exists() or not self._legacy_path.exists():
            return
        global_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self._legacy_path, global_path)

    # --- voice ------------------------------------------------------------ #

    def voice_store(self, key: ContextKey) -> ProfileStore:
        """A :class:`ProfileStore` bound to ``key``'s scope file.

        Reuses the existing store verbatim so all load/save/atomicity semantics
        (and the ``ProfileLoadError`` contract) carry over unchanged. The
        one-time legacy migration runs before the global store is handed out.
        """
        if key.is_global():
            self._migrate_legacy_if_needed()
        return ProfileStore(self.voice_dir, filename=f"{key.scope_id}.json")

    def load_voice(self, key: ContextKey) -> VoiceProfile:
        """Load the voice profile for ``key``'s own scope (empty if absent)."""
        return self.voice_store(key).load()

    def save_voice(self, key: ContextKey, profile: VoiceProfile) -> None:
        """Persist ``profile`` to ``key``'s own scope file."""
        self.voice_store(key).save(profile)

    # --- contract --------------------------------------------------------- #

    def load_contract(self, key: ContextKey) -> ContractDelta:
        """Load the contract delta for ``key``'s own scope (empty if absent).

        Reuses the atomic-write / clear-error semantics of :class:`ProfileStore`:
        a missing file is an empty delta, and a malformed (hand-edited) file
        raises :class:`~workspec.profile.ProfileLoadError` rather than leaking a
        raw traceback. The contract delta is intentionally *not* migrated from any
        legacy file — it is a new artifact with no v1 ancestor.
        """
        path = self.contract_path(key)
        if not path.exists():
            return ContractDelta()
        try:
            return ContractDelta.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ProfileLoadError(f"could not read contract delta at {path}: {exc}") from exc

    def save_contract(self, key: ContextKey, delta: ContractDelta) -> None:
        """Persist ``delta`` to ``key``'s own scope file (atomic write)."""
        path = self.contract_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(path, delta.model_dump_json(indent=2))

    # --- capability ------------------------------------------------------- #

    def load_capability(self, key: ContextKey) -> Capability:
        """Load the owner-set capability for ``key``'s own scope (no backoff).

        Returns the default bucket (``developing``) when no file exists, so an
        un-rated recipient is treated as ``developing`` without any inference
        (Decision 4). A malformed (hand-edited) file raises
        :class:`~workspec.profile.ProfileLoadError` rather than leaking a raw
        traceback, matching the other artifacts' clear-error contract.
        """
        path = self.capability_path(key)
        if not path.exists():
            return Capability()
        try:
            return Capability.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ProfileLoadError(f"could not read capability at {path}: {exc}") from exc

    def save_capability(self, key: ContextKey, capability: Capability) -> None:
        """Persist the owner-set ``capability`` to ``key``'s own scope (atomic write).

        The only path that writes a bucket: it runs solely behind an explicit
        owner command, never from any observed signal (Decision 4).
        """
        path = self.capability_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(path, capability.model_dump_json(indent=2))
