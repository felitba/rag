"""Download and prepare the ECCV 2018 type-aware inference artifacts.

The original authors' Google Drive checkpoint link currently returns 404. This
uses a public checkpoint based on the ECCV 2018 implementation, trained on the
Polyvore disjoint split, plus that split's matching type-space ordering.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import pickletools
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "models" / "type_aware"

# Pinned public artifacts. The checkpoint is from the original type-aware
# model implementation, trained on the disjoint Polyvore split.
CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/gyao15/fashion/"
    "0a4b64b459870a0f117d2bfed054d83f9d780cbc/"
    "runs/Type_Specific_Fashion_Compatibility/origin/model_best.pth.tar"
)
CHECKPOINT_SHA256 = "293e6d1ebc8160acced37b1caef3be9a7d9505c3d0c105fd922a334be9c9596c"
TYPESPACE_URL = (
    "https://huggingface.co/datasets/Stylique/Polyvore/resolve/"
    "1365152fecb3a8daee9d49c7d9d7f5809cc60b05/disjoint/typespaces.p"
    "?download=true"
)
TYPESPACE_SHA256 = "71edc80d00089631465ddadc768f78cde536ef541dc4734dd780c7bd0a815613"

SAFE_PICKLE_OPCODES = {
    "APPEND",
    "GET",
    "LIST",
    "MARK",
    "PUT",
    "STOP",
    "STRING",
    "TUPLE",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, path: Path, expected_sha256: str) -> None:
    curl = shutil.which("curl")
    if curl:
        # On macOS, curl uses the system Keychain, which may have certificates
        # that Python's bundled OpenSSL store does not know about.
        subprocess.run(
            [
                curl,
                "--fail",
                "--location",
                "--retry",
                "2",
                "--connect-timeout",
                "20",
                "--max-time",
                "120",
                "--output",
                str(path),
                url,
            ],
            check=True,
        )
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "fashion-rag/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path.name}: got {actual}")


def _load_typespaces(path: Path) -> list[tuple[str, str]]:
    payload = path.read_bytes()
    opcodes = {opcode.name for opcode, _argument, _position in pickletools.genops(payload)}
    unsupported = opcodes - SAFE_PICKLE_OPCODES
    if unsupported:
        raise ValueError(f"typespaces.p uses unsupported pickle opcodes: {sorted(unsupported)}")

    value = pickle.loads(payload)
    if not isinstance(value, (list, tuple)):
        raise ValueError("typespaces.p must contain a list or tuple")
    if not all(
        isinstance(pair, (list, tuple))
        and len(pair) == 2
        and all(isinstance(category, str) for category in pair)
        for pair in value
    ):
        raise ValueError("typespaces.p contains an invalid type pair")
    return [tuple(pair) for pair in value]


def _load_trusted_checkpoint(path: Path) -> dict[str, Any]:
    # The file is pinned by SHA-256 above. Keep the restricted unpickler enabled
    # and allow only NumPy scalar/dtype classes used by its training metadata.
    numpy_core = getattr(np, "_core", None)
    if numpy_core is None:  # NumPy 1.x
        numpy_core = np.core
    scalar = numpy_core.multiarray.scalar
    safe_globals = [
        (scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        np.ndarray,
        type(np.dtype(np.float64)),
    ]
    with torch.serialization.safe_globals(safe_globals):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("checkpoint does not contain a state_dict")
    return checkpoint["state_dict"]


def _validate_state_dict(state: dict[str, Any], type_space_count: int) -> None:
    def find_tensor(suffix: str):
        for key, value in state.items():
            normalized = key.removeprefix("module.")
            if normalized.endswith(suffix) and isinstance(value, torch.Tensor):
                return value
        return None

    masks = find_tensor("masks.weight")
    projection = find_tensor("fc_embed.weight")
    if masks is None or projection is None:
        raise ValueError("checkpoint is missing the type-aware masks or image projection")
    if masks.ndim != 2 or projection.ndim != 2:
        raise ValueError("checkpoint contains invalid mask or projection dimensions")
    if masks.shape[0] != type_space_count or masks.shape[1] != projection.shape[0]:
        raise ValueError(
            f"checkpoint/typespaces mismatch: masks={tuple(masks.shape)}, "
            f"projection={tuple(projection.shape)}, type_spaces={type_space_count}"
        )
    required = ("conv1.weight", "fc_embed.weight", "masks.weight")
    if any(find_tensor(suffix) is None for suffix in required):
        raise ValueError("checkpoint is missing required image encoder weights")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="type-aware-", dir=OUTPUT_DIR) as temp_dir:
        temp = Path(temp_dir)
        raw_checkpoint = temp / "checkpoint.pth.tar"
        raw_typespaces = temp / "typespaces.p"
        prepared_checkpoint = temp / "model_best.pth.tar"

        print("Downloading pinned checkpoint...")
        _download(CHECKPOINT_URL, raw_checkpoint, CHECKPOINT_SHA256)
        print("Downloading matching Polyvore disjoint type spaces...")
        _download(TYPESPACE_URL, raw_typespaces, TYPESPACE_SHA256)

        type_spaces = _load_typespaces(raw_typespaces)
        state = _load_trusted_checkpoint(raw_checkpoint)
        _validate_state_dict(state, len(type_spaces))
        if not all(isinstance(value, torch.Tensor) for value in state.values()):
            raise ValueError("checkpoint state_dict contains non-tensor values")

        # Strip optimizer/training metadata so inference only loads tensors.
        torch.save({"state_dict": state}, prepared_checkpoint)
        _validate_state_dict(_load_prepared_state(prepared_checkpoint), len(type_spaces))

        os.replace(raw_typespaces, OUTPUT_DIR / "typespaces.p")
        os.replace(prepared_checkpoint, OUTPUT_DIR / "model_best.pth.tar")

    print(f"Ready: {OUTPUT_DIR}")
    print(f"Type spaces: {len(type_spaces)}")
    print("Checkpoint: 256 feature channels, 64-d embeddings")


def _load_prepared_state(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint["state_dict"]


if __name__ == "__main__":
    main()
