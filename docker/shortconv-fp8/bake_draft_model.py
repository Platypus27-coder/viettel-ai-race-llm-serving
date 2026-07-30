#!/usr/bin/env python3
"""Optionally bake the pinned speculative draft model into a custom image."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable


DESTINATION = Path("/opt/draft/LFM2.5-350M")
EXPECTED_MODEL_ID = "LiquidAI/LFM2.5-350M"
REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)
LEGAL_FILENAMES = frozenset(
    {
        "LICENSE",
        "LICENSE.MD",
        "LICENSE.TXT",
        "NOTICE",
        "NOTICE.MD",
        "NOTICE.TXT",
    }
)


def _is_safe_relative_path(value: str) -> bool:
    path = Path(value)
    return (
        bool(value)
        and "\\" not in value
        and not path.is_absolute()
        and ".." not in path.parts
    )


def _legal_files_from_repository(files: Iterable[str]) -> tuple[str, ...]:
    """Return root-relative legal files that must be preserved in the image."""
    legal_files = {
        name
        for name in files
        if _is_safe_relative_path(name)
        and Path(name).name.upper() in LEGAL_FILENAMES
    }
    return tuple(sorted(legal_files))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _legal_file_manifest(destination: Path, legal_files: Iterable[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative_path in legal_files:
        path = destination / relative_path
        if not path.is_file():
            raise FileNotFoundError(relative_path)
        records.append(
            {
                "path": relative_path,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def main() -> int:
    enabled = os.environ.get("BAKE_DRAFT_MODEL", "0")
    if enabled == "0":
        print("Draft-model baking disabled (BAKE_DRAFT_MODEL=0).")
        return 0
    if enabled != "1":
        print("BAKE_DRAFT_MODEL must be exactly 0 or 1.", file=sys.stderr)
        return 2

    model_id = os.environ["DRAFT_MODEL_ID"]
    revision = os.environ["DRAFT_MODEL_REVISION"]
    if model_id != EXPECTED_MODEL_ID:
        print(
            f"DRAFT_MODEL_ID must be exactly {EXPECTED_MODEL_ID!r}.",
            file=sys.stderr,
        )
        return 2
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        print("DRAFT_MODEL_REVISION must be a 40-character lowercase commit SHA.", file=sys.stderr)
        return 2

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        print("The base vLLM image does not provide huggingface_hub.", file=sys.stderr)
        raise SystemExit(1) from exc

    repository_files = HfApi().list_repo_files(repo_id=model_id, revision=revision)
    legal_files = _legal_files_from_repository(repository_files)
    license_files = tuple(
        path
        for path in legal_files
        if Path(path).name.upper().startswith("LICENSE")
    )
    if not license_files:
        print(
            "The pinned draft revision has no distributable LICENSE file; refusing to bake it.",
            file=sys.stderr,
        )
        return 1

    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=DESTINATION,
        allow_patterns=(
            "*.json",
            "*.safetensors",
            "*.txt",
            "*.model",
            "*.jinja",
            *legal_files,
        ),
    )
    missing = [name for name in REQUIRED_FILES if not (DESTINATION / name).is_file()]
    if missing:
        print(f"Draft model download is incomplete: {', '.join(missing)}", file=sys.stderr)
        return 1

    try:
        legal_file_records = _legal_file_manifest(DESTINATION, legal_files)
    except FileNotFoundError as exc:
        print(
            f"Draft legal file was listed upstream but was not downloaded: {exc.args[0]}",
            file=sys.stderr,
        )
        return 1

    (DESTINATION / "BUILD_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "model_id": model_id,
                "revision": revision,
                "required_files": REQUIRED_FILES,
                "legal_files": legal_file_records,
                "notice_present": any(
                    Path(path).name.upper().startswith("NOTICE") for path in legal_files
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Baked {model_id}@{revision} into {DESTINATION}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
