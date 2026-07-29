#!/usr/bin/env python3
"""Optionally bake the pinned speculative draft model into a custom image."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


DESTINATION = Path("/opt/draft/LFM2.5-350M")
REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)


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
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        print("DRAFT_MODEL_REVISION must be a 40-character lowercase commit SHA.", file=sys.stderr)
        return 2

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        print("The base vLLM image does not provide huggingface_hub.", file=sys.stderr)
        raise SystemExit(1) from exc

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
            "LICENSE",
        ),
    )
    missing = [name for name in REQUIRED_FILES if not (DESTINATION / name).is_file()]
    if missing:
        print(f"Draft model download is incomplete: {', '.join(missing)}", file=sys.stderr)
        return 1

    (DESTINATION / "BUILD_MANIFEST.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "revision": revision,
                "required_files": REQUIRED_FILES,
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
