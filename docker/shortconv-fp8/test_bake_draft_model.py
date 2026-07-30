"""Focused tests for legal-file preservation in the draft image bake helper."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


BAKE_PATH = Path(__file__).with_name("bake_draft_model.py")
REPOSITORY_ROOT = BAKE_PATH.parents[2]
BAKE_SPEC = importlib.util.spec_from_file_location("bake_draft_model", BAKE_PATH)
assert BAKE_SPEC is not None and BAKE_SPEC.loader is not None
bake = importlib.util.module_from_spec(BAKE_SPEC)
sys.modules[BAKE_SPEC.name] = bake
BAKE_SPEC.loader.exec_module(bake)


class DraftBakeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = {
            "BAKE_DRAFT_MODEL": "1",
            "DRAFT_MODEL_ID": bake.EXPECTED_MODEL_ID,
            "DRAFT_MODEL_REVISION": "a" * 40,
            # A build credential must never be copied into BUILD_MANIFEST.json.
            "HF_TOKEN": "test-only-secret",
        }

    def test_preserves_and_records_license_and_notice(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "draft"
            calls: list[dict[str, object]] = []

            class FakeApi:
                def list_repo_files(self, **kwargs: object) -> list[str]:
                    calls.append({"api": kwargs})
                    return ["config.json", "LICENSE", "NOTICE"]

            def snapshot_download(**kwargs: object) -> None:
                calls.append({"download": kwargs})
                destination.mkdir(parents=True, exist_ok=True)
                for required in bake.REQUIRED_FILES:
                    (destination / required).write_bytes(b"draft-model")
                (destination / "LICENSE").write_text("license text\n", encoding="utf-8")
                (destination / "NOTICE").write_text("notice text\n", encoding="utf-8")

            fake_hub = types.ModuleType("huggingface_hub")
            fake_hub.HfApi = FakeApi
            fake_hub.snapshot_download = snapshot_download

            with (
                patch.object(bake, "DESTINATION", destination),
                patch.dict(os.environ, self.environment, clear=False),
                patch.dict(sys.modules, {"huggingface_hub": fake_hub}),
            ):
                self.assertEqual(bake.main(), 0)

            download_call = calls[1]["download"]
            self.assertIn("LICENSE", download_call["allow_patterns"])
            self.assertIn("NOTICE", download_call["allow_patterns"])

            manifest_text = (destination / "BUILD_MANIFEST.json").read_text(encoding="utf-8")
            manifest = json.loads(manifest_text)
            self.assertEqual(manifest["schema_version"], 2)
            self.assertTrue(manifest["notice_present"])
            self.assertEqual(
                [record["path"] for record in manifest["legal_files"]],
                ["LICENSE", "NOTICE"],
            )
            self.assertTrue(all(len(record["sha256"]) == 64 for record in manifest["legal_files"]))
            self.assertNotIn(self.environment["HF_TOKEN"], manifest_text)
            self.assertNotIn("/model", manifest_text)

    def test_license_is_required_before_download(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeApi:
            def list_repo_files(self, **kwargs: object) -> list[str]:
                calls.append({"api": kwargs})
                return ["config.json", "NOTICE"]

        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.HfApi = FakeApi
        fake_hub.snapshot_download = lambda **kwargs: self.fail("download must not run")

        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            patch.object(bake, "DESTINATION", Path(temporary_directory) / "draft"),
            patch.dict(os.environ, self.environment, clear=False),
            patch.dict(sys.modules, {"huggingface_hub": fake_hub}),
        ):
            self.assertEqual(bake.main(), 1)
        self.assertEqual(len(calls), 1)

    def test_rejects_a_non_pinned_draft_model_before_network_access(self) -> None:
        environment = dict(self.environment, DRAFT_MODEL_ID="LiquidAI/LFM2.5-1.2B-Instruct")
        with patch.dict(os.environ, environment, clear=False):
            self.assertEqual(bake.main(), 2)

    def test_listed_notice_must_be_downloaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "draft"

            class FakeApi:
                def list_repo_files(self, **kwargs: object) -> list[str]:
                    return ["LICENSE", "NOTICE"]

            def snapshot_download(**kwargs: object) -> None:
                destination.mkdir(parents=True, exist_ok=True)
                for required in bake.REQUIRED_FILES:
                    (destination / required).write_bytes(b"draft-model")
                (destination / "LICENSE").write_text("license text\n", encoding="utf-8")

            fake_hub = types.ModuleType("huggingface_hub")
            fake_hub.HfApi = FakeApi
            fake_hub.snapshot_download = snapshot_download

            with (
                patch.object(bake, "DESTINATION", destination),
                patch.dict(os.environ, self.environment, clear=False),
                patch.dict(sys.modules, {"huggingface_hub": fake_hub}),
            ):
                self.assertEqual(bake.main(), 1)
            self.assertFalse((destination / "BUILD_MANIFEST.json").exists())

    def test_docker_build_context_cannot_copy_target_or_secrets(self) -> None:
        dockerfile = (REPOSITORY_ROOT / "Dockerfile").read_text(encoding="utf-8")
        copy_lines = [
            line.strip()
            for line in dockerfile.splitlines()
            if line.lstrip().startswith("COPY ")
        ]
        self.assertEqual(
            copy_lines,
            [
                "COPY docker/shortconv-fp8/patch_vllm_shortconv_fp8.py /opt/vllm-patches/",
                "COPY docker/shortconv-fp8/bake_draft_model.py /opt/vllm-patches/",
            ],
        )
        self.assertTrue(all("/model" not in line for line in copy_lines))

        dockerignore = (REPOSITORY_ROOT / ".dockerignore").read_text(encoding="utf-8")
        self.assertTrue(dockerignore.startswith("# The custom image requires only"))
        self.assertIn("\n**\n", dockerignore)
        self.assertNotIn("!**", dockerignore)


if __name__ == "__main__":
    unittest.main()
