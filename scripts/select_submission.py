"""Render one controlled candidate from the v6 incumbent Compose file.

The repository keeps ``docker-compose.yml`` as the only submission artifact.
This helper deliberately starts from that file and changes exactly one
experiment group at a time.  Render to a temporary path, validate it, and
promote it to the root file only after a successful portal result.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


V6_REQUIRED_ARGUMENTS = (
    "--max-model-len=8192",
    "--gpu-memory-utilization=0.97",
    "--quantization=fp8",
    "--kv-cache-dtype=fp8_e4m3",
    "--enable-prefix-caching",
)

UNCONTROLLED_ARGUMENT_PREFIXES = (
    "--max-num-seqs",
    "--max-num-batched-tokens",
    "--enable-chunked-prefill",
    "--async-scheduling",
    "--performance-mode",
    "--block-size",
    "--calculate-kv-scales",
)

IMAGE_LINE = re.compile(
    r"^(?P<indent>[ \t]*)image:[ \t]*(?P<value>.+?)[ \t]*$", re.MULTILINE
)
COMMAND_ITEM = re.compile(r"^(?P<indent>\s*)-\s+(?P<value>.+?)\s*$")
DIGEST = re.compile(r"@sha256:[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class Candidate:
    """A portal experiment whose change is intentionally isolated."""

    name: str
    description: str
    additional_arguments: tuple[str, ...] = ()
    requires_custom_image: bool = False


CANDIDATES: dict[str, Candidate] = {
    "v6-incumbent": Candidate(
        "v6-incumbent",
        "Unchanged v6: FP8 weights, FP8 E4M3 KV cache, and APC.",
    ),
    "shortconv-fp8": Candidate(
        "shortconv-fp8",
        "v6 with the custom image that quantizes ShortConv projections.",
        requires_custom_image=True,
    ),
    "speculative-draft": Candidate(
        "speculative-draft",
        "v6 with the baked local LFM2.5-350M draft model (4 draft tokens).",
        additional_arguments=(
            '--speculative-config={"model":"/opt/draft/LFM2.5-350M",'
            '"num_speculative_tokens":4,"draft_tensor_parallel_size":1,'
            '"max_model_len":8192}',
        ),
        requires_custom_image=True,
    ),
    "batch1536": Candidate(
        "batch1536",
        "v6 with only max-num-batched-tokens=1536 changed.",
        additional_arguments=("--max-num-batched-tokens=1536",),
    ),
    "batch1024": Candidate(
        "batch1024",
        "v6 with only max-num-batched-tokens=1024 changed.",
        additional_arguments=("--max-num-batched-tokens=1024",),
    ),
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unquote_yaml_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _quote_yaml_scalar(value: str) -> str:
    """Quote only values that YAML could otherwise parse as a mapping."""
    if any(character in value for character in "{}[],:#"):
        return "'" + value.replace("'", "''") + "'"
    return value


def _argument_prefix(argument: str) -> str:
    return argument.split("=", maxsplit=1)[0]


def _command_item_value(line: str) -> str | None:
    match = COMMAND_ITEM.match(line)
    if not match:
        return None
    return _unquote_yaml_scalar(match.group("value"))


def validate_v6_source(compose_text: str) -> None:
    """Reject a source that is not the narrow v6 baseline for experiments."""
    command_arguments = {
        argument
        for line in compose_text.splitlines()
        if (argument := _command_item_value(line)) is not None
    }
    missing = [
        argument for argument in V6_REQUIRED_ARGUMENTS if argument not in command_arguments
    ]
    if missing:
        raise ValueError(
            "Source Compose is not the v6 incumbent; missing " + ", ".join(missing)
        )

    unexpected = []
    for line in compose_text.splitlines():
        argument = _command_item_value(line)
        if argument and argument.startswith(UNCONTROLLED_ARGUMENT_PREFIXES):
            unexpected.append(argument)
    if unexpected:
        raise ValueError(
            "Source Compose contains scheduler/performance experiments that are "
            "not part of v6: " + ", ".join(unexpected)
        )


def _replace_image(compose_text: str, image: str) -> str:
    matches = [match for match in IMAGE_LINE.finditer(compose_text)]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one Compose image line; found " + str(len(matches))
        )
    match = matches[0]
    return (
        compose_text[: match.start()]
        + f"{match.group('indent')}image: {image}"
        + compose_text[match.end() :]
    )


def _replace_or_insert_argument(compose_text: str, argument: str) -> str:
    """Replace a matching vLLM option or insert it after APC.

    Keeping the insertion next to ``--enable-prefix-caching`` makes the
    rendered diff small and prevents unrelated scheduler options from being
    introduced by this tool.
    """
    prefix = _argument_prefix(argument)
    lines = compose_text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        current = _command_item_value(line)
        if current and (current == prefix or current.startswith(prefix + "=")):
            indent = COMMAND_ITEM.match(line).group("indent")  # type: ignore[union-attr]
            newline = "\r\n" if line.endswith("\r\n") else "\n"
            lines[index] = f"{indent}- {_quote_yaml_scalar(argument)}{newline}"
            return "".join(lines)

    for index, line in enumerate(lines):
        if _command_item_value(line) == "--enable-prefix-caching":
            indent = COMMAND_ITEM.match(line).group("indent")  # type: ignore[union-attr]
            newline = "\r\n" if line.endswith("\r\n") else "\n"
            lines.insert(index + 1, f"{indent}- {_quote_yaml_scalar(argument)}{newline}")
            return "".join(lines)
    raise ValueError("Could not find --enable-prefix-caching in source Compose")


def render_compose(
    candidate_name: str,
    source_compose: str,
    custom_image: str | None = None,
) -> str:
    """Return a candidate Compose document with only its declared change."""
    try:
        candidate = CANDIDATES[candidate_name]
    except KeyError as exc:
        raise ValueError(f"Unknown candidate: {candidate_name}") from exc

    validate_v6_source(source_compose)
    if candidate.requires_custom_image:
        if not custom_image:
            raise ValueError(f"{candidate.name} requires --custom-image pinned by digest")
        if not DIGEST.search(custom_image):
            raise ValueError(
                "--custom-image must be immutable and end with @sha256:<64 hex chars>"
            )
    elif custom_image:
        raise ValueError(f"{candidate.name} must keep the incumbent image")

    rendered = source_compose
    if custom_image:
        rendered = _replace_image(rendered, custom_image)
    for argument in candidate.additional_arguments:
        rendered = _replace_or_insert_argument(rendered, argument)

    header = (
        f"# Controlled candidate: {candidate.name}\n"
        f"# Source v6 Compose SHA-256: {sha256_text(source_compose)}\n"
        f"# Change: {candidate.description}\n"
    )
    return header + rendered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("docker-compose.yml"),
        help="v6 incumbent Compose file; defaults to the root submission artifact",
    )
    parser.add_argument(
        "--custom-image",
        help=(
            "Digest-pinned custom image required by shortconv-fp8 and "
            "speculative-draft"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Candidate output. Use --promote to intentionally replace --source.",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Allow --output to be the source/root submission file after review.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        parser.error(f"Source Compose does not exist: {source}")
    if source == output and not args.promote:
        parser.error("Refusing to overwrite --source without --promote")

    try:
        rendered = render_compose(
            args.candidate,
            source.read_text(encoding="utf-8"),
            args.custom_image,
        )
    except ValueError as exc:
        parser.error(str(exc))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
