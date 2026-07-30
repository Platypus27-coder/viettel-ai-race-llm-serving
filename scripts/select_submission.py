"""Render one controlled candidate from its declared Compose parent.

The repository keeps ``docker-compose.yml`` as the only submission artifact.
This helper deliberately changes exactly one experiment group at a time.
Most candidates start from the v6 incumbent.  Scheduler children of
``speculative-draft`` are deliberately named as such and can only inherit a
digest-pinned, renderer-produced speculative parent; this prevents a later
batch experiment from silently falling back to v6.
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
V6_IMAGE = "vllm/vllm-openai:v0.22.1"

UNCONTROLLED_ARGUMENT_PREFIXES = (
    "--max-num-seqs",
    "--max-num-batched-tokens",
    "--speculative-config",
    "--enable-chunked-prefill",
    "--async-scheduling",
    "--performance-mode",
    "--block-size",
    "--calculate-kv-scales",
)

SPECULATIVE_DRAFT_ARGUMENT = (
    '--speculative-config={"method":"draft_model",'
    '"model":"/opt/draft/LFM2.5-350M",'
    '"num_speculative_tokens":4,"draft_tensor_parallel_size":1,'
    '"max_model_len":8192}'
)

IMAGE_LINE = re.compile(
    r"^(?P<indent>[ \t]*)image:[ \t]*(?P<value>.+?)[ \t]*$", re.MULTILINE
)
COMMAND_ITEM = re.compile(r"^(?P<indent>\s*)-\s+(?P<value>.+?)\s*$")
CONTROLLED_CANDIDATE_LINE = re.compile(
    r"^# Controlled candidate:[ \t]*(?P<name>[a-z0-9][a-z0-9-]*)[ \t]*$",
    re.MULTILINE,
)
PARENT_CANDIDATE_LINE = re.compile(
    r"^# Parent candidate:[ \t]*(?P<name>[a-z0-9][a-z0-9-]*)[ \t]*$",
    re.MULTILINE,
)
PARENT_COMPOSE_SHA_LINE = re.compile(r"^# Parent Compose SHA-256: [0-9a-f]{64}[ \t]*$")
LEGACY_SOURCE_SHA_LINE = re.compile(r"^# Source v6 Compose SHA-256: [0-9a-f]{64}[ \t]*$")
CHANGE_LINE = re.compile(r"^# Change: .*$", re.MULTILINE)
# The contest accepts custom images only from public Docker Hub repositories.
# Keep the renderer fail-closed: a digest alone does not make a GHCR/private
# registry image eligible for submission.  Docker Hub accepts the short form
# ``namespace/repository`` and its explicit ``docker.io`` aliases.
DOCKER_HUB_DIGEST_IMAGE = re.compile(
    r"^(?:(?:docker\.io|index\.docker\.io)/)?"
    # A dot in the first path component denotes a registry host in Docker's
    # reference grammar, never a short-form Docker Hub namespace.
    r"[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?/"
    r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?"
    r"(?::[A-Za-z0-9_][A-Za-z0-9_.-]*)?"
    r"@sha256:[0-9a-fA-F]{64}$"
)


@dataclass(frozen=True)
class Candidate:
    """A portal experiment whose change is intentionally isolated."""

    name: str
    description: str
    additional_arguments: tuple[str, ...] = ()
    requires_custom_image: bool = False
    parent_candidate: str = "v6-incumbent"
    requires_digest_pinned_parent: bool = False


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
        additional_arguments=(SPECULATIVE_DRAFT_ARGUMENT,),
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
    "speculative-draft-batch1536": Candidate(
        "speculative-draft-batch1536",
        "The digest-pinned speculative-draft parent with only "
        "max-num-batched-tokens=1536 changed.",
        additional_arguments=("--max-num-batched-tokens=1536",),
        parent_candidate="speculative-draft",
        requires_digest_pinned_parent=True,
    ),
    "speculative-draft-batch1024": Candidate(
        "speculative-draft-batch1024",
        "The digest-pinned speculative-draft parent with only "
        "max-num-batched-tokens=1024 changed.",
        additional_arguments=("--max-num-batched-tokens=1024",),
        parent_candidate="speculative-draft",
        requires_digest_pinned_parent=True,
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


def _command_arguments(compose_text: str) -> list[str]:
    return [
        argument
        for line in compose_text.splitlines()
        if (argument := _command_item_value(line)) is not None
    ]


def _metadata_value(
    compose_text: str, pattern: re.Pattern[str], label: str
) -> str | None:
    matches = [match.group("name") for match in pattern.finditer(compose_text)]
    if len(matches) > 1:
        raise ValueError(
            f"Expected at most one {label} metadata line; found {len(matches)}"
        )
    return matches[0] if matches else None


def _strip_renderer_metadata(compose_text: str) -> str:
    """Drop a prior renderer header before adding the new candidate metadata.

    A scheduler child must contain one current candidate identity, not a stack
    of parent headers that could be misread as multiple controlled changes.
    Only a complete header at the very beginning is removed; normal Compose
    comments are left untouched.
    """
    lines = compose_text.splitlines(keepends=True)
    if not lines or not CONTROLLED_CANDIDATE_LINE.fullmatch(lines[0].rstrip("\r\n")):
        return compose_text

    required_patterns = (
        PARENT_CANDIDATE_LINE,
        PARENT_COMPOSE_SHA_LINE,
        CHANGE_LINE,
    )
    if len(lines) >= 4 and all(
        pattern.fullmatch(lines[index].rstrip("\r\n"))
        for index, pattern in enumerate(required_patterns, start=1)
    ):
        return "".join(lines[4:])

    # Accept an artifact rendered by the previous helper version so that it
    # can be re-rendered for review, but never remove an incomplete header.
    legacy_patterns = (LEGACY_SOURCE_SHA_LINE, CHANGE_LINE)
    if len(lines) >= 3 and all(
        pattern.fullmatch(lines[index].rstrip("\r\n"))
        for index, pattern in enumerate(legacy_patterns, start=1)
    ):
        return "".join(lines[3:])
    return compose_text


def _compose_image(compose_text: str) -> str:
    matches = [match for match in IMAGE_LINE.finditer(compose_text)]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one Compose image line; found " + str(len(matches))
        )
    return _unquote_yaml_scalar(matches[0].group("value"))


def _validate_v6_arguments(command_arguments: list[str]) -> None:
    missing = [
        argument for argument in V6_REQUIRED_ARGUMENTS if argument not in command_arguments
    ]
    if missing:
        raise ValueError(
            "Source Compose is not the v6 incumbent; missing " + ", ".join(missing)
        )


def _uncontrolled_arguments(
    command_arguments: list[str], *, allowed: tuple[str, ...] = ()
) -> list[str]:
    return [
        argument
        for argument in command_arguments
        if argument.startswith(UNCONTROLLED_ARGUMENT_PREFIXES)
        and argument not in allowed
    ]


def validate_v6_source(compose_text: str) -> None:
    """Reject a source that is not the narrow v6 baseline for experiments."""
    controlled_candidate = _metadata_value(
        compose_text, CONTROLLED_CANDIDATE_LINE, "controlled candidate"
    )
    if controlled_candidate not in {None, "v6-incumbent"}:
        raise ValueError(
            "Source Compose is a controlled "
            f"{controlled_candidate} candidate, not the v6 incumbent"
        )

    command_arguments = _command_arguments(compose_text)
    _validate_v6_arguments(command_arguments)
    image = _compose_image(compose_text)
    if image != V6_IMAGE:
        raise ValueError(
            "Source Compose is not the v6 incumbent; expected image " + V6_IMAGE
        )
    unexpected = _uncontrolled_arguments(command_arguments)
    if unexpected:
        raise ValueError(
            "Source Compose contains scheduler/performance experiments that are "
            "not part of v6: " + ", ".join(unexpected)
        )


def validate_speculative_draft_parent(compose_text: str) -> None:
    """Validate the exact, immutable parent accepted by speculative schedulers."""
    controlled_candidate = _metadata_value(
        compose_text, CONTROLLED_CANDIDATE_LINE, "controlled candidate"
    )
    parent_candidate = _metadata_value(
        compose_text, PARENT_CANDIDATE_LINE, "parent candidate"
    )
    if (
        controlled_candidate != "speculative-draft"
        or parent_candidate != "v6-incumbent"
    ):
        raise ValueError(
            "Scheduler children of speculative decoding require a renderer-produced "
            "speculative-draft parent"
        )

    command_arguments = _command_arguments(compose_text)
    _validate_v6_arguments(command_arguments)
    speculative_arguments = [
        argument
        for argument in command_arguments
        if argument.startswith("--speculative-config")
    ]
    if speculative_arguments != [SPECULATIVE_DRAFT_ARGUMENT]:
        raise ValueError(
            "Speculative parent must contain exactly the approved 4-token "
            "LFM2.5-350M --speculative-config"
        )
    unexpected = _uncontrolled_arguments(
        command_arguments, allowed=(SPECULATIVE_DRAFT_ARGUMENT,)
    )
    if unexpected:
        raise ValueError(
            "Speculative parent contains scheduler/performance experiments that are "
            "not part of the approved parent: " + ", ".join(unexpected)
        )

    image = _compose_image(compose_text)
    if not DOCKER_HUB_DIGEST_IMAGE.fullmatch(image):
        raise ValueError(
            "Speculative parent image must be a Docker Hub namespace/repository "
            "pinned by @sha256:<64 hex chars>"
        )


def validate_source_for_candidate(candidate: Candidate, compose_text: str) -> None:
    """Validate the declared parent before applying the candidate's one change."""
    if candidate.parent_candidate == "v6-incumbent":
        validate_v6_source(compose_text)
        return
    if candidate.parent_candidate == "speculative-draft":
        validate_speculative_draft_parent(compose_text)
        return
    raise ValueError(f"Unsupported parent candidate: {candidate.parent_candidate}")


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

    validate_source_for_candidate(candidate, source_compose)
    if candidate.requires_custom_image:
        if not custom_image:
            raise ValueError(f"{candidate.name} requires --custom-image pinned by digest")
        if not DOCKER_HUB_DIGEST_IMAGE.fullmatch(custom_image):
            raise ValueError(
                "--custom-image must be a Docker Hub namespace/repository "
                "pinned by @sha256:<64 hex chars>; verify public pull access "
                "before promotion"
            )
    elif custom_image:
        if candidate.requires_digest_pinned_parent:
            raise ValueError(
                f"{candidate.name} must inherit the image from its validated parent"
            )
        raise ValueError(f"{candidate.name} must keep the incumbent image")

    rendered = _strip_renderer_metadata(source_compose)
    if custom_image:
        rendered = _replace_image(rendered, custom_image)
    for argument in candidate.additional_arguments:
        rendered = _replace_or_insert_argument(rendered, argument)

    header = (
        f"# Controlled candidate: {candidate.name}\n"
        f"# Parent candidate: {candidate.parent_candidate}\n"
        f"# Parent Compose SHA-256: {sha256_text(source_compose)}\n"
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
        help=(
            "Declared parent Compose file; defaults to the root v6 incumbent. "
            "speculative-draft-batch* requires the rendered speculative-draft "
            "parent."
        ),
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
