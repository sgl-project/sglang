#!/usr/bin/env python3
"""Validate the structural requirements of a patch-refinement commit message."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SUBJECT_RE = re.compile(
    r"^(?P<type>[a-z][a-z0-9-]*)\((?P<scope>[^()\s][^()]*)\):\s+(?P<summary>\S.*)$"
)
SECTION_RE = re.compile(r"^(Why|How|Dependency|Test):\s*$", re.MULTILINE)
EMPTY_VALUE_RE = re.compile(r"(?:<[^>]+>|\b(?:todo|tbd|none)\b)", re.IGNORECASE)
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
TEST_CASE_PROMPT = "Test case表示该commit需要哪些case进行验证，例如精度测试、需要特定arg、环境变量的性能测试"


def read_message(path: str | None) -> str:
    if path is None or path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def sections(message: str) -> dict[str, str]:
    matches = list(SECTION_RE.finditer(message))
    result: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(message)
        result[match.group(1)] = message[match.end() : end].strip()
    return result


def has_meaningful_value(body: str, label: str) -> bool:
    match = re.search(
        rf"^\s*-?\s*{re.escape(label)}:\s*(.+)$", body, re.MULTILINE | re.IGNORECASE
    )
    return bool(match and not EMPTY_VALUE_RE.search(match.group(1)))


def non_bullet_lines(body: str) -> list[str]:
    return [
        line
        for line in body.splitlines()
        if line.strip() and not re.match(r"^\s*-\s+\S", line)
    ]


def validate_e2e(e2e: str) -> list[str]:
    errors: list[str] = []
    model_matches = list(re.finditer(r"^  - ([^:<\n][^:\n]*):\s*$", e2e, re.MULTILINE))
    if not model_matches:
        return ["E2E must contain at least one model-titled item"]

    for index, match in enumerate(model_matches):
        model = match.group(1).strip()
        end = (
            model_matches[index + 1].start()
            if index + 1 < len(model_matches)
            else len(e2e)
        )
        body = e2e[match.end() : end]
        prefix = f"E2E model '{model}'"

        for label in ("Model data type", "Affected platforms"):
            value = re.search(
                rf"^    - {re.escape(label)}:\s*(\S.*)$",
                body,
                re.MULTILINE | re.IGNORECASE,
            )
            if not value or EMPTY_VALUE_RE.search(value.group(1)):
                errors.append(f"{prefix} must contain a concrete '{label}:' subtitle")

        test_case = re.search(
            r"^    - Test case:\s*$", body, re.MULTILINE | re.IGNORECASE
        )
        if not test_case:
            errors.append(f"{prefix} must contain a 'Test case:' subtitle")
            errors.append(TEST_CASE_PROMPT)
            continue

        case_body = body[test_case.end() :]
        cases = re.findall(r"^      -\s+(\S.*)$", case_body, re.MULTILINE)
        reserved_metadata = re.compile(
            r"^(?:Status|Configuration|Command/Procedure|Result/Blocker):",
            re.IGNORECASE,
        )
        concrete_cases = [
            case
            for case in cases
            if not reserved_metadata.match(case) and not EMPTY_VALUE_RE.search(case)
        ]
        if not concrete_cases:
            errors.append(
                f"{prefix} Test case must contain at least one caller-provided validation case"
            )
            errors.append(TEST_CASE_PROMPT)

    return errors


def validate(message: str) -> list[str]:
    errors: list[str] = []
    lines = message.strip().splitlines()
    if not lines:
        return ["commit message is empty"]

    if CJK_RE.search(message):
        errors.append(
            "commit message must be written in English; translate Chinese caller input while preserving literal identifiers and commands"
        )

    subject = SUBJECT_RE.match(lines[0])
    if not subject:
        errors.append(
            "subject must match 'type(scope): summary' with a non-empty scope"
        )
        commit_type = ""
    else:
        commit_type = subject.group("type")

    section_matches = list(SECTION_RE.finditer(message))
    section_names = [match.group(1) for match in section_matches]
    if section_names != ["Why", "How", "Dependency", "Test"]:
        errors.append(
            "message must contain exactly Why, How, Dependency, and Test modules in that order"
        )

    parsed = sections(message)
    for name in ("Why", "How", "Dependency", "Test"):
        if name in parsed:
            if not parsed[name]:
                errors.append(f"{name} module must not be empty")
            elif non_bullet_lines(parsed[name]):
                errors.append(f"every non-empty line in {name} must be a bullet point")

    why_required = commit_type not in {"chore", "typo"}
    if "Why" not in parsed:
        errors.append("Why module is required")
    elif why_required:
        why_bullets = re.findall(r"^-\s+(\S.*)$", parsed["Why"], re.MULTILINE)
        meaningful_bullets = [
            bullet for bullet in why_bullets if not EMPTY_VALUE_RE.search(bullet)
        ]
        if not meaningful_bullets:
            errors.append(
                "Why must contain at least one concrete bullet describing the problem this commit solves"
            )
        if any(
            re.match(r"(?:Problem|Impact if omitted):", bullet, re.IGNORECASE)
            for bullet in why_bullets
        ):
            errors.append(
                "Why must use direct problem-focused bullets without separate 'Problem:' or 'Impact if omitted:' fields"
            )
    elif not has_meaningful_value(parsed["Why"], "Exemption"):
        errors.append("chore/typo Why must contain an 'Exemption:' bullet")

    if "How" not in parsed:
        errors.append("How module is required immediately after Why")
    else:
        how = parsed["How"]
        how_bullets = re.findall(r"^-\s+(\S.*)$", how, re.MULTILINE)
        none_bullets = [bullet for bullet in how_bullets if bullet.lower() == "none"]
        if none_bullets:
            if len(how_bullets) != 1 or not re.fullmatch(
                r"\s*-\s+None\s*", how, re.IGNORECASE
            ):
                errors.append(
                    "How None must be exactly '- None' and cannot be combined with other content"
                )
        elif not how_bullets or any(
            EMPTY_VALUE_RE.search(bullet) for bullet in how_bullets
        ):
            errors.append(
                "How must contain concrete model-generated bullets or exactly '- None'"
            )

    if "Dependency" not in parsed:
        errors.append(
            "caller-provided Dependency is required; use a None bullet only when the caller confirms no dependency"
        )
    else:
        dependency = parsed["Dependency"]
        no_dependency = bool(re.fullmatch(r"\s*-\s+None\s*", dependency, re.IGNORECASE))
        has_real_dependency = bool(
            re.search(
                r"^\s*-\s+(?:Assumptions|Guards|Prerequisite commits|Library changes):\s*(?!<)\S.+$",
                dependency,
                re.MULTILINE | re.IGNORECASE,
            )
        )
        if not no_dependency and not has_real_dependency:
            errors.append(
                "Dependency must contain caller-provided dependency bullets or exactly '- None'"
            )
        if (
            no_dependency
            and len([line for line in dependency.splitlines() if line.strip()]) != 1
        ):
            errors.append(
                "Dependency None bullet cannot be combined with dependency entries"
            )
        if (
            not no_dependency
            and has_meaningful_value(dependency, "Assumptions")
            and not has_meaningful_value(dependency, "Guards")
        ):
            errors.append(
                "Dependency assumptions must identify their enforcing assertions or guards"
            )

    test = parsed.get("Test")
    if test is None:
        errors.append("Test section is required")
        return errors

    unit_heading = re.search(r"^-\s+Unit:\s*$", test, re.MULTILINE | re.IGNORECASE)
    if not unit_heading:
        errors.append("Test must contain a Unit subsection")
    else:
        unit_and_rest = test[unit_heading.end() :]
        unit = re.split(
            r"^-\s+E2E:\s*$",
            unit_and_rest,
            maxsplit=1,
            flags=re.MULTILINE | re.IGNORECASE,
        )[0]
        unit_entries = re.findall(r"^  -\s+(\S.*)$", unit, re.MULTILINE)
        none_entries = [
            entry for entry in unit_entries if entry.strip().lower() == "none"
        ]
        if not unit_entries:
            errors.append(
                "Unit must contain UT filenames from the commit or caller, or exactly '- None'"
            )
        elif none_entries:
            if len(unit_entries) != 1:
                errors.append("Unit None bullet cannot be combined with UT filenames")
        else:
            if any(EMPTY_VALUE_RE.search(entry) for entry in unit_entries):
                errors.append("Unit filenames must not contain placeholders")
            reserved = re.compile(r"^(?:Status|Command|Coverage):", re.IGNORECASE)
            if any(reserved.match(entry) for entry in unit_entries):
                errors.append(
                    "Unit must contain only UT filenames, without status, command, or coverage fields"
                )
    if not re.search(r"^\s*-?\s*E2E:\s*$", test, re.MULTILINE | re.IGNORECASE):
        errors.append("Test must contain an E2E subsection")
        return errors

    e2e = re.split(
        r"^\s*-?\s*E2E:\s*$", test, maxsplit=1, flags=re.MULTILINE | re.IGNORECASE
    )[1]
    errors.extend(validate_e2e(e2e))

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "message_file", nargs="?", help="commit message file; omit or use '-' for stdin"
    )
    args = parser.parse_args()

    try:
        message = read_message(args.message_file)
    except (OSError, UnicodeError) as exc:
        print(f"error: cannot read commit message: {exc}", file=sys.stderr)
        return 2

    errors = validate(message)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("commit message structure is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
