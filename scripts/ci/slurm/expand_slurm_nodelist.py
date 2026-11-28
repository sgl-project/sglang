"""Expand scheduler NodeList values without requiring Slurm's hostnames command."""

import re
import sys


def expand_nodelist(nodelist: str) -> list[str]:
    if not nodelist or nodelist == "(null)":
        return []

    hosts = []
    # Commas inside a bracket group separate indices, not hostnames.
    for expression in re.split(r",(?![^\[]*\])", nodelist):
        if "[" not in expression:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", expression):
                raise ValueError(f"Invalid scheduler hostname: {expression!r}")
            hosts.append(expression)
            continue

        match = re.fullmatch(r"([^\[\]]*)\[([^\[\]]+)\](.*)", expression)
        if match is None:
            raise ValueError(f"Invalid scheduler hostlist: {expression!r}")
        prefix, indices, suffix = match.groups()
        for index_range in indices.split(","):
            match = re.fullmatch(r"([0-9]+)(?:-([0-9]+))?", index_range)
            if match is None:
                raise ValueError(f"Invalid scheduler host range: {index_range!r}")
            first, last = match.groups()
            last = last or first
            if int(first) > int(last):
                raise ValueError(f"Descending scheduler host range: {index_range!r}")
            width = len(first) if first.startswith("0") else 0
            for index in range(int(first), int(last) + 1):
                hosts.extend(expand_nodelist(f"{prefix}{index:0{width}d}{suffix}"))
    return hosts


if __name__ == "__main__":
    try:
        hosts = expand_nodelist(sys.stdin.read().strip())
    except ValueError as exc:
        sys.exit(str(exc))
    for host in hosts:
        print(host)
