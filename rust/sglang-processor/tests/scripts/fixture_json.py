"""The JSON layout of the parity fixtures."""

import json

# Lists of stream frames and events, written one item per line.
ROWS = ("frames", "events")


def dump(fixture):
    return compact(fixture, 0) + "\n"


def compact(value, indent, rows=False):
    """JSON indented by two, with each value that fits in 100 columns on one line.

    A list of numbers or strings fills its lines up to 100 columns.
    """
    flat = json.dumps(value, ensure_ascii=False)
    if not isinstance(value, (dict, list)) or not value or indent + len(flat) <= 100:
        return flat
    pad = " " * (indent + 2)
    if isinstance(value, dict):
        items = [
            f"{pad}{json.dumps(k, ensure_ascii=False)}: "
            + compact(v, indent + 2, rows=k in ROWS)
            for k, v in value.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    if not any(isinstance(v, (dict, list)) for v in value):
        return "[\n" + fill(value, pad) + "\n" + " " * indent + "]"
    if rows:
        items = [pad + json.dumps(v, ensure_ascii=False) for v in value]
    else:
        items = [pad + compact(v, indent + 2) for v in value]
    return "[\n" + ",\n".join(items) + "\n" + " " * indent + "]"


def fill(values, pad):
    lines, line = [], ""
    for v in values:
        item = json.dumps(v, ensure_ascii=False) + ","
        if line and len(pad) + len(line) + 1 + len(item) > 100:
            lines.append(pad + line)
            line = ""
        line = f"{line} {item}" if line else item
    lines.append(pad + line[:-1])
    return "\n".join(lines)
