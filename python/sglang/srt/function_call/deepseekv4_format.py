import re

_HEREDOC_START = re.compile(
    r"""(?<!<)<<(?!<)(-?)[ \t]*(?:'([^'\n]+)'|"([^"\n]+)"|([A-Za-z_][A-Za-z0-9_]*))"""
)


def mask_literals(text: str, *, heredocs: bool = False) -> str:
    """Keep offsets while excluding Markdown/code literals from protocol detection."""
    masked = list(text)
    fence = quote = ""
    ticks = 0
    escaped = False
    paired_quotes = {'"': '"', "'": "'", "\u201c": "\u201d", "\u2018": "\u2019"}
    pending_heredocs: list[tuple[str, bool]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        if pending_heredocs:
            delimiter, strip_tabs = pending_heredocs[0]
            end_line = line.rstrip("\r\n")
            if (end_line.lstrip("\t") if strip_tabs else end_line) == delimiter:
                pending_heredocs.pop(0)
            for index, char in enumerate(line):
                if char not in "\r\n":
                    masked[offset + index] = " "
            offset += len(line)
            continue
        boundary = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        protected = bool(fence)
        if boundary and not quote and not ticks:
            token, suffix = boundary.groups()
            if not fence:
                fence = token
            elif (
                token[0] == fence[0] and len(token) >= len(fence) and not suffix.strip()
            ):
                fence = ""
            protected = True
        if not quote and not ticks and line.lstrip().startswith(">"):
            protected = True
        if protected:
            for index, char in enumerate(line):
                if char not in "\r\n":
                    masked[offset + index] = " "
            offset += len(line)
            continue
        index = 0
        while index < len(line):
            char = line[index]
            if quote:
                if char not in "\r\n":
                    masked[offset + index] = " "
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = ""
            elif char == "`":
                end = index + 1
                while end < len(line) and line[end] == "`":
                    end += 1
                width = end - index
                if not ticks:
                    ticks = width
                elif ticks == width:
                    ticks = 0
                masked[offset + index : offset + end] = [" "] * width
                index = end
                continue
            elif ticks:
                if char not in "\r\n":
                    masked[offset + index] = " "
            elif (
                heredocs
                and char == "#"
                and (
                    index == 0 or line[index - 1].isspace() or line[index - 1] in ";|&("
                )
            ):
                for position in range(index, len(line)):
                    if line[position] not in "\r\n":
                        masked[offset + position] = " "
                break
            elif (
                heredocs
                and char == "<"
                and (opener := _HEREDOC_START.match(line, index))
            ):
                delimiter = next(
                    value for value in opener.groups()[1:] if value is not None
                )
                pending_heredocs.append((delimiter, opener[1] == "-"))
                width = opener.end() - index
                masked[offset + index : offset + opener.end()] = [" "] * width
                index = opener.end()
                continue
            elif char in paired_quotes and (
                char in "\u201c\u2018"
                or offset + index == 0
                or not (
                    text[offset + index - 1].isalnum()
                    or text[offset + index - 1] == "_"
                )
            ):
                quote = paired_quotes[char]
                masked[offset + index] = " "
            index += 1
        offset += len(line)
    return "".join(masked)
