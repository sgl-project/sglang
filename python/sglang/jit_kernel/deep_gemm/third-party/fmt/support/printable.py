#!/usr/bin/env python3

# This script is based on
# https://github.com/rust-lang/rust/blob/master/library/core/src/unicode/printable.py
# distributed under https://github.com/rust-lang/rust/blob/master/LICENSE-MIT.

# This script uses the following Unicode tables:
# - UnicodeData.txt


from collections import namedtuple
import csv
import os
import subprocess

NUM_CODEPOINTS=0x110000

def to_ranges(iter):
    current = None
    for i in iter:
        if current is None or i != current[1] or i in (0x10000, 0x20000):
            if current is not None:
                yield tuple(current)
            current = [i, i + 1]
        else:
            current[1] += 1
    if current is not None:
        yield tuple(current)

def get_escaped(codepoints):
    for c in codepoints:
        if (c.class_ or "Cn") in "Cc Cf Cs Co Cn Zl Zp Zs".split() and c.value != ord(' '):
            yield c.value

def get_file(f):
    try:
        return open(os.path.basename(f))
    except FileNotFoundError:
        subprocess.run(["curl", "-O", f], check=True)
        return open(os.path.basename(f))

Codepoint = namedtuple('Codepoint', 'value class_')

def get_codepoints(f):
    r = csv.reader(f, delimiter=";")
    prev_codepoint = 0
    class_first = None
    for row in r:
        codepoint = int(row[0], 16)
        name = row[1]
        class_ = row[2]

        if class_first is not None:
            if not name.endswith("Last>"):
                raise ValueError("Missing Last after First")

        for c in range(prev_codepoint + 1, codepoint):
            yield Codepoint(c, class_first)

        class_first = None
        if name.endswith("First>"):
            class_first = class_

        yield Codepoint(codepoint, class_)
        prev_codepoint = codepoint

    if class_first is not None:
        raise ValueError("Missing Last after First")

    for c in range(prev_codepoint + 1, NUM_CODEPOINTS):
        yield Codepoint(c, None)

def compress_singletons(singletons):
    uppers = [] # (upper, # items in lowers)
    lowers = []

    for i in singletons:
        upper = i >> 8
        lower = i & 0xff
        if len(uppers) == 0 or uppers[-1][0] != upper:
            uppers.append((upper, 1))
        else:
            upper, count = uppers[-1]
            uppers[-1] = upper, count + 1
        lowers.append(lower)

    return uppers, lowers

def compress_normal(normal):
    # lengths 0x00..0x7f are encoded as 00, 01, ..., 7e, 7f
    # lengths 0x80..0x7fff are encoded as 80 80, 80 81, ..., ff fe, ff ff
    compressed = [] # [truelen, (truelenaux), falselen, (falselenaux)]

    prev_start = 0
    for start, count in normal:
        truelen = start - prev_start
        falselen = count
        prev_start = start + count

        assert truelen < 0x8000 and falselen < 0x8000
        entry = []
        if truelen > 0x7f:
            entry.append(0x80 | (truelen >> 8))
            entry.append(truelen & 0xff)
        else:
            entry.append(truelen & 0x7f)
        if falselen > 0x7f:
            entry.append(0x80 | (falselen >> 8))
            entry.append(falselen & 0xff)
        else:
            entry.append(falselen & 0x7f)

        compressed.append(entry)

    return compressed

def print_singletons(uppers, lowers, uppersname, lowersname):
    print("  static constexpr singleton {}[] = {{".format(uppersname))
    for u, c in uppers:
        print("    {{{:#04x}, {}}},".format(u, c))
    print("  };")
    print("  static constexpr unsigned char {}[] = {{".format(lowersname))
    for i in range(0, len(lowers), 8):
        print("    {}".format(" ".join("{:#04x},".format(l) for l in lowers[i:i+8])))
    print("  };")

def print_normal(normal, normalname):
    print("  static constexpr unsigned char {}[] = {{".format(normalname))
    for v in normal:
        print("    {}".format(" ".join("{:#04x},".format(i) for i in v)))
    print("  };")

def main():
    file = get_file("https://www.unicode.org/Public/UNIDATA/UnicodeData.txt")

    codepoints = get_codepoints(file)

    CUTOFF=0x10000
    singletons0 = []
    singletons1 = []
    normal0 = []
    normal1 = []
    extra = []

    for a, b in to_ranges(get_escaped(codepoints)):
        if a > 2 * CUTOFF:
            extra.append((a, b - a))
        elif a == b - 1:
            if a & CUTOFF:
                singletons1.append(a & ~CUTOFF)
            else:
                singletons0.append(a)
        elif a == b - 2:
            if a & CUTOFF:
                singletons1.append(a & ~CUTOFF)
                singletons1.append((a + 1) & ~CUTOFF)
            else:
                singletons0.append(a)
                singletons0.append(a + 1)
        else:
            if a >= 2 * CUTOFF:
                extra.append((a, b - a))
            elif a & CUTOFF:
                normal1.append((a & ~CUTOFF, b - a))
            else:
                normal0.append((a, b - a))

    singletons0u, singletons0l = compress_singletons(singletons0)
    singletons1u, singletons1l = compress_singletons(singletons1)
    normal0 = compress_normal(normal0)
    normal1 = compress_normal(normal1)

    print("""\
FMT_FUNC auto is_printable(uint32_t cp) -> bool {\
""")
    print_singletons(singletons0u, singletons0l, 'singletons0', 'singletons0_lower')
    print_singletons(singletons1u, singletons1l, 'singletons1', 'singletons1_lower')
    print_normal(normal0, 'normal0')
    print_normal(normal1, 'normal1')
    print("""\
  auto lower = static_cast<uint16_t>(cp);
  if (cp < 0x10000) {
    return is_printable(lower, singletons0,
                        sizeof(singletons0) / sizeof(*singletons0),
                        singletons0_lower, normal0, sizeof(normal0));
  }
  if (cp < 0x20000) {
    return is_printable(lower, singletons1,
                        sizeof(singletons1) / sizeof(*singletons1),
                        singletons1_lower, normal1, sizeof(normal1));
  }\
""")
    for a, b in extra:
        print("  if (0x{:x} <= cp && cp < 0x{:x}) return false;".format(a, a + b))
    print("""\
  return cp < 0x{:x};
}}\
""".format(NUM_CODEPOINTS))

if __name__ == '__main__':
    main()
