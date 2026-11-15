"""A SASS check for the ptxas register-input (RS) wgmma race, run on JIT builds that issue wgmma from registers.

ptxas through CUDA 13.2 can let code after an RS wgmma overwrite that wgmma's A registers while it is still in
flight: it copy-propagates across ``wgmma.wait_group N`` with N >= 1 and drops the moves that protected them (the CUDA
13.2 known issue "Data Race in WGMMA A/B Register Copy Propagation", fixed in 13.3). The result is a silent,
timing-dependent wrong answer that a unit test can miss, so a module built from such kernels passes
``require_no_rs_wgmma_hazard`` as its ``load_jit(post_build=...)`` and is never loaded or cached when the check fails.

Model (conservative): every GMMA joins the open group and the one carrying gsb0 closes it; ``WARPGROUP.DEPBAR.LE gsb0,
N`` retires all but the newest N closed groups. An RS HGMMA reads 4 A registers. The walk is a fixpoint over the
function's control-flow graph, so a group kept in flight across a loop back edge is checked against the next
iteration. A violation is an instruction (or a later GMMA's accumulator) writing a pending RS HGMMA's A registers, or
a non-GMMA instruction touching a pending GMMA's accumulators.
"""

from __future__ import annotations

import collections
import os
import pathlib
import re
import subprocess
from typing import Dict, List, Tuple

_LINE = re.compile(r"^\s+/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
_FUNC = re.compile(r"Function : (\S+)")
_PRED = re.compile(r"^(@!?U?P[T0-9]+)\s+")
_REG = re.compile(r"^R(\d+)$")
_ANYREG = re.compile(r"(?<![A-Z_])R(\d+)(\.64|\.128)?(?![0-9])")
_ISPRED = re.compile(r"^!?U?P[T0-9]$")
_GMMA = re.compile(r"^[HQIB]GMMA\.64x(\d+)x(\d+)\.(F32|F16|S32)")
_MEM_LD = (
    "LD",
    "LDG",
    "LDS",
    "LDL",
    "LDC",
    "ATOM",
    "ATOMG",
    "ATOMS",
    "LDGSTS",
    "SUATOM",
    "SULD",
)
_MAX_GROUPS = 16
_MAX_STATES = 4096


def _split_ops(s: str) -> List[str]:
    out, depth, cur = [], 0, ""
    for ch in s:
        depth += ch in "[("
        depth -= ch in "])"
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _dest_regs(op: str, ops: List[str]) -> set:
    """Registers an instruction writes (R only: uniform and predicate registers cannot alias an A operand)."""
    if not ops:
        return set()
    first = ops[0]
    if _ISPRED.match(first) and len(ops) > 1 and _REG.match(ops[1]):
        first = ops[
            1
        ]  # SHFL PT, R3, ... and LOP3.LUT P0, R2, ... write the second operand
    m = _REG.match(first)
    if not m:
        return set()
    parts = op.split(".")
    base, w = parts[0], 1
    if base in _MEM_LD:
        w = 4 if ".128" in op else 2 if ".64" in op else 1
    elif base in ("LDSM", "MOVM"):
        w = 4 if parts[-1] in ("4", "x4") else 2 if parts[-1] in ("2", "x2") else 1
    elif base == "IMAD" and "WIDE" in parts:
        w = 2
    elif base == "CS2R":
        w = 1 if "32" in parts else 2
    elif base in ("DADD", "DMUL", "DFMA", "DMNMX", "DMMA"):
        w = 2
    elif (
        base in ("F2F", "I2F", "F2I", "I2I")
        and len(parts) > 1
        and parts[1] in ("F64", "S64", "U64")
    ):
        w = 2
    elif base in ("HMMA", "IMMA"):
        acc32 = any(p in ("F32", "S32") for p in parts[2:3])
        w = {
            "16816": 4 if acc32 else 2,
            "1688": 4 if acc32 else 2,
            "8816": 2,
            "8832": 2,
        }.get(parts[1] if len(parts) > 1 else "", 4)
    return set(range(int(m.group(1)), int(m.group(1)) + w))


class _Ins:
    __slots__ = (
        "addr",
        "pred",
        "op",
        "ops",
        "text",
        "dst",
        "src",
        "gmma",
        "depbar",
        "succ",
        "back",
        "end",
    )

    def __init__(self, addr: int, text: str):
        self.addr, self.text = addr, text
        m = _PRED.match(text)
        self.pred = m.group(1) if m else ""
        parts = (text[m.end() :] if m else text).split(None, 1)
        self.op = parts[0]
        self.ops = _split_ops(parts[1]) if len(parts) > 1 else []
        self.gmma, self.depbar, self.end, self.back = None, None, False, False
        self.dst, self.src, self.succ = set(), set(), []
        g = _GMMA.match(self.op)
        if g:
            n = int(g.group(1))
            w = n // 4 if g.group(3) == "F16" else n // 2
            d = _REG.match(self.ops[0])
            a = _REG.match(self.ops[1]) if len(self.ops) > 1 else None
            self.gmma = dict(
                acc=set(range(int(d.group(1)), int(d.group(1)) + w)) if d else set(),
                a=set(range(int(a.group(1)), int(a.group(1)) + 4)) if a else set(),
                rs=bool(a),
                gsb0="gsb0" in self.ops,
            )
        elif self.op == "WARPGROUP.DEPBAR.LE" and self.ops and self.ops[0] == "gsb0":
            self.depbar = int(self.ops[1], 0)
        else:
            self.dst = _dest_regs(self.op, self.ops)
            for o in self.ops[1:] if self.dst else self.ops:
                for m2 in _ANYREG.finditer(o):
                    r = int(m2.group(1))
                    self.src.update(
                        range(
                            r,
                            r
                            + (
                                4
                                if m2.group(2) == ".128"
                                else 2
                                if m2.group(2) == ".64"
                                else 1
                            ),
                        )
                    )


# Callers put this in their load_jit key, so that a change to the check also rebuilds and rechecks cached builds.
CHECK_VERSION = "2"


def parse_sass(text: str) -> Dict[str, List[_Ins]]:
    """{function name: instructions} from ``cuobjdump -sass`` output."""
    funcs, cur = collections.OrderedDict(), None
    for line in text.split("\n"):
        m = _FUNC.search(line)
        if m:
            cur = funcs[m.group(1)] = []
        elif cur is not None and (m := _LINE.match(line)):
            cur.append(_Ins(int(m.group(1), 16), m.group(2)))
    return funcs


def _link(ins: List[_Ins]) -> bool:
    """Fill in successors; False if the control flow could not be resolved (an indirect branch)."""
    idx = {x.addr: i for i, x in enumerate(ins)}
    ok = True
    for i, x in enumerate(ins):
        base = x.op.split(".")[0]
        nxt = [i + 1] if i + 1 < len(ins) else []
        always = x.pred in ("", "@PT")
        tgt = None
        for o in x.ops:
            if m := re.match(r"^`?\(?(0x[0-9a-f]+)", o):
                tgt = int(m.group(1), 16)
        if x.pred == "@!PT":
            x.succ = nxt
        elif base in ("BRA", "JMP", "CALL"):
            t = [idx[tgt]] if tgt in idx else []
            ok &= tgt is None or tgt in idx
            x.succ = t + nxt if base == "CALL" or not always else t
            x.back = bool(t) and ins[t[0]].addr <= x.addr
        elif base in ("BRX", "JMX"):
            ok = False
            x.succ = nxt
        elif base in ("EXIT", "RET", "BPT", "KILL") and always:
            x.end = True
        else:
            x.succ = nxt
    return ok


def check_function(ins: List[_Ins]) -> Tuple[int, List[str], bool]:
    """(RS GMMA count, violations, analysis complete) for one function."""
    complete = _link(ins)
    if not ins:
        return 0, [], complete
    gm = {i: x.gmma for i, x in enumerate(ins) if x.gmma}
    states = [set() for _ in ins]
    states[0].add(((), ()))
    work, queued, viol = [0], {0}, {}
    while work:
        i = work.pop()
        queued.discard(i)
        x = ins[i]
        outs = set()
        # A predicated wgmma or wait may not run, so both outcomes flow on.
        maybe = x.pred not in ("", "@PT")
        for groups, opn in states[i]:
            pend = [h for g in groups for h in g] + list(opn)
            if x.pred == "@!PT":
                outs.add((groups, opn))
                continue
            if maybe and (x.gmma or x.depbar is not None):
                outs.add((groups, opn))
            if x.gmma:
                for h in pend:
                    if gm[h]["rs"] and x.gmma["acc"] & gm[h]["a"]:
                        viol[(h, i, "A-write")] = x.gmma["acc"] & gm[h]["a"]
                opn2 = opn + (i,)
                if x.gmma["gsb0"]:
                    groups, opn2 = groups + (opn2,), ()
                    if len(groups) > _MAX_GROUPS:
                        groups, complete = groups[-_MAX_GROUPS:], False
                outs.add((groups, opn2))
                continue
            if x.depbar is not None:
                outs.add((groups[len(groups) - x.depbar :] if x.depbar else (), opn))
                continue
            for h in pend:
                if gm[h]["rs"] and x.dst & gm[h]["a"]:
                    viol[(h, i, "A-write")] = x.dst & gm[h]["a"]
                if (x.dst | x.src) & gm[h]["acc"]:
                    viol[(h, i, "acc-touch")] = (x.dst | x.src) & gm[h]["acc"]
            outs.add((groups, opn))
        for s in x.succ:
            new = outs - states[s]
            if new:
                if len(states[s]) + len(new) > _MAX_STATES:
                    return sum(g["rs"] for g in gm.values()), [], False
                states[s] |= new
                if s not in queued:
                    queued.add(s)
                    work.append(s)
    msgs = [
        f"{kind}: {ins[h].text!r} at {ins[h].addr:#06x} pending when {ins[i].text!r} at {ins[i].addr:#06x} "
        f"touches R{sorted(regs)}"
        for (h, i, kind), regs in sorted(viol.items())
    ]
    return sum(g["rs"] for g in gm.values()), msgs, complete


def check_sass(text: str) -> Tuple[int, List[str]]:
    """(RS GMMA count, problems) over every function of a SASS dump; an incomplete analysis is a problem."""
    rs_total, problems = 0, []
    for name, ins in parse_sass(text).items():
        rs, viol, complete = check_function(ins)
        rs_total += rs
        problems += [f"{name}: {v}" for v in viol]
        if not complete:
            problems.append(
                f"{name}: control flow or in-flight groups beyond what the check models"
            )
    return rs_total, problems


def require_no_rs_wgmma_hazard(library: pathlib.Path) -> None:
    """Raise unless the built object's SASS has RS wgmmas and none of them can have its registers overwritten."""
    from sglang.kernels.jit.utils.compile.toolchain import cuda_home

    cuobjdump = os.path.join(cuda_home(), "bin", "cuobjdump")
    p = subprocess.run(
        [cuobjdump, "-sass", str(library)], capture_output=True, text=True
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"cuobjdump -sass {library} failed: {p.stderr.strip()[-2000:]}"
        )
    rs, problems = check_sass(p.stdout)
    if problems or rs == 0:
        raise RuntimeError(
            f"Refusing JIT build {library.name}: the register-input wgmma check found "
            + (
                f"{len(problems)} problems:\n" + "\n".join(problems[:20])
                if problems
                else "no RS wgmma in its SASS"
            )
        )
