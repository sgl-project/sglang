# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""llguidance utils"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List, Dict, Tuple, Iterator
from abc import ABC, abstractmethod
import re


@dataclass
class Position:
    """Tracks position in source text for error reporting"""

    text: str
    pos: int

    def advance(self, n: int = 1) -> "Position":
        return Position(self.text, self.pos + n)

    def current(self) -> str:
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def peek(self, n: int = 1) -> str:
        return self.text[self.pos : self.pos + n]

    def __str__(self) -> str:
        line_no = self.text.count("\n", 0, self.pos) + 1
        pref = self.text[max(0, self.pos - 20) : self.pos]
        suff = self.text[self.pos : self.pos + 20]
        return f"line {line_no}, {repr(pref)} ^ {repr(suff)}"


class ParseError(Exception):
    def __init__(self, pos: Position, message: str):
        self.pos = pos
        super().__init__(f"{message} at {pos}")


# AST Node Classes
class ASTNode(ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass

    def is_atomic(self) -> bool:
        return True

    def is_terminal(self) -> bool:
        return all(c.is_terminal() for c in self.children())

    def top_str(self) -> str:
        return self.__str__()

    def simplify(self) -> "ASTNode":
        return self

    def children(self) -> list["ASTNode"]:
        return []


@dataclass
class LiteralNode(ASTNode):
    value: str

    def __str__(self) -> str:
        return f'"{self.value}"'


@dataclass
class RegexNode(ASTNode):
    rx: str

    def __str__(self) -> str:
        return f"/{self.rx}/"


@dataclass
class RuleRefNode(ASTNode):
    name: str
    target: Optional["RuleNode"] = None

    def is_terminal(self) -> bool:
        if self.target is None:
            return False
        return self.target.rule_is_terminal

    def __str__(self) -> str:
        if self.target is None:
            return self.name
        return self.target.name


@dataclass
class RepetitionNode(ASTNode):
    node: ASTNode
    min_times: int
    max_times: Optional[int]  # None represents unlimited

    def children(self) -> List[ASTNode]:
        return [self.node]

    def simplify(self) -> ASTNode:
        self.node = self.node.simplify()
        return self

    def __str__(self) -> str:
        inner = str(self.node)
        if not self.node.is_atomic():
            inner = f"({inner})"
        if self.min_times == 0 and self.max_times is None:
            return f"{inner}*"
        if self.min_times == 1 and self.max_times is None:
            return f"{inner}+"
        if self.min_times == 0 and self.max_times == 1:
            return f"{inner}?"
        max_str = str(self.max_times) if self.max_times is not None else ""
        return f"{inner}{{{self.min_times},{max_str}}}"


@dataclass
class SequenceNode(ASTNode):
    nodes: List[ASTNode]

    def __str__(self) -> str:
        if not self.nodes:
            return '""'
        return " ".join(str(node) for node in self.nodes)

    def is_atomic(self) -> bool:
        return False

    def simplify(self) -> ASTNode:
        for i in range(len(self.nodes)):
            self.nodes[i] = self.nodes[i].simplify()
        if len(self.nodes) == 1:
            return self.nodes[0]
        return self

    def children(self) -> list[ASTNode]:
        return self.nodes


@dataclass
class AlternativeNode(ASTNode):
    alternatives: List[ASTNode]

    def top_str(self) -> str:
        return "\n     | ".join(str(alt) for alt in self.alternatives)

    def __str__(self) -> str:
        return "(" + " | ".join(str(alt) for alt in self.alternatives) + ")"

    def is_atomic(self) -> bool:
        return False

    def simplify(self) -> ASTNode:
        for i in range(len(self.alternatives)):
            self.alternatives[i] = self.alternatives[i].simplify()
        if len(self.alternatives) == 1:
            return self.alternatives[0]
        return self

    def children(self) -> list[ASTNode]:
        return self.alternatives


@dataclass
class RuleNode(ASTNode):
    name: str
    alternatives: ASTNode
    comment: str
    rule_is_terminal: bool = False
    order = 0

    def children(self) -> List[ASTNode]:
        return [self.alternatives]

    def __str__(self) -> str:
        return f"{self.comment}{self.name}: {self.alternatives.top_str()}"


class GrammarParser:
    def __init__(self):
        self.curr_comment = ""
        pass

    def parse(self, text: str) -> dict[str, RuleNode]:
        pos = Position(text, 0)
        pos = self._skip_space(pos, allow_newlines=True)
        rules: list[RuleNode] = []

        while pos.current():
            rule, pos = self._parse_rule(pos)
            rules.append(rule)
            pos = self._skip_space(pos, allow_newlines=True)

        return {rule.name: rule for rule in rules}

    def _parse_char(self, pos: Position) -> Tuple[str, Position]:
        def is_all_hex(s: str) -> bool:
            return all(ch in "0123456789abcdefABCDEF" for ch in s)

        if pos.current() == "\\":
            if not pos.peek(2)[1]:
                raise ParseError(pos, "Incomplete escape sequence")
            pos = pos.advance()
            c = pos.current()
            if c in '"\\[]nrt':
                return "\\" + c, pos.advance()
            elif c == "x":
                hex_value = pos.peek(3)[1:3]
                if len(hex_value) != 2 or not is_all_hex(hex_value):
                    raise ParseError(
                        pos, f"Invalid \\x escape sequence: \\x{hex_value}"
                    )
                pos = pos.advance(3)
                return f"\\x{hex_value}", pos
            elif c == "u":
                hex_value = pos.peek(5)[1:5]
                if len(hex_value) != 4 or not is_all_hex(hex_value):
                    raise ParseError(
                        pos, f"Invalid \\u escape sequence: \\u{hex_value}"
                    )
                pos = pos.advance(5)
                return f"\\u{hex_value.lstrip('0')}", pos
            elif c == "U":
                hex_value = pos.peek(9)[1:9]
                if len(hex_value) != 8 or not is_all_hex(hex_value):
                    raise ParseError(
                        pos, f"Invalid \\U escape sequence: \\U{hex_value}"
                    )
                pos = pos.advance(9)
                return f"\\U{hex_value.lstrip('0')}", pos
            else:
                raise ParseError(pos, f"Invalid escape sequence \\{c}")
        elif pos.current() == "":
            raise ParseError(pos, "Unexpected end of input")

        return pos.current(), pos.advance()

    def _parse_char_class(self, pos: Position) -> Tuple[ASTNode, Position]:
        if pos.current() != "[":
            raise ParseError(pos, "Expected '['")
        r = "["
        pos = pos.advance()

        while True:
            c, pos = self._parse_char(pos)
            if c in "/[":
                r += "\\" + c
            else:
                r += c
            if c == "]":
                break

        return RegexNode(r), pos

    def _parse_literal(self, pos: Position) -> Tuple[ASTNode, Position]:
        if pos.current() != '"':
            raise ParseError(pos, "Expected '\"'")
        pos = pos.advance()
        r = ""

        while True:
            c, pos = self._parse_char(pos)
            if c == '"':
                break
            r += c

        return LiteralNode(r), pos

    @staticmethod
    def _parse_name(pos: Position) -> Tuple[str, Position]:
        start = pos.pos
        while GrammarParser._is_word_char(pos.current()):
            pos = pos.advance()
        if pos.pos == start:
            raise ParseError(pos, "Expected name")
        return pos.text[start : pos.pos], pos

    @staticmethod
    def _parse_int(pos: Position) -> Tuple[int, Position]:
        start = pos.pos
        while pos.current().isdigit():
            pos = pos.advance()
        if pos.pos == start:
            raise ParseError(pos, "Expected integer")
        return int(pos.text[start : pos.pos]), pos

    def _skip_space(self, pos: Position, allow_newlines: bool) -> Position:
        while pos.current():
            if pos.current() in " \t":
                pos = pos.advance()
            elif allow_newlines and pos.current() in "\r\n":
                pos = GrammarParser._skip_newline(pos)
            elif pos.current() == "#":
                pos = pos.advance()
                cmt = "//"
                while pos.current() and pos.current() not in "\r\n":
                    cmt += pos.current()
                    pos = pos.advance()
                self.curr_comment += cmt + "\n"
            else:
                break
        return pos

    @staticmethod
    def _skip_newline(pos: Position) -> Position:
        if pos.current() == "\r":
            pos = pos.advance()
            if pos.current() == "\n":
                pos = pos.advance()
        elif pos.current() == "\n":
            pos = pos.advance()
        return pos

    @staticmethod
    def _is_word_char(c: str) -> bool:
        return c.isalnum() or c == "-"

    def _parse_rule(self, pos: Position) -> Tuple[RuleNode, Position]:
        name, pos = self._parse_name(pos)
        pos = self._skip_space(pos, allow_newlines=False)

        if pos.peek(3) != "::=":
            raise ParseError(pos, "Expected ::=")
        pos = pos.advance(3)

        pos = self._skip_space(pos, allow_newlines=True)
        alternatives, pos = self._parse_alternatives(pos, is_nested=False)

        pos = self._skip_newline(pos)
        cmt = self.curr_comment
        self.curr_comment = ""
        return RuleNode(name, alternatives, cmt), pos

    def _parse_alternatives(
        self, pos: Position, is_nested: bool
    ) -> Tuple[AlternativeNode, Position]:
        alternatives: list[ASTNode] = []

        while True:
            sequence, pos = self._parse_sequence(pos, is_nested)
            alternatives.append(sequence)

            pos = self._skip_space(pos, allow_newlines=is_nested)
            if pos.current() != "|":
                break

            pos = pos.advance()
            pos = self._skip_space(pos, allow_newlines=True)

        return AlternativeNode(alternatives), pos

    def _parse_sequence(
        self, pos: Position, is_nested: bool
    ) -> Tuple[SequenceNode, Position]:
        nodes: List[ASTNode] = []

        while (
            pos.current()
            and pos.current() not in "|)"
            and (is_nested or pos.current() not in "\r\n")
        ):
            if pos.current() == '"':
                node, pos = self._parse_literal(pos)
                nodes.append(node)
            elif pos.current() == "[":
                node, pos = self._parse_char_class(pos)
                nodes.append(node)
            elif pos.current() == "(":
                node, pos = self._parse_group(pos, is_nested=is_nested)
                nodes.append(node)
            elif pos.current() == ".":
                nodes.append(RegexNode("."))
                pos = pos.advance()
            elif self._is_word_char(pos.current()):
                name, pos = self._parse_name(pos)
                nodes.append(RuleRefNode(name))
            else:
                break

            pos = self._skip_space(pos, allow_newlines=is_nested)
            pos = self._parse_repetition(pos, nodes)
            pos = self._skip_space(pos, allow_newlines=is_nested)

        return SequenceNode(nodes), pos

    def _parse_group(self, pos: Position, is_nested: bool) -> Tuple[ASTNode, Position]:
        if pos.current() != "(":
            raise ParseError(pos, "Expected '('")
        pos = pos.advance()
        pos = self._skip_space(pos, True)

        alternatives, pos = self._parse_alternatives(pos, is_nested=True)

        if pos.current() != ")":
            raise ParseError(pos, "Expected ')'")
        pos = pos.advance()

        return alternatives, self._skip_space(pos, is_nested)

    def _parse_repetition(self, pos: Position, nodes: List[ASTNode]) -> Position:
        if not nodes:
            return pos

        if pos.current() == "*":
            nodes[-1] = RepetitionNode(nodes[-1], 0, None)
            return pos.advance()
        elif pos.current() == "+":
            nodes[-1] = RepetitionNode(nodes[-1], 1, None)
            return pos.advance()
        elif pos.current() == "?":
            nodes[-1] = RepetitionNode(nodes[-1], 0, 1)
            return pos.advance()
        elif pos.current() == "{":
            pos = pos.advance()
            pos = self._skip_space(pos, True)
            min_times, pos = self._parse_int(pos)
            pos = self._skip_space(pos, True)

            if pos.current() == "}":
                nodes[-1] = RepetitionNode(nodes[-1], min_times, min_times)
                return pos.advance()
            elif pos.current() == ",":
                pos = self._skip_space(pos.advance(), True)
                max_times = None
                if pos.current().isdigit():
                    max_times, pos = self._parse_int(pos)
                pos = self._skip_space(pos, True)
                if pos.current() != "}":
                    raise ParseError(pos, "Expected '}'")
                nodes[-1] = RepetitionNode(nodes[-1], min_times, max_times)
                return pos.advance()
            else:
                raise ParseError(pos, "Expected ',' or '}'")

        return pos


def resolve(rules: dict[str, RuleNode]):
    def rename(r: RuleNode, name: str):
        if name in rules:
            raise Exception(f"Rule '{name}' already exists")
        del rules[r.name]
        r.name = name
        rules[name] = r

    for i, r in enumerate(rules.values()):
        r.order = i
        r.alternatives = r.alternatives.simplify()

    def all_children(node: ASTNode) -> Iterator[ASTNode]:
        for c in node.children():
            yield c
            yield from all_children(c)

    for r in rules.values():
        for node in all_children(r):
            if isinstance(node, RuleRefNode):
                if node.name not in rules:
                    raise Exception(f"Rule '{node.name}' not found")
                node.target = rules[node.name]

    if "root" not in rules:
        raise Exception("No 'root' rule found")
    rename(rules["root"], "start")

    num_fix = 1
    while num_fix > 0:
        num_fix = 0
        for r in rules.values():
            if (
                r.name != "start"
                and not r.rule_is_terminal
                and r.alternatives.is_terminal()
            ):
                r.rule_is_terminal = True
                num_fix += 1

    for r in list(rules.values()):
        new_name = r.name.replace("-", "_")
        # convert fooBar_Baz to foo_bar_baz
        new_name = re.sub(r"([a-z])([A-Z])", r"\1_\2", new_name).lower()
        if r.rule_is_terminal:
            new_name = new_name.upper()
        else:
            new_name = new_name.lower()
        if r.name != new_name:
            rename(r, new_name)

def ebnf_to_lark(ebnf: str) -> str:
    parser = GrammarParser()
    rules = parser.parse(ebnf)
    resolve(rules)
    rlist = list(rules.values())
    rlist.sort(key=lambda r: r.order)
    
    lark_grm = "%llguidance {}\n\n"
    prev_nl = True
    for r in rlist:
        s = str(r)

        if not prev_nl and "\n" in s:
            lark_grm += "\n"

        lark_grm += s + "\n"

        prev_nl = "\n" in s
        if prev_nl:
            lark_grm += "\n"

    return lark_grm