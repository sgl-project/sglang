"""
Faster constrained decoding.
Reference: https://lmsys.org/blog/2024-02-05-compressed-fsm/
"""

import dataclasses
from collections import defaultdict

import interegular
import outlines.caching

from sglang.srt.constrained import (
    FSMInfo,
    disk_cache,
    make_byte_level_fsm,
    make_deterministic_fsm,
)
from sglang.srt.constrained.base_cache import BaseCache

IP_REGEX = r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"


@dataclasses.dataclass
class JumpEdge:
    symbol: str = None
    symbol_next_state: int = None
    byte: int = None
    byte_next_state: int = None


class JumpForwardMap:
    def __init__(self, regex_string):
        @disk_cache()
        def _init_state_to_jump_forward(regex_string):
            regex_pattern = interegular.parse_pattern(regex_string)

            byte_fsm = make_byte_level_fsm(
                regex_pattern.to_fsm().reduce(), keep_utf8=True
            )
            regex_fsm, _ = make_deterministic_fsm(byte_fsm)

            fsm_info: FSMInfo = regex_fsm.fsm_info

            symbol_to_id = fsm_info.alphabet_symbol_mapping
            id_to_symbol = {}
            for symbol, id_ in symbol_to_id.items():
                id_to_symbol.setdefault(id_, []).append(symbol)

            transitions = fsm_info.transitions
            outgoings_ct = defaultdict(int)
            state_to_jump_forward = {}

            for (state, id_), next_state in transitions.items():
                if id_ == fsm_info.alphabet_anything_value:
                    continue
                symbols = id_to_symbol[id_]
                for c in symbols:
                    if len(c) > 1:
                        # Skip byte level transitions
                        continue

                    outgoings_ct[state] += 1
                    if outgoings_ct[state] > 1:
                        if state in state_to_jump_forward:
                            del state_to_jump_forward[state]
                        break

                    state_to_jump_forward[state] = JumpEdge(
                        symbol=c,
                        symbol_next_state=next_state,
                    )

            # Process the byte level jump forward
            outgoings_ct = defaultdict(int)
            for (state, id_), next_state in transitions.items():
                if id_ == fsm_info.alphabet_anything_value:
                    continue
                symbols = id_to_symbol[id_]
                for c in symbols:
                    byte_ = None
                    if len(c) == 1 and ord(c) < 0x80:
                        # ASCII character
                        byte_ = ord(c)
                    elif len(c) > 1:
                        # FIXME: This logic is due to the leading \x00
                        # https://github.com/outlines-dev/outlines/pull/930
                        byte_ = int(symbols[0][1:], 16)

                    if byte_ is not None:
                        outgoings_ct[state] += 1
                        if outgoings_ct[state] > 1:
                            if state in state_to_jump_forward:
                                del state_to_jump_forward[state]
                            break
                        e = state_to_jump_forward.get(state, JumpEdge())
                        e.byte = byte_
                        e.byte_next_state = next_state
                        state_to_jump_forward[state] = e

            return state_to_jump_forward

        self.state_to_jump_forward = _init_state_to_jump_forward(regex_string)

    def jump_forward_symbol(self, state):
        jump_forward_str = ""
        next_state = state
        while state in self.state_to_jump_forward:
            e = self.state_to_jump_forward[state]
            if e.symbol is None:
                break
            jump_forward_str += e.symbol
            next_state = e.symbol_next_state
            state = next_state

        return jump_forward_str, next_state

    def jump_forward_byte(self, state):
        if state not in self.state_to_jump_forward:
            return None

        jump_forward_bytes = []
        next_state = None
        while state in self.state_to_jump_forward:
            e = self.state_to_jump_forward[state]
            assert e.byte is not None and e.byte_next_state is not None
            jump_forward_bytes.append((e.byte, e.byte_next_state))
            next_state = e.byte_next_state
            state = next_state

        return jump_forward_bytes

    def is_jump_forward_symbol_state(self, state):
        return (
            state in self.state_to_jump_forward
            and self.state_to_jump_forward[state].symbol is not None
        )


class JumpForwardCache(BaseCache):
    def __init__(self):
        super().__init__()

    def init_value(self, regex):
        return JumpForwardMap(regex)


def test_main(regex_string):
    jump_forward_map = JumpForwardMap(regex_string)
    for state, e in jump_forward_map.state_to_jump_forward.items():
        if e.symbol is not None:
            jump_forward_str, next_state = jump_forward_map.jump_forward_symbol(state)
            print(f"{state} -> {next_state}", jump_forward_str)
        bytes_ = jump_forward_map.jump_forward_byte(state)
        print(f"{state} -> {bytes_[-1][1]}", [hex(b) for b, _ in bytes_])


if __name__ == "__main__":
    import outlines

    outlines.caching.clear_cache()
    test_main(r"The google's DNS sever address is " + IP_REGEX)
    test_main(r"霍格沃茨特快列车|霍比特人比尔博")
    # 霍格: \xe9\x9c\x8d \xe6\xa0\xbc ...
    # 霍比: \xe9\x9c\x8d \xe6\xaf\x94 ...
