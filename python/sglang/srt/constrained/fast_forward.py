import interegular
from sglang.srt.constrained.disk_cache import disk_cache
from sglang.srt.constrained.regex import FSMInfo, make_deterministic_fsm

IP_REGEX = r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"


class FastForwardMap:
    def __init__(self, regex_string):
        @disk_cache()
        def _init_state_to_fast_forward(regex_string):
            regex_pattern = interegular.parse_pattern(regex_string)
            regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm().reduce())

            fsm_info: FSMInfo = regex_fsm.fsm_info

            symbol_to_id = fsm_info.alphabet_symbol_mapping
            id_to_symbol = {}
            for symbol, id_ in symbol_to_id.items():
                id_to_symbol.setdefault(id_, []).append(symbol)

            transitions = fsm_info.transitions
            dirty_states = set()
            state_to_fast_forward = {}

            for (state, id_), next_state in transitions.items():
                if state in dirty_states:
                    continue
                if state in state_to_fast_forward:
                    dirty_states.add(state)
                    del state_to_fast_forward[state]
                    continue
                if len(id_to_symbol[id_]) > 1:
                    dirty_states.add(state)
                    continue

                state_to_fast_forward[state] = (id_to_symbol[id_][0], next_state)

            return state_to_fast_forward

        self.state_to_fast_forward = _init_state_to_fast_forward(regex_string)

    def valid_states(self):
        return self.state_to_fast_forward.keys()

    def fast_forward(self, state):
        if state not in self.state_to_fast_forward:
            return None

        fast_forward_str = ""
        next_state = None
        while state in self.state_to_fast_forward:
            symbol, next_state = self.state_to_fast_forward[state]
            fast_forward_str += symbol
            state = next_state
        return fast_forward_str, next_state


class FastForwardCache:
    def __init__(self):
        self.cache = {}

    def init_fast_forward_map(self, regex_string):
        if regex_string not in self.cache:
            fast_forward_map = FastForwardMap(regex_string)
            self.cache[regex_string] = fast_forward_map
        return self.cache[regex_string]


def test_main():
    regex_string = r"The google's DNS sever address is " + IP_REGEX
    fast_forward_map = FastForwardMap(regex_string)
    for state in fast_forward_map.valid_states():
        print(state, f'"{fast_forward_map.fast_forward(state)}"')


if __name__ == "__main__":
    test_main()
