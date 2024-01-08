# Adapted from:
# https://github.com/outlines-dev/outlines/blob/0355ab4272a5d7e4d94c4a53a52593f885b81a61/outlines/fsm/regex.py
from collections import namedtuple
from functools import lru_cache
from typing import Dict, Generator, List, Sequence, Set, Tuple

import numba
import numpy as np
from interegular.fsm import FSM, Alphabet, OblivionError, anything_else
from numba.typed.typedobjectutils import _nonoptional
from sglang.srt.constrained.tokenizer import Tokenizer


class BetterAlphabet(Alphabet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert anything_else in self._symbol_mapping
        self.anything_value = self._symbol_mapping[anything_else]

    def __getitem__(self, item):
        return self._symbol_mapping.get(item, self.anything_value)

    def copy(self):
        return BetterAlphabet(self._symbol_mapping.copy())


class BetterFSM(FSM):
    flat_transition_map: Dict[Tuple[int, int], int]
    trans_key_to_states: Dict[int, List[int]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(self.alphabet, BetterAlphabet):
            self.__dict__["alphabet"] = BetterAlphabet(self.alphabet._symbol_mapping)

        flat_transition_map = {}
        trans_key_to_states = {}
        for from_state, trans_map in self.map.items():
            for trans_key, to_state in trans_map.items():
                flat_transition_map[(from_state, trans_key)] = to_state
                trans_key_to_states.setdefault(trans_key, set()).add(from_state)

        self.__dict__["trans_key_to_states"] = trans_key_to_states
        self.__dict__["flat_transition_map"] = flat_transition_map
        self.__dict__["_fsm_info"] = None

    def copy(self):
        return BetterFSM(
            alphabet=self.alphabet.copy(),
            states=self.states.copy(),
            initial=self.initial,
            finals=self.finals.copy(),
            map=self.map.copy(),
            __no_validation__=True,
        )

    @property
    def fsm_info(self):
        if self._fsm_info is None:
            flat_transition_map_items = np.fromiter(
                ((a[0], a[1], b) for a, b in self.flat_transition_map.items()),
                dtype=np.dtype("i8, i8, i8"),
            )
            trans_key_to_states_items = np.fromiter(
                ((k, z) for k, v in self.trans_key_to_states.items() for z in v),
                dtype=np.dtype("i8, i8"),
            )
            alphabet_symbol_mapping_items = np.fromiter(
                (
                    it
                    for it in self.alphabet._symbol_mapping.items()
                    if it[0] != anything_else
                ),
                dtype=np.dtype("U1, i8"),
            )
            nb_finals = np.fromiter(self.finals, dtype=np.dtype("i8"))
            self.__dict__["_fsm_info"] = create_fsm_info(
                self.initial,
                nb_finals,
                flat_transition_map_items,
                trans_key_to_states_items,
                self.alphabet.anything_value,
                alphabet_symbol_mapping_items,
            )

        return self._fsm_info


nb_int_list_type = numba.types.ListType(numba.int64)
nb_int_pair_type = numba.types.UniTuple(numba.int64, 2)
nb_unichar_1_type = numba.types.UnicodeCharSeq(1)


@numba.njit(cache=True)
def create_fsm_info(
    py_initial,
    py_finals,
    flat_transition_map_items,
    trans_key_to_states_items,
    py_anything_value,
    alphabet_symbol_mapping_items,
):
    trans_key_to_states = numba.typed.Dict.empty(numba.int64, nb_int_list_type)
    for trans_key_and_state in trans_key_to_states_items:
        trans_key_to_states.setdefault(
            trans_key_and_state[0], numba.typed.List.empty_list(numba.int64)
        ).append(trans_key_and_state[1])

    flat_transition_map = numba.typed.Dict.empty(nb_int_pair_type, numba.int64)
    for trans_key_and_state in flat_transition_map_items:
        flat_transition_map[
            (trans_key_and_state[0], trans_key_and_state[1])
        ] = trans_key_and_state[2]

    alphabet_symbol_map = numba.typed.Dict.empty(nb_unichar_1_type, numba.int64)
    for symbol_and_trans_key in alphabet_symbol_mapping_items:
        alphabet_symbol_map[symbol_and_trans_key[0]] = symbol_and_trans_key[1]

    initial = numba.int64(py_initial)

    finals = set()
    for final in py_finals:
        finals.add(final)

    anything_value = numba.int64(py_anything_value)

    return FSMInfo(
        initial,
        finals,
        flat_transition_map,
        trans_key_to_states,
        anything_value,
        alphabet_symbol_map,
    )


FSMInfo = namedtuple(
    "FSMInfo",
    [
        "initial",
        "finals",
        "transitions",
        "trans_key_to_states",
        "alphabet_anything_value",
        "alphabet_symbol_mapping",
    ],
)


def make_deterministic_fsm(fsm: FSM) -> Tuple[BetterFSM, Dict[int, int]]:
    """Construct an equivalent FSM with deterministic state labels."""
    old_to_new_trans_keys = {
        trans_key: i
        for i, (trans_key, _) in enumerate(
            sorted(fsm.alphabet.by_transition.items(), key=lambda x: sorted(x[1]))
        )
    }

    new_symbol_mapping = {
        symbol: old_to_new_trans_keys[trans_key]
        for symbol, trans_key in fsm.alphabet._symbol_mapping.items()
    }

    new_alphabet = BetterAlphabet(new_symbol_mapping)

    new_map = {
        from_state: {
            old_to_new_trans_keys[trans_key]: to_state
            for trans_key, to_state in trans_map.items()
        }
        for from_state, trans_map in fsm.map.items()
    }

    old_to_new_states = {}
    old_to_new_states[fsm.initial] = 0

    i = 0
    seen = {fsm.initial}
    old_state_queue = [fsm.initial]
    while old_state_queue:
        old_state = old_state_queue.pop(-1)
        transitions = new_map[old_state]
        sorted_transitions = sorted(transitions.items(), key=lambda v: v[0])
        for _, old_state in sorted_transitions:
            if old_state not in seen:
                old_state_queue.append(old_state)
                seen.add(old_state)
            if old_state not in old_to_new_states:
                i += 1
                old_to_new_states[old_state] = i

    new_map = dict(
        sorted(
            (
                (
                    old_to_new_states[from_state],
                    dict(
                        sorted(
                            (
                                (trans_key, old_to_new_states[to_state])
                                for trans_key, to_state in trans_map.items()
                            ),
                            key=lambda v: v[0],
                        )
                    ),
                )
                for from_state, trans_map in new_map.items()
            ),
            key=lambda v: v[0],
        )
    )

    new_initial = 0
    new_finals = frozenset(
        sorted(old_to_new_states[old_state] for old_state in fsm.finals)
    )
    new_states = frozenset(sorted(new_map.keys()))

    new_fsm = BetterFSM(new_alphabet, new_states, new_initial, new_finals, new_map)

    return new_fsm, old_to_new_states


@numba.njit(nogil=True, cache=True)
def _walk_fsm(
    fsm_transitions: Dict[Tuple[int, int], int],
    alphabet_symbol_mapping: Dict[str, int],
    alphabet_anything_value: int,
    fsm_initial: int,
    fsm_finals: Set[int],
    input_string: str,
    start_state: int,
    full_match: bool = True,
) -> List[int]:
    state = start_state
    accepted_states: List[int] = numba.typed.List.empty_list(numba.int64)
    last_final_idx: int = numba.uint64(0)

    for i, symbol in enumerate(input_string):
        trans_key = alphabet_symbol_mapping.get(symbol, alphabet_anything_value)

        new_state = fsm_transitions.get((state, trans_key))

        if new_state is None:
            if not full_match and last_final_idx > 0:
                return accepted_states[:last_final_idx]

            return numba.typed.List.empty_list(numba.int64)

        state = new_state

        if state in fsm_finals:
            last_final_idx = numba.uint64(i + 1)

        accepted_states.append(_nonoptional(state))

    if full_match and last_final_idx - 1 != i:
        return numba.typed.List.empty_list(numba.int64)

    return accepted_states


def walk_fsm(
    fsm: BetterFSM,
    input_string: str,
    start_state: int,
    full_match: bool = True,
) -> List[int]:
    fsm_finals = fsm.finals

    state = start_state
    accepted_states: List[int] = []
    last_final_idx: int = 0

    alphabet_symbol_mapping = fsm.alphabet._symbol_mapping
    alphabet_anything_value = fsm.alphabet.anything_value
    fsm_transitions = fsm.flat_transition_map

    for i, symbol in enumerate(input_string):
        trans_key = alphabet_symbol_mapping.get(symbol, alphabet_anything_value)

        new_state = fsm_transitions.get((state, trans_key))

        if new_state is None:
            if not full_match and last_final_idx > 0:
                return accepted_states[:last_final_idx]

            return []

        state = new_state

        if state in fsm_finals:
            last_final_idx = i + 1

        accepted_states.append(state)

    if full_match and last_final_idx - 1 != i:
        return []

    return accepted_states


def fsm_union(
    fsms: Sequence[FSM],
) -> Tuple[FSM, Dict[int, Tuple[Set[Tuple[int, int]], Set[int], Dict[int, Set[int]]]]]:
    """Construct an FSM representing the union of the FSMs in `fsms`.

    This is an updated version of `interegular.fsm.FSM.union` made to return an
    extra map of component FSMs to the sets of state transitions that
    correspond to them in the new FSM.

    """

    alphabet, new_to_old = Alphabet.union(*[fsm.alphabet for fsm in fsms])

    indexed_fsms = tuple(enumerate(fsms))

    initial = {i: fsm.initial for (i, fsm) in indexed_fsms}

    # Dedicated function accepting a "superset" and returning the next
    # "superset" obtained by following this transition in the new FSM
    def follow(current_state, new_transition: int):
        next = {}
        for i, f in indexed_fsms:
            old_transition = new_to_old[i][new_transition]
            if (
                i in current_state
                and current_state[i] in f.map
                and old_transition in f.map[current_state[i]]
            ):
                next[i] = f.map[current_state[i]][old_transition]
        if not next:
            raise OblivionError
        return next

    states = [initial]
    finals: Set[int] = set()
    map: Dict[int, Dict[int, int]] = {}

    # Map component FSMs to their new state-to-state transitions, finals, and a
    # map translating component FSM states to aggregate FSM states
    fsms_to_trans_finals: Dict[
        int, Tuple[Set[Tuple[int, int]], Set[int], Dict[int, Set[int]]]
    ] = {}

    i = 0
    while i < len(states):
        state = states[i]

        # Add to the finals of the aggregate FSM whenever we hit a final in a
        # component FSM
        if any(state.get(j, -1) in fsm.finals for (j, fsm) in indexed_fsms):
            finals.add(i)

        # Compute the map for this state
        map[i] = {}
        for transition in alphabet.by_transition:
            try:
                next = follow(state, transition)
            except OblivionError:
                # Reached an oblivion state; don't list it
                continue
            else:
                try:
                    # TODO: Seems like this could--and should--be avoided
                    j = states.index(next)
                except ValueError:
                    j = len(states)
                    states.append(next)

                map[i][transition] = j

                for fsm_id, fsm_state in next.items():
                    (
                        fsm_transitions,
                        fsm_finals,
                        fsm_old_to_new,
                    ) = fsms_to_trans_finals.setdefault(fsm_id, (set(), set(), {}))
                    old_from = state[fsm_id]
                    old_to = fsm_state
                    fsm_old_to_new.setdefault(old_from, set()).add(i)
                    fsm_old_to_new.setdefault(old_to, set()).add(j)
                    fsm_transitions.add((i, j))
                    if fsm_state in fsms[fsm_id].finals:
                        fsm_finals.add(j)

        i += 1

    fsm = FSM(
        alphabet=alphabet,
        states=range(len(states)),
        initial=0,
        finals=finals,
        map=map,
        __no_validation__=True,
    )

    fsm, old_to_new_states = make_deterministic_fsm(fsm)
    _fsms_to_trans_finals = {
        fsm_id: (
            {(old_to_new_states[s1], old_to_new_states[s2]) for s1, s2 in transitions},
            {old_to_new_states[s] for s in finals},
            {
                old_state: {old_to_new_states[new_state] for new_state in new_states}
                for old_state, new_states in old_to_new.items()
            },
        )
        for fsm_id, (transitions, finals, old_to_new) in sorted(
            fsms_to_trans_finals.items(), key=lambda x: x[0]
        )
    }

    return (
        fsm,
        _fsms_to_trans_finals,
    )


def get_sub_fsms_from_seq(
    state_seq: Sequence[int],
    fsms_to_trans_finals: Dict[
        int, Tuple[Set[Tuple[int, int]], Set[int], Dict[int, Set[int]]]
    ],
) -> Generator[Tuple[int, bool, bool], None, None]:
    """Get the indices of the sub-FSMs in `fsm` that could have matched the state sequence `state_seq`.

    Parameters
    ----------
    state_seq
        A state sequence.
    fsms_to_trans_finals
        A map from FSM indices to tuples containing sets of their state transitions
        and sets of the final/accept states.

    Returns
    -------
    A generator returning tuples containing each sub-FSM index (in the order
    they were union-ed to construct `fsm`) and booleans indicating whether or
    not there is another valid transition from the last state in the sequence
    for the associated sub-FSM (i.e. if the FSM can continue
    accepting/matching) and whether or not the sequence ends in a final state
    of the sub-FSM.
    """
    state_seq_transitions = set(zip(state_seq[:-1], state_seq[1:]))
    last_fsm_state = state_seq[-1]
    yield from (
        (
            # The sub-FMS index
            fsm_idx,
            # Is there another possible transition in this sub-FSM?
            any(last_fsm_state == from_s for (from_s, to_s) in transitions),
            # Is this sub-FSM in a final state?
            state_seq[-1] in finals,
        )
        for fsm_idx, (transitions, finals, _) in fsms_to_trans_finals.items()
        if state_seq_transitions.issubset(transitions)
    )


@numba.njit(cache=True, nogil=True)
def state_scan_tokens(
    fsm_transitions: Dict[Tuple[int, int], int],
    alphabet_symbol_mapping: Dict[str, int],
    alphabet_anything_value: int,
    fsm_initial: int,
    fsm_finals: Set[int],
    vocabulary: Dict[str, List[int]],
    start_state: int,
) -> Set[Tuple[int, int]]:
    res = set()

    for token, token_ids in vocabulary.items():
        state_seq = _walk_fsm(
            fsm_transitions,
            alphabet_symbol_mapping,
            alphabet_anything_value,
            fsm_initial,
            fsm_finals,
            token,
            start_state,
            False,
        )

        if state_seq is not None and len(state_seq) < len(token):
            continue

        for token_id in token_ids:
            res.add((token_id, state_seq[-1]))

    return res


def create_fsm_index_end_to_end(
    fsm_info: FSMInfo,
    vocabulary: Dict[str, List[int]],
) -> Dict[int, Set[Tuple[int, int]]]:
    """Create an FSM state-to-vocabulary map/index through end-to-end token parsing."""

    # TODO: Consider using a `List` of `Set`s instead; that way we can JIT this
    # code, too.
    states_to_token_subsets: Dict[int, Set[Tuple[int, int]]] = {}
    seen: Set[int] = set()
    next_states = {fsm_info.initial}

    while next_states:
        start_state = next_states.pop()

        token_ids_end_states = state_scan_tokens(
            fsm_info.transitions,
            fsm_info.alphabet_symbol_mapping,
            fsm_info.alphabet_anything_value,
            fsm_info.initial,
            fsm_info.finals,
            vocabulary,
            start_state,
        )

        for token_id_and_end_state in token_ids_end_states:
            states_to_token_subsets.setdefault(start_state, set()).add(
                token_id_and_end_state
            )
            end_state = token_id_and_end_state[1]
            if end_state not in seen:
                next_states.add(end_state)

        seen.add(start_state)

    return states_to_token_subsets


# TODO: Cannot cache typed collections to disk, yet.  See
# https://github.com/numba/numba/issues/4698
@lru_cache
def reduced_vocabulary(tokenizer: "Tokenizer"):
    """Create a map from decoded vocabulary tokens to lists of equivalent token ids."""
    vocabulary = numba.typed.Dict.empty(
        numba.types.string, numba.types.ListType(numba.int64)
    )
    empty_token_ids = set()
    for token, token_idx in tokenizer.vocabulary.items():
        if token in tokenizer.special_tokens:
            continue

        token_str = tokenizer.convert_token_to_string(token)

        if token_str:
            vocabulary.setdefault(
                token_str,
                numba.typed.List.empty_list(numba.int64),
            ).append(numba.int64(token_idx))
        else:
            empty_token_ids.add(numba.int64(token_idx))

    return vocabulary, empty_token_ids


def create_fsm_index_tokenizer(
    fsm: BetterFSM,
    tokenizer: "Tokenizer",
) -> Tuple[Dict[int, Dict[int, int]], Set[int]]:
    """Construct an FMS index from a tokenizer.

    This uses the end-to-end approach of `create_fsm_index_end_to_end`.

    .. warning::

        `fsm` needs to be deterministically ordered so that future caching makes sense.

    """
    vocabulary, empty_token_ids = reduced_vocabulary(tokenizer)

    states_to_token_subsets = create_fsm_index_end_to_end(fsm.fsm_info, vocabulary)

    # Allow transitions to EOS from all terminals FSM states that are
    # reachable
    # TODO: Do we really need this anymore?
    for state in fsm.fsm_info.finals:
        subset = states_to_token_subsets.get(state)
        if subset is not None:
            subset.add((tokenizer.eos_token_id, state))

    # Convert to token-to-end-state maps
    states_to_token_subsets = {k: dict(v) for k, v in states_to_token_subsets.items()}

    return states_to_token_subsets, empty_token_ids
