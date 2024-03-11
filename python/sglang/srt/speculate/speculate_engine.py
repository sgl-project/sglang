from typing import List, Tuple
import numpy as np


class TrieNode:
    def __init__(self, token_id, parent, weight=0.0):
        self.token_id = token_id
        self.parent = parent
        self.weight = weight

        self.depth = 0 if parent is None else parent.depth + 1
        self.children = []

        self.spec_prob = 0
        self.probs_sum = 0
        self.idx = 0

    def __lt__(self, other):
        return (self.weight, -self.depth) < (other.weight, -other.depth)

    def __eq__(self, other):
        return self.depth == other.depth and abs(self.weight - other.weight) < 1e-6

    def sample_children(self, verified_indices: List[int], verified_ids: List[int]):
        spec_probs = np.array([child.spec_prob for child in self.children])
        spec_probs = spec_probs.astype(np.float64)
        other_prob = max(0, self.probs_sum - np.sum(spec_probs))
        spec_probs = np.append(spec_probs, other_prob)
        spec_probs /= np.sum(spec_probs)
        sample_index = np.random.multinomial(1, spec_probs).argmax()

        if sample_index < len(self.children):
            verified_indices.append(self.children[sample_index].idx)
            verified_ids.append(self.children[sample_index].token_id)
            return self.children[sample_index].sample_children(
                verified_indices, verified_ids
            )


class SpeculateTries:
    def __init__(self, max_spec_num, first_token_fixed=True):
        self.max_spec_num = max_spec_num
        self.first_token_fixed = first_token_fixed

        self.root = TrieNode(-2, None)
        self.node_list = []

        # flattened info
        self.tree_ids = None
        self.tree_nodes = None
        self.parent_indices = None
        self.tree_depths = None
        self.tree_mask = None

    def spec_len(self):
        return len(self.node_list)

    def is_empty(self):
        return self.spec_len() == 0

    def insert(self, tree_ids: List[int], weight: float):
        cur_node = self.root

        for token_id in tree_ids:
            child_node = None
            for child in cur_node.children:
                if child.token_id == token_id:
                    child_node = child
                    child_node.weight += weight
                    break

            if child_node is None:
                child_node = TrieNode(token_id, cur_node, weight)
                cur_node.children.append(child_node)
                self.node_list.append(child_node)

            if len(self.node_list) > self.max_spec_num:
                min_node = min(self.node_list)
                assert min_node.children == []
                min_node.parent.children.remove(min_node)
                self.node_list.remove(min_node)
                if min_node == child_node:
                    break

            cur_node = child_node

    def _print_helper(self, node, indent, tokenizer=None):
        for child in node.children:
            print(
                "-" * indent
                + (
                    str(child.token_id)
                    if tokenizer is None
                    else tokenizer.convert_ids_to_tokens([child.token_id])[0]
                )
            )
            self._print_helper(child, indent + 2, tokenizer)

    def pretty_print(self, tokenizer=None):
        self._print_helper(self.root, 0, tokenizer)

    def _flatten_helper(
        self, node, node_idx, tree_nodes, parent_indices, tree_depths, tree_mask
    ):
        for child in node.children:
            tree_nodes.append(child)
            parent_indices.append(node_idx)
            tree_depths.append(child.depth)
            child_idx = len(tree_nodes) - 1
            tree_mask[child_idx, child_idx] = 1
            if node_idx >= 0:
                tree_mask[child_idx] |= tree_mask[node_idx]
            self._flatten_helper(
                child, child_idx, tree_nodes, parent_indices, tree_depths, tree_mask
            )

    def flatten(self):
        num_tokens = len(self.node_list)
        self.tree_nodes, self.parent_indices, self.tree_depths, self.tree_mask = (
            [],
            [],
            [],
            np.zeros((num_tokens, num_tokens), dtype=np.int32),
        )

        self._flatten_helper(
            self.root,
            -1,
            self.tree_nodes,
            self.parent_indices,
            self.tree_depths,
            self.tree_mask,
        )
        self.tree_ids = [node.token_id for node in self.tree_nodes]

    def fill_probs(self, probs_and_sum: List[Tuple]):
        for i, (spec_prob, probs_sum) in enumerate(probs_and_sum):
            self.tree_nodes[i].spec_prob = spec_prob
            self.tree_nodes[i].probs_sum = probs_sum
            self.tree_nodes[i].idx = i

    def sample_tries(self):
        if not self.first_token_fixed:
            raise NotImplementedError("First token not fixed")

        assert len(self.root.children) == 1
        first_node = self.root.children[0]
        verified_indices, verified_ids = [first_node.idx], [first_node.token_id]
        first_node.sample_children(verified_indices, verified_ids)
        return verified_indices, verified_ids


class SpeculateEngine:
    def __init__(
        self,
        tokenizer,
        min_match_len: int = 3,
        max_spec_len: int = 10,
        max_spec_num: int = 50,
    ):
        self.tokenizer = tokenizer
        self.min_match_len = min_match_len
        self.max_spec_len = max_spec_len
        self.max_spec_num = max_spec_num

        self.entries = []
        self.prev_ids = []

    def add_entry_tokens(self, tokens: List[int]):
        self.entries.append(tokens)

    def add_entry_text(self, ref_text: str):
        lines = ref_text.split("\n")
        for line in lines:
            self.add_entry_tokens(self.tokenizer.encode(line))

    def add_entries_from_file(self, ref_file_path: str):
        lines = open(ref_file_path, "r").readlines()
        for line in lines:
            self.add_entry_tokens(self.tokenizer.encode(line))

    def _match(self, x: List[int], y: List[int]):
        for p in range(min(len(x), len(y))):
            if x[-p - 1] != y[-p - 1]:
                return p
        return min(len(x), len(y))

    def set_prev_ids(self, prev_ids: List[int]):
        self.prev_ids = prev_ids

    def search(
        self,
        tokens_to_match: List[int],
        include_last_matched=True,
        min_match_len=None,
        max_spec_len=None,
        max_spec_num=None,
    ):
        # resolve default values
        min_match_len = self.min_match_len if min_match_len is None else min_match_len
        max_spec_len = self.max_spec_len if max_spec_len is None else max_spec_len
        max_spec_num = self.max_spec_num if max_spec_num is None else max_spec_num

        # Bruteforce search currently
        spec_tries = SpeculateTries(max_spec_num, include_last_matched)

        if len(tokens_to_match) < min_match_len:
            return spec_tries

        for entry in self.entries + [self.prev_ids]:
            for i in range(min_match_len, len(entry)):
                matched_len = self._match(tokens_to_match, entry[:i])
                if matched_len >= min_match_len:
                    if include_last_matched:
                        spec_tries.insert(
                            entry[i - 1 : i + max_spec_len], matched_len**2.0
                        )
                    else:
                        spec_tries.insert(entry[i : i + max_spec_len], matched_len**2.0)

        spec_tries.flatten()
        return spec_tries


def test():
    from sglang.srt.hf_transformers_utils import get_tokenizer

    tokenizer = get_tokenizer("meta-llama/Llama-2-7b-chat-hf")
    speculate_engine = SpeculateEngine(tokenizer)

    ref_text = """\
    The location of Hogwarts is in Scotland, UK.
    The headmaster of Hogwarts is Albus Dumbledore.
    The potions teacher in Hogwarts is Severus Snape.
    The transfiguration teacher in Hogwarts is Minerva McGonagall.
    The herbology teacher in Hogwarts is Pomona Sprout.
    The defense against the dark arts teacher in Hogwarts is Gilderoy Lockhart."""

    speculate_engine.add_entry_text(ref_text)

    prompt1 = "The latitude of Hogwarts is"
    prompt2 = "The transfiguration teacher in Hogwarts is"

    spec_tries1 = speculate_engine.search(tokenizer.encode(prompt1))
    spec_tries2 = speculate_engine.search(tokenizer.encode(prompt2))

    spec_tries1.pretty_print(tokenizer)
    print(tokenizer.convert_ids_to_tokens(spec_tries1.tree_ids))
    print(spec_tries1.parent_indices)
    print(spec_tries1.tree_mask)
    print("=" * 100)

    spec_tries2.pretty_print(tokenizer)
    print(tokenizer.convert_ids_to_tokens(spec_tries2.tree_ids))
    print(spec_tries2.parent_indices)
    print(spec_tries2.tree_mask)
    print("=" * 100)


if __name__ == "__main__":
    test()
