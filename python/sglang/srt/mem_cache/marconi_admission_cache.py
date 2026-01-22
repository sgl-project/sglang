from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def _key_match(key0: Tuple[int, ...], key1: Tuple[int, ...]) -> int:
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i


class AdmissionNode:
    def __init__(self):
        self.children: Dict[object, "AdmissionNode"] = {}
        self.parent: Optional["AdmissionNode"] = None
        self.key: Tuple[int, ...] = tuple()
        self.value: List[int] = []
        self.prefix_len = 0


class MarconiAdmissionTree:
    def __init__(self):
        self.root_node = AdmissionNode()

    def match_prefix(
        self, token_ids: List[int], extra_key: Optional[str]
    ) -> Tuple[int, bool]:
        node = self.root_node
        key = tuple(token_ids)
        matched_len = 0
        reusable_len = 0
        while key:
            child_key = self._child_key(extra_key, key)
            child = node.children.get(child_key)
            if child is None:
                break
            prefix_len = _key_match(child.key, key)
            matched_len += prefix_len
            if prefix_len < len(child.key):
                break
            reusable_len += prefix_len
            key = key[prefix_len:]
            node = child
        branchoff_required = matched_len > reusable_len
        return matched_len, branchoff_required

    def insert(self, token_ids: List[int], extra_key: Optional[str]) -> None:
        node = self.root_node
        key = tuple(token_ids)
        value = list(token_ids)
        while key:
            child_key = self._child_key(extra_key, key)
            child = node.children.get(child_key)
            if child is None:
                new_node = AdmissionNode()
                new_node.parent = node
                new_node.key = key
                new_node.value = value
                new_node.prefix_len = node.prefix_len + len(value)
                node.children[child_key] = new_node
                return
            prefix_len = _key_match(child.key, key)
            if prefix_len == len(child.key):
                if prefix_len == len(key):
                    return
                node = child
                key = key[prefix_len:]
                value = value[prefix_len:]
                continue
            node = self._split_node(child, prefix_len, extra_key)
            key = key[prefix_len:]
            value = value[prefix_len:]

    def _split_node(
        self, child: AdmissionNode, split_len: int, extra_key: Optional[str]
    ) -> AdmissionNode:
        new_node = AdmissionNode()
        new_node.parent = child.parent
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        new_node.prefix_len = new_node.parent.prefix_len + len(new_node.value)
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        child.prefix_len = new_node.prefix_len + len(child.value)
        new_node.children[self._child_key(extra_key, child.key)] = child
        new_node.parent.children[self._child_key(extra_key, new_node.key)] = new_node
        return new_node

    @staticmethod
    def _child_key(extra_key: Optional[str], key: Tuple[int, ...]):
        token = key[0]
        return token if extra_key is None else (extra_key, token)
