# -*- coding: utf-8 -*-

import logging
import os
from typing import List, Tuple

import numpy as np
from torch.utils.cpp_extension import load

logger = logging.getLogger(__name__)

_abs_path = os.path.dirname(os.path.abspath(__file__))
ngram_cache_cpp = load(
    name="ngram_cache_cpp",
    sources=[
        f"{_abs_path}/ngram_cache_binding.cpp",
        f"{_abs_path}/ngram.cpp",
    ],
    extra_cflags=["-O3", "-std=c++20"],
)


class NgramCache:
    def __init__(
        self,
        branch_length=18,
        min_match_window_size=1,
        max_match_window_size=10,
        min_bfs_breadth=1,
        max_bfs_breadth=8,
        draft_token_num=8,
        match_type="BFS",
        capacity=1000000,
    ):
        param = ngram_cache_cpp.Param()
        param.branch_length = branch_length
        param.min_match_window_size = min_match_window_size
        param.max_match_window_size = max_match_window_size
        param.min_bfs_breadth = min_bfs_breadth
        param.max_bfs_breadth = max_bfs_breadth
        param.draft_token_num = draft_token_num
        param.match_type = match_type
        self.cache = ngram_cache_cpp.Ngram(capacity, param)

        self.default_mask = np.ones((1, 1), dtype=np.int64)
        self.draft_token_num = draft_token_num

    def batch_put(self, batch_tokens: List[List[int]]):
        self.cache.asyncInsert(batch_tokens)

    def synchronize(self):
        self.cache.synchronize()

    def reset(self):
        self.cache.reset()

    def batch_get(self, batch_tokens: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        result = self.cache.batchMatch(batch_tokens)
        return np.array(result.token), np.array(result.mask)

    def leaf_paths_from_mask(
        self, tokens: List[int], tree_mask: List[List[int]]
    ) -> List[List[int]]:
        """
        Find all leaf paths according to the binary tree_mask (i.e., paths that are not prefixes of any other path).

        Args:
            mask   : List[List[int]]   # nxn binary matrix
            tokens : List[int]         # token list corresponding to columns

        Returns:
            List[List[int]]            # token lists of only the leaf paths, preserving their order of appearance
        """

        row_sets = [
            (i, {idx for idx, v in enumerate(row) if v == 1})
            for i, row in enumerate(tree_mask)
        ]
        leaf_sets = []
        leaf_rows = []

        for i, cur_set in reversed(row_sets):
            if any(cur_set <= kept for kept in leaf_sets):
                continue
            leaf_sets.append(cur_set)
            leaf_rows.append(i)

        leaf_rows.reverse()
        result = []
        for r in leaf_rows:
            path = [tokens[col] for col in range(len(tokens)) if tree_mask[r][col] == 1]
            result.append(path)

        return result

    def debug_result(
        self, decoding_ids: np.ndarray, decoding_masks: np.ndarray, tokenizer=None
    ):
        decoding_ids = decoding_ids.reshape(-1, self.draft_token_num)
        decoding_masks = decoding_masks.reshape(
            -1, self.draft_token_num, self.draft_token_num
        )
        logger.info(f"\n{decoding_ids=}\n{decoding_masks=}")
        for i in range(decoding_ids.shape[0]):
            leaf_paths = self.leaf_paths_from_mask(
                decoding_ids[i].tolist(), decoding_masks[i].tolist()
            )
            if tokenizer is None:
                logger.info(f"draft path {i}: {leaf_paths}")
            else:
                logger.info(f"result {i}:")
                for leaf_path in leaf_paths:
                    logger.info(
                        f"draft path {i}: {leaf_path} -> {tokenizer.decode(leaf_path, ensure_ascii=False)}"
                    )


# main function
if __name__ == "__main__":
    format = f"%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(
        level=logging.DEBUG,
        format=format,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    token_ids = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 44, 55, 66, 77, 88, 99, 100],
    ]
    cache = NgramCache(branch_length=12, draft_token_num=8)
    cache.batch_put(token_ids)

    cache.synchronize()
    decoding_ids, decoding_masks = cache.batch_get([[1, 2, 3], [3, 44], [3, 6, 999]])

    cache.debug_result(decoding_ids, decoding_masks)
