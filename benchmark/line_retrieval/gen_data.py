"""
Generate line data for line retrieval task.

Usage:
python3 gen_data.py --number 1000
"""
import argparse
from collections import defaultdict
import json

from tqdm import tqdm
import numpy as np


def generate_lines(random_words, num_lines, redirect_ratio):
    prefix = "Here is a list of lines, each with its corresponding REGISTER_CONTENT value. Please memorize them. Be prepared to provide the REGISTER_CONTENT value for a specific line index when I ask."
    suffix = "The list has ended. Please give the final REGISTER_CONTENT value for a specific line after resovling the redirections and references. For example, the REGISTER_CONTENT of Line __idx0__ is __val0__. The REGISTER_CONTENT of Line __idx1__ is __val1__. The REGISTER_CONTENT of Line __idx2__ is __val2__. The REGISTER_CONTENT of Line ??? is"

    # Raw lines
    visited_indices = set([None])
    visited_values = set([None])

    lines = []
    redirects = []
    indices = []
    values = []
    for i in tqdm(range(num_lines)):
        line_index = None
        while line_index in visited_indices:
            line_index = "-".join(np.random.choice(random_words, size=(2,)))
        visited_indices.add(line_index)

        line_value = np.random.randint(low=0, high=999999)
        line_value = f"{line_value:06}"

        line = f"Line {line_index}: The REGISTER_CONTENT is {line_value}."
        lines.append(line)
        redirects.append(None)
        indices.append(line_index)
        values.append(line_value)

    # Add redirect
    if redirect_ratio > 0:
        num_redirect_lines = int(len(lines) * redirect_ratio)
        redirect_indices = np.random.choice(np.arange(len(lines)),
            size=(num_redirect_lines,), replace=False)
        for i in redirect_indices:
            target_idx = np.random.choice(min(i * 2 + 100, num_lines))
            lines[i] = f"Line {indices[i]}: The REGISTER_CONTENT is the same as Line {indices[target_idx]}."
            redirects[i] = target_idx

    # Build links and find sources
    links = [[] for _ in range(num_lines)]
    contains_ring = set()
    for i in range(num_lines):
        if redirects[i] is None:
            continue

        tmp_link = []
        cur = i
        visited = set()
        while redirects[cur] is not None:
            visited.add(cur)
            tmp_link.append(redirects[cur])
            cur = redirects[cur]

            if cur in visited:
                contains_ring.add(i)
                tmp_link = None
                break
        values[i] = values[cur]
        links[i] = tmp_link

    # Group by num_links
    group_by_num_hoops = defaultdict(list)
    for i in range(num_lines):
        if i in contains_ring:
            continue
        group_by_num_hoops[len(links[i]) + 1].append(i)

    keys = sorted(list(group_by_num_hoops.keys()))
    for num_links in keys:
        print(f"#links: {num_links}, #lines: {len(group_by_num_hoops[num_links])}")

    # Append few-shot examples
    hoop1_candidates = list(group_by_num_hoops[1])
    hoop1_candidate_keys = {c: max([c] + links[c]) for c in hoop1_candidates}
    hoop1_candidates.sort(key=lambda c: hoop1_candidate_keys[c])
    hoop2_candidates = list(group_by_num_hoops[2])
    hoop2_candidate_keys = {c: max([c] + links[c]) for c in hoop2_candidates}
    hoop2_candidates.sort(key=lambda c: hoop2_candidate_keys[c])

    i = hoop1_candidates[5]
    suffix = suffix.replace("__idx0__", indices[i]).replace("__val0__", values[i])
    if len(hoop2_candidates):
        i = hoop2_candidates[0]
        suffix = suffix.replace("__idx1__", indices[i]).replace("__val1__", values[i])
        i = hoop2_candidates[1]
        suffix = suffix.replace("__idx2__", indices[i]).replace("__val2__", values[i])
    else:
        i = hoop1_candidates[1]
        suffix = suffix.replace("__idx1__", indices[i]).replace("__val1__", values[i])
        i = hoop1_candidates[10]
        suffix = suffix.replace("__idx2__", indices[i]).replace("__val2__", values[i])

    obj = {
        "prefix": prefix,
        "suffix": suffix,
        "lines": lines,
        "indices": indices,
        "values": values,
        "links": links,
        "group_by_num_hoops": group_by_num_hoops,
        "contains_ring": sorted(list(contains_ring)),
    }
    return obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", type=int)
    parser.add_argument("--redirect-ratio", type=float, default=0.0)
    args = parser.parse_args()

    num_lines = args.number

    random_words_filename = "random_words.json"
    random_words = json.load(open(random_words_filename, "r"))

    np.random.seed(42)
    obj = generate_lines(random_words, num_lines, args.redirect_ratio)

    fout = f"lines_{num_lines}_{args.redirect_ratio:.1f}.json"
    with open(fout, "w") as fout:
        json.dump(obj, fout, indent=2)
