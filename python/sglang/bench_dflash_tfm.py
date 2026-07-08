import argparse
import csv
import hashlib
import heapq
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

MTBENCH_COMMIT = "587d5cfa1609a43d192cedb8441cac3c17db105d"
MTBENCH_URL = (
    f"https://raw.githubusercontent.com/lm-sys/FastChat/{MTBENCH_COMMIT}/"
    "fastchat/llm_judge/data/mt_bench/question.jsonl"
)
LCB_DATASET = "livecodebench/code_generation_lite"
LCB_FILENAME = "test6.jsonl"

SHARECHAT_REPO = "tucnguyen/ShareChat"
SHARECHAT_SEED = 20260617
SHARECHAT_ENGLISH_TOTAL = 250
SHARECHAT_NON_ENGLISH_PER_LANGUAGE = 50
SHARECHAT_MAX_POOL_PER_CELL = 300
SHARECHAT_MIN_TOTAL_USER_CHARS = 20
SHARECHAT_MAX_TOTAL_USER_CHARS = 12000
SHARECHAT_MAX_SINGLE_USER_CHARS = 8000
SHARECHAT_MAX_USER_TURNS = 8
SHARECHAT_FILENAME = (
    f"sharechat_english{SHARECHAT_ENGLISH_TOTAL}"
    f"_nonenglish5lang{SHARECHAT_NON_ENGLISH_PER_LANGUAGE * 5}"
    f"_seed{SHARECHAT_SEED}.jsonl"
)
SHARECHAT_SOURCE_FILES = {
    "chatgpt": "chatgpt_results_final_language_filtered.csv",
    "claude": "claude_results_final_language_filtered.csv",
    "gemini": "gemini_results_final_language_filtered.csv",
    "grok": "grok_results_final_language_filtered.csv",
    "perplexity": "perplexity_results_final_language_filtered.csv",
}
SHARECHAT_TOPICS = [
    "specific_info",
    "other",
    "computer_programming",
    "mathematical_calculation",
    "tutoring_or_teaching",
    "write_fiction",
    "argument_or_summary_generation",
    "unclear",
    "create_an_image",
    "relationships_and_personal_reflection",
    "edit_or_critique_provided_text",
    "greetings_and_chitchat",
    "health_fitness_beauty_or_self_care",
    "data_analysis",
    "how_to_advice",
    "analyze_an_image",
    "purchasable_products",
    "translation",
    "personal_writing_or_communication",
    "cooking_and_recipes",
    "asking_about_the_model",
    "creative_ideation",
    "games_and_role_play",
    "generate_or_retrieve_other_media",
]
SHARECHAT_TARGET_LANGUAGES = ["Japanese", "Spanish", "Russian", "Chinese", "French"]
SHARECHAT_ENGLISH_PLATFORM_TARGET = {
    "chatgpt": 250,
    "grok": 80,
    "perplexity": 80,
    "gemini": 50,
    "claude": 40,
}
SHARECHAT_NON_ENGLISH_PLATFORM_TARGET = {
    "chatgpt": 300,
    "grok": 80,
    "perplexity": 70,
    "gemini": 30,
    "claude": 20,
}
SHARECHAT_DEFAULT_PATH = Path.home() / "datasets" / "sharechat" / SHARECHAT_FILENAME

SUBSAMPLE_SEED = 20260617
DATASET_LIMITS = {"gsm8k": 128, "math500": 128}
STRATIFIED_DATASETS = {"math500", "livecodebench"}
CACHE_DIR = Path(
    os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
) / "sglang_bench_dflash_tfm"

DATASET_ALIASES = {"lcb": "livecodebench"}
DATASET_NAMES = [
    "mtbench",
    "sharechat",
    "gsm8k",
    "math500",
    "aime25",
    "humaneval",
    "mbpp",
    "livecodebench",
]


def _http_json(url, payload=None, timeout=60.0, method=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method=method or ("POST" if payload is not None else "GET"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode()
    return json.loads(body) if body else {}


def _wait_ready(base_url, timeout):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(base_url + "/health", method="GET")
            with urllib.request.urlopen(req, timeout=10.0):
                return
        except Exception as e:
            last_error = e
            time.sleep(2.0)
    raise TimeoutError(f"server {base_url} did not become ready: {last_error}")


def _flush_cache(base_url, timeout=60.0):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(base_url + "/flush_cache", data=b"", method="POST")
            with urllib.request.urlopen(req, timeout=timeout):
                return
        except urllib.error.HTTPError as e:
            last_error = e
            time.sleep(1.0)
    raise RuntimeError(f"flush_cache did not succeed before timeout: {last_error}")


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as f:
        f.write(resp.read())
    tmp.replace(path)


def _load_mtbench():
    path = CACHE_DIR / "mt_bench_questions.jsonl"
    if not path.exists():
        _download(MTBENCH_URL, path)
    with path.open() as f:
        rows = [json.loads(line) for line in f]
    questions = [
        (int(r["question_id"]), str(r["turns"][0]), str(r["category"])) for r in rows
    ]
    return questions, f"lm-sys/FastChat@{MTBENCH_COMMIT[:12]} mt_bench/question.jsonl"


def _sharechat_topic_quota(total):
    base, remainder = divmod(total, len(SHARECHAT_TOPICS))
    return {
        topic: base + (1 if idx < remainder else 0)
        for idx, topic in enumerate(SHARECHAT_TOPICS)
    }


def _sharechat_stable_key(seed, *parts):
    raw = "|".join(str(part) for part in (seed, *parts)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") / 2**64


def _sharechat_collect_candidates(source_paths):
    csv.field_size_limit(sys.maxsize)
    topic_set = set(SHARECHAT_TOPICS)
    english_pools = {topic: [] for topic in SHARECHAT_TOPICS}
    non_english_pools = {
        (language, topic): []
        for language in SHARECHAT_TARGET_LANGUAGES
        for topic in SHARECHAT_TOPICS
    }

    def add_to_pool(pool, row):
        item = (-float(row["sample_key"]), row)
        if len(pool) < SHARECHAT_MAX_POOL_PER_CELL:
            heapq.heappush(pool, item)
        elif item[0] > pool[0][0]:
            heapq.heapreplace(pool, item)

    def emit(*, platform, url, user_turns, topic_counts, language_counts, source_turns_count):
        if not user_turns:
            return
        total_chars = sum(len(turn) for turn in user_turns)
        if total_chars < SHARECHAT_MIN_TOTAL_USER_CHARS:
            return
        if total_chars > SHARECHAT_MAX_TOTAL_USER_CHARS or any(
            len(turn) > SHARECHAT_MAX_SINGLE_USER_CHARS for turn in user_turns
        ):
            return
        if len(user_turns) > SHARECHAT_MAX_USER_TURNS:
            return
        official_topics = Counter(
            {topic: count for topic, count in topic_counts.items() if topic in topic_set}
        )
        if not official_topics:
            return
        topic = official_topics.most_common(1)[0][0]
        language = language_counts.most_common(1)[0][0] if language_counts else "unknown"
        row = {
            "source": "sharechat",
            "platform": platform,
            "url": url,
            "topic": topic,
            "detected_language_final": language,
            "turns": list(user_turns),
            "turns_count": len(user_turns),
            "source_turns_count": source_turns_count,
            "total_user_chars": total_chars,
            "sample_key": _sharechat_stable_key(SHARECHAT_SEED, platform, url),
        }
        if language == "English":
            add_to_pool(english_pools[topic], row)
        elif language in SHARECHAT_TARGET_LANGUAGES:
            add_to_pool(non_english_pools[(language, topic)], row)

    def norm(value):
        return str(value or "").strip()

    for platform, path in source_paths.items():
        current_url = None
        user_turns = []
        topic_counts = Counter()
        language_counts = Counter()
        source_turns_count = ""
        with path.open(newline="", encoding="utf-8", errors="replace") as file:
            reader = csv.DictReader(file)
            for row in reader:
                url = norm(row.get("url"))
                if current_url is None:
                    current_url = url
                elif url != current_url:
                    emit(
                        platform=platform,
                        url=current_url,
                        user_turns=user_turns,
                        topic_counts=topic_counts,
                        language_counts=language_counts,
                        source_turns_count=source_turns_count,
                    )
                    current_url = url
                    user_turns = []
                    topic_counts = Counter()
                    language_counts = Counter()
                    source_turns_count = ""
                if not source_turns_count:
                    source_turns_count = norm(row.get("turns_count"))
                if norm(row.get("role")).lower() in {"user", "human"}:
                    text = norm(row.get("plain_text"))
                    if text:
                        user_turns.append(text)
                        topic = norm(row.get("topic"))
                        if topic in topic_set:
                            topic_counts[topic] += 1
                        language_counts[norm(row.get("detected_language_final")) or "unknown"] += 1
            if current_url is not None:
                emit(
                    platform=platform,
                    url=current_url,
                    user_turns=user_turns,
                    topic_counts=topic_counts,
                    language_counts=language_counts,
                    source_turns_count=source_turns_count,
                )

    english = {
        topic: [item[1] for item in sorted(pool, reverse=True)]
        for topic, pool in english_pools.items()
    }
    non_english = {
        key: [item[1] for item in sorted(pool, reverse=True)]
        for key, pool in non_english_pools.items()
    }
    return english, non_english


def _sharechat_choose_balanced(candidates, *, used_urls, by_platform, by_topic, platform_target):
    available = [candidate for candidate in candidates if candidate["url"] not in used_urls]
    if not available:
        return None

    def score(candidate):
        platform = str(candidate["platform"])
        target = platform_target.get(platform, 1)
        return (
            by_platform[platform] >= target,
            by_platform[platform] / target,
            by_topic[str(candidate["topic"])],
            float(candidate["sample_key"]),
        )

    return min(available, key=score)


def _sharechat_select_english(pools, *, total):
    quotas = _sharechat_topic_quota(total)
    selected = []
    used_urls = set()
    by_platform = Counter()
    by_topic = Counter()
    for topic in SHARECHAT_TOPICS:
        for _ in range(quotas[topic]):
            candidate = _sharechat_choose_balanced(
                pools[topic],
                used_urls=used_urls,
                by_platform=by_platform,
                by_topic=by_topic,
                platform_target=SHARECHAT_ENGLISH_PLATFORM_TARGET,
            )
            if candidate is None:
                break
            selected.append(candidate)
            used_urls.add(str(candidate["url"]))
            by_platform[str(candidate["platform"])] += 1
            by_topic[str(candidate["topic"])] += 1

    if len(selected) < total:
        remaining = [
            candidate
            for topic in SHARECHAT_TOPICS
            for candidate in pools[topic]
            if candidate["url"] not in used_urls
        ]
        remaining.sort(
            key=lambda candidate: (
                by_platform[str(candidate["platform"])]
                / SHARECHAT_ENGLISH_PLATFORM_TARGET.get(str(candidate["platform"]), 1),
                by_topic[str(candidate["topic"])],
                float(candidate["sample_key"]),
            )
        )
        for candidate in remaining:
            if len(selected) >= total:
                break
            selected.append(candidate)
            used_urls.add(str(candidate["url"]))
            by_platform[str(candidate["platform"])] += 1
            by_topic[str(candidate["topic"])] += 1
    return selected[:total]


def _sharechat_select_non_english(pools, *, per_language):
    total = per_language * len(SHARECHAT_TARGET_LANGUAGES)
    global_topic_quota = _sharechat_topic_quota(total)
    language_topic_quota = {
        language: _sharechat_topic_quota(per_language)
        for language in SHARECHAT_TARGET_LANGUAGES
    }
    selected = []
    used_urls = set()
    by_platform = Counter()
    by_topic = Counter()
    by_language = Counter()

    def choose(language, topic):
        candidates = [
            candidate for candidate in pools[(language, topic)] if candidate["url"] not in used_urls
        ]
        if not candidates:
            return None

        def score(candidate):
            platform = str(candidate["platform"])
            target = SHARECHAT_NON_ENGLISH_PLATFORM_TARGET.get(platform, 1)
            return (
                by_topic[str(candidate["topic"])] >= global_topic_quota[str(candidate["topic"])],
                by_platform[platform] >= target,
                by_platform[platform] / target,
                by_topic[str(candidate["topic"])],
                float(candidate["sample_key"]),
            )

        return min(candidates, key=score)

    for language in SHARECHAT_TARGET_LANGUAGES:
        for topic in SHARECHAT_TOPICS:
            for _ in range(language_topic_quota[language][topic]):
                candidate = choose(language, topic)
                if candidate is None:
                    break
                selected.append(candidate)
                used_urls.add(str(candidate["url"]))
                by_platform[str(candidate["platform"])] += 1
                by_topic[str(candidate["topic"])] += 1
                by_language[language] += 1

    for language in SHARECHAT_TARGET_LANGUAGES:
        if by_language[language] >= per_language:
            continue
        candidates = [
            candidate
            for topic in SHARECHAT_TOPICS
            for candidate in pools[(language, topic)]
            if candidate["url"] not in used_urls
        ]
        candidates.sort(
            key=lambda candidate: (
                by_topic[str(candidate["topic"])] >= global_topic_quota[str(candidate["topic"])],
                by_topic[str(candidate["topic"])],
                by_platform[str(candidate["platform"])]
                / SHARECHAT_NON_ENGLISH_PLATFORM_TARGET.get(str(candidate["platform"]), 1),
                float(candidate["sample_key"]),
            )
        )
        for candidate in candidates:
            if by_language[language] >= per_language:
                break
            selected.append(candidate)
            used_urls.add(str(candidate["url"]))
            by_platform[str(candidate["platform"])] += 1
            by_topic[str(candidate["topic"])] += 1
            by_language[language] += 1

    return selected[:total]


def _build_sharechat_sample(path):
    source_paths = {}
    for platform, filename in SHARECHAT_SOURCE_FILES.items():
        print(f"sharechat: downloading {SHARECHAT_REPO}/{filename}", flush=True)
        source_paths[platform] = Path(
            hf_hub_download(SHARECHAT_REPO, filename, repo_type="dataset")
        )
    print("sharechat: selecting the stratified sample", flush=True)
    english_pools, non_english_pools = _sharechat_collect_candidates(source_paths)
    english = _sharechat_select_english(english_pools, total=SHARECHAT_ENGLISH_TOTAL)
    non_english = _sharechat_select_non_english(
        non_english_pools, per_language=SHARECHAT_NON_ENGLISH_PER_LANGUAGE
    )

    rng = random.Random(SHARECHAT_SEED)
    rng.shuffle(english)
    rng.shuffle(non_english)
    combined = [{"combined_split": "english", **row} for row in english] + [
        {"combined_split": "non_english_5lang", **row} for row in non_english
    ]
    rng.shuffle(combined)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as file:
        for idx, row in enumerate(combined):
            language = str(row["detected_language_final"])
            file.write(
                json.dumps(
                    {
                        "id": idx,
                        "source": "sharechat",
                        "split": str(row.get("combined_split") or "combined"),
                        "category": f"sharechat/{row['topic']}/{language}",
                        "platform": row["platform"],
                        "url": row["url"],
                        "topic": row["topic"],
                        "detected_language_final": language,
                        "language_group": "English" if language == "English" else "non-English",
                        "turns": row["turns"],
                        "turns_count": row["turns_count"],
                        "source_turns_count": row["source_turns_count"],
                        "total_user_chars": row["total_user_chars"],
                        "sample_seed": SHARECHAT_SEED,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    tmp.replace(path)


def _load_sharechat(path):
    if not path.exists():
        built = CACHE_DIR / SHARECHAT_FILENAME
        if not built.exists():
            print(
                f"sharechat: {path} not found; rebuilding the pinned sample from "
                f"hf://{SHARECHAT_REPO} (seed {SHARECHAT_SEED})",
                flush=True,
            )
            _build_sharechat_sample(built)
        path = built
    questions = []
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            turns = [str(t).strip() for t in row.get("turns", ()) if str(t).strip()]
            if turns:
                topic = str(row.get("topic") or "unknown").strip() or "unknown"
                questions.append(
                    (int(row.get("id", idx)), turns[0], f"sharechat_topic/{topic}")
                )
    return questions, f"{path.name} sha256:{_sha256_file(path)[:12]}"


def _load_hf(repo, subset, split, to_prompt, to_category):
    ds = load_dataset(repo, subset, split=split) if subset else load_dataset(repo, split=split)
    questions = [(idx, to_prompt(row), to_category(row)) for idx, row in enumerate(ds)]
    fingerprint = getattr(ds, "_fingerprint", "") or ""
    subset_tag = f"/{subset}" if subset else ""
    return questions, f"hf://{repo}{subset_tag} split={split} fingerprint={fingerprint}"


def _mbpp_prompt(row):
    prompt = str(row["prompt"]).strip()
    tests = row.get("test_list")
    if isinstance(tests, list) and tests:
        prompt += "\n\nExamples/tests:\n" + "\n".join(str(t) for t in tests)
    return prompt


def _load_livecodebench():
    path = Path(hf_hub_download(LCB_DATASET, LCB_FILENAME, repo_type="dataset"))
    questions = []
    with path.open() as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = str(row["question_content"]).strip()
            starter = str(row.get("starter_code") or "").strip()
            if starter:
                prompt += f"\n\nStarter code:\n{starter}"
            platform = str(row.get("platform") or "unknown").strip() or "unknown"
            difficulty = str(row.get("difficulty") or "unknown").strip() or "unknown"
            questions.append((idx, prompt, f"{platform}/{difficulty}"))
    return questions, f"hf://{LCB_DATASET}/{LCB_FILENAME} sha256:{_sha256_file(path)[:12]}"


def _load_dataset(name, sharechat_path):
    loaders = {
        "mtbench": _load_mtbench,
        "sharechat": lambda: _load_sharechat(sharechat_path),
        "gsm8k": lambda: _load_hf(
            "openai/gsm8k", "main", "test", lambda r: str(r["question"]), lambda r: "math"
        ),
        "math500": lambda: _load_hf(
            "HuggingFaceH4/MATH-500", None, "test",
            lambda r: str(r["problem"]), lambda r: str(r["subject"]),
        ),
        "aime25": lambda: _load_hf(
            "math-ai/aime25", None, "test", lambda r: str(r["problem"]), lambda r: "math"
        ),
        "humaneval": lambda: _load_hf(
            "openai/openai_humaneval", None, "test", lambda r: str(r["prompt"]), lambda r: "code"
        ),
        "mbpp": lambda: _load_hf(
            "google-research-datasets/mbpp", "sanitized", "test", _mbpp_prompt, lambda r: "code"
        ),
        "livecodebench": _load_livecodebench,
    }
    return loaders[name]()


def _stable_key(seed, name, qid, category):
    raw = f"{seed}|{name}|{qid}|{category}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _allocate_stratified_counts(group_sizes, limit):
    total = sum(group_sizes.values())
    if limit >= total:
        return dict(group_sizes)
    quotas = {
        group: min(size, int(limit * size / total))
        for group, size in group_sizes.items()
    }
    remaining = limit - sum(quotas.values())
    remainders = sorted(
        group_sizes,
        key=lambda group: (
            -((limit * group_sizes[group] / total) - int(limit * group_sizes[group] / total)),
            group,
        ),
    )
    while remaining > 0:
        progressed = False
        for group in remainders:
            if quotas[group] >= group_sizes[group]:
                continue
            quotas[group] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return quotas


def _subsample(name, questions, limit, seed):
    if limit is None or limit <= 0 or len(questions) <= limit:
        return questions
    if name in STRATIFIED_DATASETS:
        groups = {}
        for question in questions:
            groups.setdefault(question[2], []).append(question)
        quotas = _allocate_stratified_counts(
            {group: len(items) for group, items in groups.items()}, limit
        )
        selected = []
        for group, items in groups.items():
            ranked = sorted(
                items, key=lambda q: _stable_key(seed, name, q[0], q[2])
            )
            selected.extend(ranked[: quotas[group]])
        selected.sort(key=lambda question: question[0])
        return selected
    picked = sorted(
        questions, key=lambda q: _stable_key(seed, name, q[0], q[2])
    )[:limit]
    return sorted(picked, key=lambda question: question[0])


def _chat_prompt(user_text, reasoning):
    think_prefix = "<think>\n" if reasoning else "<think>\n\n</think>\n\n"
    return (
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{think_prefix}"
    )


def _one_request(args, name, index, total, qid, text):
    if args.flush_cache_between_requests:
        _flush_cache(args.base_url)
    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "ignore_eos": False,
        "sampling_seed": args.seed + index,
    }
    payload = {
        "text": _chat_prompt(text, args.reasoning == "on"),
        "sampling_params": sampling_params,
        "return_logprob": False,
        "stream": False,
        "log_metrics": True,
    }
    try:
        resp = _http_json(args.base_url + "/generate", payload, timeout=args.request_timeout)
        if isinstance(resp, list):
            resp = resp[0]
        meta_info = resp["meta_info"]
        completion_tokens = int(meta_info.get("completion_tokens", 0) or 0)
        verify_ct = int(meta_info.get("spec_verify_ct", 0) or 0)
        steps = verify_ct if verify_ct > 0 else completion_tokens
        finish = meta_info.get("finish_reason")
        if isinstance(finish, dict):
            finish = finish.get("type", finish)
        result = {
            "tokens": completion_tokens,
            "steps": steps,
            "error": None,
        }
        print(
            f"[{name} {index + 1}/{total}] qid={qid} tokens={completion_tokens} "
            f"steps={steps} accept_length={completion_tokens / max(steps, 1):.2f} "
            f"latency={float(meta_info.get('e2e_latency', 0.0)):.1f}s finish={finish}",
            flush=True,
        )
        return result
    except Exception as e:
        print(f"[{name} {index + 1}/{total}] qid={qid} ERROR: {e}", flush=True)
        return {"tokens": 0, "steps": 0, "error": str(e)}


def _run_dataset(args, name, questions):
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(_one_request, args, name, index, len(questions), qid, text)
            for index, (qid, text, _category) in enumerate(questions)
        ]
        results = [f.result() for f in futures]
    wall = time.perf_counter() - start
    tokens = sum(r["tokens"] for r in results)
    errors = sum(1 for r in results if r["error"])
    accept_lengths = [r["tokens"] / r["steps"] for r in results if r["steps"] > 0]
    return {
        "dataset": name,
        "prompts": len(questions),
        "errors": errors,
        "tokens": tokens,
        "wall": wall,
        "tok_s": tokens / wall if wall > 0 else 0.0,
        "accept_length": (
            sum(accept_lengths) / len(accept_lengths) if accept_lengths else 0.0
        ),
    }


def _parse_baseline(pairs):
    baseline = {}
    for pair in pairs:
        name, _, value = pair.partition("=")
        name = DATASET_ALIASES.get(name, name)
        if name not in DATASET_NAMES or not value:
            raise SystemExit(f"--baseline expects DATASET=TOK_S pairs, got: {pair}")
        baseline[name] = float(value)
    return baseline


def _print_summary(args, rows, baseline):
    def speedup(name, tok_s):
        if name in baseline and baseline[name] > 0:
            return f"{tok_s / baseline[name]:.2f}x"
        return "-"

    header = (
        f"{'dataset':<14} {'prompts':>7} {'errors':>6} {'gen_tokens':>11} "
        f"{'wall_s':>9} {'tok/s/seq':>10} {'accept_length':>13} {'speedup':>8}"
    )
    print("=" * len(header))
    print("DFlash-TfM paper benchmark summary")
    print(
        f"base_url={args.base_url} model={args.model} temperature={args.temperature} "
        f"reasoning={args.reasoning} max_new_tokens={args.max_new_tokens} "
        f"concurrency={args.concurrency} "
        f"flush_cache_between_requests={args.flush_cache_between_requests} "
        f"per_dataset_limit={args.per_dataset_limit or 'paper'} "
        f"subsample_seed={args.subsample_seed} seed={args.seed}"
    )
    print("-" * len(header))
    print(header)
    for row in rows:
        print(
            f"{row['dataset']:<14} {row['prompts']:>7} {row['errors']:>6} "
            f"{row['tokens']:>11} {row['wall']:>9.1f} {row['tok_s']:>10.1f} "
            f"{row['accept_length']:>13.2f} {speedup(row['dataset'], row['tok_s']):>8}"
        )
    print("-" * len(header))
    macro_tok_s = sum(r["tok_s"] for r in rows) / len(rows)
    macro_accept_length = sum(r["accept_length"] for r in rows) / len(rows)
    if baseline and all(r["dataset"] in baseline for r in rows):
        macro_baseline = sum(baseline[r["dataset"]] for r in rows) / len(rows)
        macro_speedup = f"{macro_tok_s / macro_baseline:.2f}x"
    else:
        macro_speedup = "-"
    print(
        f"{'Macro Avg.':<14} {'':>7} {'':>6} {'':>11} {'':>9} {macro_tok_s:>10.1f} "
        f"{macro_accept_length:>13.2f} {macro_speedup:>8}"
    )
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(
        description="DFlash-TfM paper benchmark harness (README step 4)."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="", help="Recorded in the summary header only.")
    parser.add_argument("--datasets", nargs="+", default=["mtbench"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--reasoning", choices=("on", "off"), default="on")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--flush-cache-between-requests", action="store_true")
    parser.add_argument(
        "--per-dataset-limit",
        type=int,
        default=None,
        help=(
            "Override the seeded subsample cap for every dataset. Default is the "
            "pinned paper protocol: gsm8k and math500 capped at 128, everything "
            "else in full."
        ),
    )
    parser.add_argument("--subsample-seed", type=int, default=SUBSAMPLE_SEED)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sharechat-path", type=Path, default=SHARECHAT_DEFAULT_PATH)
    parser.add_argument(
        "--baseline",
        nargs="*",
        default=[],
        help="DATASET=TOK_S pairs from the autoregressive run, used for speedup rows.",
    )
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--server-ready-timeout", type=float, default=120.0)
    args = parser.parse_args()

    args.base_url = args.base_url.rstrip("/")
    names = [DATASET_ALIASES.get(n, n) for n in args.datasets]
    unknown = [n for n in names if n not in DATASET_NAMES]
    if unknown:
        raise SystemExit(f"unknown datasets: {unknown} (choose from {DATASET_NAMES})")
    baseline = _parse_baseline(args.baseline)

    _wait_ready(args.base_url, args.server_ready_timeout)
    try:
        info = _http_json(args.base_url + "/get_server_info", timeout=30.0)
        keys = ("model_path", "speculative_algorithm", "speculative_dflash_tfm_tree_budget")
        print("server: " + " ".join(f"{k}={info[k]}" for k in keys if info.get(k) is not None))
    except Exception:
        pass

    resolved = []
    for name in names:
        questions, snapshot = _load_dataset(name, args.sharechat_path)
        limit = (
            args.per_dataset_limit
            if args.per_dataset_limit is not None
            else DATASET_LIMITS.get(name, 0)
        )
        selected = _subsample(name, questions, limit, args.subsample_seed)
        print(f"dataset {name}: {snapshot} ({len(selected)}/{len(questions)} prompts)")
        resolved.append((name, selected))

    for _ in range(max(args.warmup_requests, 0)):
        _one_request(args, "warmup", 0, 1, resolved[0][1][0][0], resolved[0][1][0][1])
    if args.flush_cache_between_requests:
        _flush_cache(args.base_url)

    rows = [_run_dataset(args, name, selected) for name, selected in resolved]
    _print_summary(args, rows, baseline)


if __name__ == "__main__":
    main()
