import asyncio
import json
import pickle
import random
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from sglang.benchmark.datasets import DATASET_MAPPING, get_dataset
from sglang.benchmark.datasets.common import DatasetRow, gen_mm_prompt
from sglang.benchmark.datasets.custom import sample_custom_requests
from sglang.benchmark.datasets.generated_shared_prefix import (
    GeneratedSharedPrefixDataset,
    _zipf_group_probs,
    get_gen_prefix_cache_path,
    sample_generated_shared_prefix_requests,
)
from sglang.benchmark.datasets.image import sample_image_requests
from sglang.benchmark.datasets.mmmu import sample_mmmu_requests
from sglang.benchmark.datasets.mooncake import get_mooncake_request_over_time
from sglang.benchmark.datasets.openai_dataset import sample_openai_requests
from sglang.benchmark.datasets.random import sample_random_requests
from sglang.benchmark.datasets.sharegpt import sample_sharegpt_requests
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=40, suite="base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-c-test-cpu")


class _DummyTokenTensor:
    def __init__(self, value: int):
        self.value = value

    def numel(self) -> int:
        return self.value


def create_lightweight_tokenizer() -> PreTrainedTokenizerFast:
    """Create a local lightweight tokenizer for CPU-only dataset tests."""
    vocab = {"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "[EOS]": 3}
    vocab.update({f"tok_{i}": i + 4 for i in range(2048)})

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    hf_tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] }}:"
        "{% if message['content'] is string %}"
        "{{ message['content'] }}"
        "{% else %}"
        "{% for item in message['content'] %}"
        "{% if item['type'] == 'text' %}{{ item['text'] }}{% else %}[IMAGE]{% endif %}"
        "{% endfor %}"
        "{% endif %}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant:{% endif %}"
    )
    return hf_tokenizer


class DummyProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        self.tokenizer = tokenizer
        self.image_token_id = None

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            return_dict=False,
        )

    def __call__(self, text, images=None, padding=False, return_tensors="pt"):
        text_len = len(self.tokenizer.encode(text[0]))
        image_tokens = 4 * len(images) if images else 0
        return {"input_ids": _DummyTokenTensor(text_len + image_tokens)}


class _FakeMMMUDataset:
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def select(self, indices):
        if isinstance(indices, range):
            indices = list(indices)
        return _FakeMMMUDataset([self.records[i] for i in indices])

    def __iter__(self):
        return iter(self.records)


def make_args(**overrides):
    args = {
        "dataset_name": "sharegpt",
        "dataset_path": "",
        "num_prompts": 2,
        "sharegpt_output_len": None,
        "sharegpt_context_len": None,
        "prompt_suffix": "",
        "apply_chat_template": False,
        "tokenize_prompt": False,
        "random_input_len": 8,
        "random_output_len": 4,
        "random_range_ratio": 0.0,
        "image_count": 1,
        "random_image_count": False,
        "image_format": "png",
        "image_content": "blank",
        "image_resolution": "8x8",
        "backend": "sglang",
        "gsp_num_groups": 2,
        "gsp_prompts_per_group": 2,
        "gsp_system_prompt_len": 8,
        "gsp_question_len": 4,
        "gsp_output_len": 4,
        "gsp_range_ratio": 0.0,
        "gsp_fast_prepare": False,
        "gsp_send_routing_key": False,
        "gsp_num_turns": 1,
        "gsp_ordered": False,
        "gsp_group_distribution": "uniform",
        "gsp_zipf_alpha": None,
        "seed": 1,
        "mooncake_workload": "conversation",
        "speed_bench_category": None,
        "speed_bench_output_len": 512,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


class TestBenchmarkDatasetsAPI(unittest.TestCase):
    def setUp(self):
        self.tokenizer = create_lightweight_tokenizer()
        self.processor = DummyProcessor(self.tokenizer)
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir_path = Path(self.tmpdir.name)
        # Redirect ~ for the GSP on-disk cache to the per-test tempdir, so
        # tests never read/write the real ~/.cache/sglang/benchmark. The Zipf
        # tests in particular compare freshly generated rows against the
        # uniform path, and a stale cache file from prior runs would silently
        # short-circuit the uniform path and break that comparison.
        self._home_patch = patch(
            "sglang.benchmark.datasets.generated_shared_prefix.Path.home",
            return_value=self.tmpdir_path,
        )
        self._home_patch.start()

    def tearDown(self):
        self._home_patch.stop()
        self.tmpdir.cleanup()

    def _write_sharegpt_json(self):
        data = [
            {
                "conversations": [
                    {"value": "hello world"},
                    {"value": "answer one"},
                ]
            },
            {
                "conversations": [
                    {"value": "how are you"},
                    {"value": "answer two"},
                ]
            },
            {
                "conversations": [
                    {"value": "third prompt"},
                    {"value": "answer three"},
                ]
            },
        ]
        path = self.tmpdir_path / "sharegpt.json"
        with open(path, "w") as f:
            json.dump(data, f)
        return str(path)

    def _write_custom_jsonl(self):
        rows = [
            {
                "conversations": [
                    {"content": "custom prompt 1"},
                    {"content": "custom answer 1"},
                ]
            },
            {
                "conversations": [
                    {"value": "custom prompt 2"},
                    {"value": "custom answer 2"},
                ]
            },
        ]
        path = self.tmpdir_path / "custom.jsonl"
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return str(path)

    def _write_openai_jsonl(self):
        rows = [
            {
                "messages": [{"role": "user", "content": "What is 1+1?"}],
                "max_tokens": 7,
                "temperature": 0.3,
            },
            {
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "max_tokens": 8,
                "tools": [{"type": "function", "function": {"name": "tool_a"}}],
            },
        ]
        path = self.tmpdir_path / "openai.jsonl"
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return str(path)

    def _write_speed_bench_jsonl(self):
        rows = [
            {
                "question_id": "sb_001",
                "category": "low_entropy",
                "turns": ["Complete this Python function: def add(a, b):"],
            },
            {
                "question_id": "sb_002",
                "category": "mixed",
                "turns": [
                    "Explain the concept of attention mechanisms in transformers."
                ],
            },
            {
                "question_id": "sb_003",
                "category": "high_entropy",
                "turns": ["Write a short story about a robot discovering music."],
            },
            {
                "question_id": "sb_004",
                "category": "low_entropy",
                "turns": [
                    "Sort the following list in ascending order: [5, 2, 8, 1, 9]"
                ],
            },
        ]
        path = self.tmpdir_path / "speed_bench.jsonl"
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return str(path)

    def _write_mooncake_jsonl(self):
        rows = [
            {"timestamp": 1000, "hash_ids": [1, 2], "output_length": 5},
            {"timestamp": 2000, "hash_ids": [3, 4], "output_length": 6},
        ]
        path = self.tmpdir_path / "mooncake.jsonl"
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return str(path)

    async def _collect_mooncake_rows(self, records):
        out = []
        async for row in get_mooncake_request_over_time(
            input_requests=records,
            tokenizer=self.tokenizer,
            slowdown_factor=0.0,
            num_rounds=1,
        ):
            out.append(row)
        return out

    def test_sharegpt_sampler(self):
        dataset_path = self._write_sharegpt_json()
        rows = sample_sharegpt_requests(
            dataset_path=dataset_path,
            num_requests=2,
            tokenizer=self.tokenizer,
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(isinstance(row, DatasetRow) for row in rows))

    def test_random_sampler(self):
        dataset_path = self._write_sharegpt_json()
        rows_text = sample_random_requests(
            input_len=8,
            output_len=4,
            num_prompts=2,
            range_ratio=0.0,
            tokenizer=self.tokenizer,
            dataset_path=dataset_path,
            random_sample=False,
            return_text=True,
        )
        rows_ids = sample_random_requests(
            input_len=8,
            output_len=4,
            num_prompts=2,
            range_ratio=0.0,
            tokenizer=self.tokenizer,
            dataset_path=dataset_path,
            random_sample=False,
            return_text=False,
        )
        self.assertTrue(all(isinstance(row, DatasetRow) for row in rows_text))
        self.assertTrue(all(isinstance(row.prompt, list) for row in rows_ids))

    def test_custom_sampler(self):
        dataset_path = self._write_custom_jsonl()
        rows = sample_custom_requests(
            dataset_path=dataset_path,
            num_requests=2,
            tokenizer=self.tokenizer,
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(isinstance(row, DatasetRow) for row in rows))

    def test_openai_sampler(self):
        dataset_path = self._write_openai_jsonl()
        rows = sample_openai_requests(
            dataset_path=dataset_path,
            num_requests=2,
            tokenizer=self.tokenizer,
        )
        self.assertEqual(len(rows), 2)
        self.assertIn("temperature", rows[0].extra_request_body)
        self.assertIn("tools", rows[1].extra_request_body)

    def test_generated_shared_prefix_sampler(self):
        args = make_args(gsp_num_groups=2, gsp_prompts_per_group=2)
        rows = sample_generated_shared_prefix_requests(
            num_groups=args.gsp_num_groups,
            prompts_per_group=args.gsp_prompts_per_group,
            system_prompt_len=args.gsp_system_prompt_len,
            question_len=args.gsp_question_len,
            output_len=args.gsp_output_len,
            range_ratio=args.gsp_range_ratio,
            tokenizer=self.tokenizer,
            seed=args.seed,
        )
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(isinstance(row, DatasetRow) for row in rows))

    def test_image_sampler(self):
        rows = sample_image_requests(
            num_requests=2,
            image_count=1,
            input_len=8,
            output_len=4,
            range_ratio=0.0,
            processor=self.processor,
            image_content="blank",
            image_format="png",
            image_resolution="8x8",
            backend="sglang",
            random_image_count=False,
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(isinstance(row, DatasetRow) for row in rows))
        self.assertTrue(all(row.image_data for row in rows))

    def test_gen_mm_prompt_excludes_special_tokens(self):
        tokenizer = create_lightweight_tokenizer()
        multimodal_special_tokens = [
            "<|image_pad|>",
            "<|video_pad|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|vision_pad|>",
        ]
        tokenizer.add_special_tokens(
            {"additional_special_tokens": multimodal_special_tokens}
        )
        special_token_ids = set(
            tokenizer.convert_tokens_to_ids(multimodal_special_tokens)
        )
        image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        captured_population = {}

        def fake_choices(population, k):
            captured_population["tokens"] = population
            return population[:k]

        with patch(
            "sglang.benchmark.datasets.common.random.choices",
            side_effect=fake_choices,
        ):
            gen_mm_prompt(tokenizer, image_pad_id, token_num=8)

        sampled_pool = set(captured_population["tokens"])
        self.assertFalse(special_token_ids & sampled_pool)
        self.assertTrue(sampled_pool)

    def test_mmmu_sampler(self):
        fake_records = [
            {"image_1": Image.new("RGB", (4, 4), color="white"), "question": "q1"},
            {"image_1": Image.new("RGB", (4, 4), color="white"), "question": "q2"},
            {"image_1": Image.new("RGB", (4, 4), color="white"), "question": "q3"},
        ]
        fake_dataset = _FakeMMMUDataset(fake_records)
        with patch(
            "sglang.benchmark.datasets.mmmu.load_dataset", return_value=fake_dataset
        ):
            rows = sample_mmmu_requests(
                num_requests=2,
                processor=self.processor,
                backend="sglang",
                fixed_output_len=6,
                random_sample=False,
            )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(isinstance(row, DatasetRow) for row in rows))

    def test_mooncake_scheduler(self):
        records = [
            {"timestamp": 1000, "hash_ids": [1], "output_length": 5},
            {"timestamp": 2000, "hash_ids": [2], "output_length": 6},
        ]
        rows = asyncio.run(self._collect_mooncake_rows(records))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(isinstance(row, DatasetRow) for row in rows))

    def test_speed_bench_sampler(self):
        dataset_path = self._write_speed_bench_jsonl()
        args = make_args(
            dataset_name="speed-bench",
            dataset_path=dataset_path,
            num_prompts=3,
        )
        from sglang.benchmark.datasets.speed_bench import SpeedBenchDataset

        dataset = SpeedBenchDataset.from_args(args)
        rows = dataset.load(self.tokenizer)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(isinstance(row, DatasetRow) for row in rows))
        self.assertTrue(all(row.output_len == 512 for row in rows))
        self.assertTrue(all(row.prompt_len > 0 for row in rows))

    def test_speed_bench_category_filter(self):
        dataset_path = self._write_speed_bench_jsonl()
        args = make_args(
            dataset_name="speed-bench",
            dataset_path=dataset_path,
            num_prompts=2,
            speed_bench_category="low_entropy",
        )
        from sglang.benchmark.datasets.speed_bench import SpeedBenchDataset

        dataset = SpeedBenchDataset.from_args(args)
        rows = dataset.load(self.tokenizer)
        # Only 2 low_entropy rows in the fixture, num_prompts=2
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(isinstance(row, DatasetRow) for row in rows))

    def test_speed_bench_output_len_override(self):
        dataset_path = self._write_speed_bench_jsonl()
        args = make_args(
            dataset_name="speed-bench",
            dataset_path=dataset_path,
            num_prompts=2,
            speed_bench_output_len=128,
        )
        from sglang.benchmark.datasets.speed_bench import SpeedBenchDataset

        dataset = SpeedBenchDataset.from_args(args)
        rows = dataset.load(self.tokenizer)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row.output_len == 128 for row in rows))

    def test_speed_bench_empty_category_raises(self):
        dataset_path = self._write_speed_bench_jsonl()
        args = make_args(
            dataset_name="speed-bench",
            dataset_path=dataset_path,
            num_prompts=1,
            speed_bench_category="nonexistent_category",
        )
        from sglang.benchmark.datasets.speed_bench import SpeedBenchDataset

        dataset = SpeedBenchDataset.from_args(args)
        with self.assertRaises(ValueError):
            dataset.load(self.tokenizer)

    def test_speed_bench_no_path_raises(self):
        args = make_args(
            dataset_name="speed-bench",
            dataset_path="",
            num_prompts=1,
        )
        from sglang.benchmark.datasets.speed_bench import SpeedBenchDataset

        with self.assertRaises(ValueError):
            SpeedBenchDataset.from_args(args)

    def test_dataset_mapping_and_dispatch(self):
        expected = {
            "sharegpt",
            "custom",
            "openai",
            "random",
            "random-ids",
            "generated-shared-prefix",
            "mmmu",
            "image",
            "mooncake",
            "speed-bench",
        }
        self.assertTrue(expected.issubset(set(DATASET_MAPPING.keys())))

        sharegpt_path = self._write_sharegpt_json()
        mooncake_path = self._write_mooncake_jsonl()

        random_args = make_args(dataset_name="random-ids", tokenize_prompt=True)
        random_rows = get_dataset(random_args, self.tokenizer, model_id="dummy-model")
        self.assertEqual(len(random_rows), random_args.num_prompts)
        self.assertTrue(all(isinstance(row.prompt, list) for row in random_rows))

        sharegpt_args = make_args(dataset_name="sharegpt", dataset_path=sharegpt_path)
        sharegpt_rows = get_dataset(
            sharegpt_args, self.tokenizer, model_id="dummy-model"
        )
        self.assertEqual(len(sharegpt_rows), sharegpt_args.num_prompts)

        mooncake_args = make_args(
            dataset_name="mooncake",
            dataset_path=mooncake_path,
            num_prompts=1,
        )
        mooncake_rows = get_dataset(
            mooncake_args, self.tokenizer, model_id="dummy-model"
        )
        self.assertEqual(len(mooncake_rows), 1)
        self.assertIsInstance(mooncake_rows[0], dict)

        with patch(
            "sglang.benchmark.datasets.image.get_processor",
            return_value=self.processor,
        ):
            image_args = make_args(dataset_name="image")
            image_rows = get_dataset(image_args, self.tokenizer, model_id="dummy-model")
        self.assertEqual(len(image_rows), image_args.num_prompts)

        fake_mmmu_dataset = _FakeMMMUDataset(
            [{"image_1": Image.new("RGB", (4, 4), color="white"), "question": "q"}]
        )
        with (
            patch(
                "sglang.benchmark.datasets.mmmu.get_processor",
                return_value=self.processor,
            ),
            patch(
                "sglang.benchmark.datasets.mmmu.load_dataset",
                return_value=fake_mmmu_dataset,
            ),
        ):
            mmmu_args = make_args(dataset_name="mmmu", num_prompts=1)
            mmmu_rows = get_dataset(mmmu_args, self.tokenizer, model_id="dummy-model")
        self.assertEqual(len(mmmu_rows), 1)

        gsp_args = make_args(
            dataset_name="generated-shared-prefix",
            gsp_num_groups=2,
            gsp_prompts_per_group=2,
        )
        gsp_rows = get_dataset(gsp_args, self.tokenizer, model_id="dummy-model")
        self.assertEqual(len(gsp_rows), 4)
        self.assertTrue(all(isinstance(row, DatasetRow) for row in gsp_rows))

        speed_bench_path = self._write_speed_bench_jsonl()
        speed_bench_args = make_args(
            dataset_name="speed-bench",
            dataset_path=speed_bench_path,
            num_prompts=2,
        )
        speed_bench_rows = get_dataset(
            speed_bench_args, self.tokenizer, model_id="dummy-model"
        )
        self.assertEqual(len(speed_bench_rows), 2)
        self.assertTrue(all(isinstance(row, DatasetRow) for row in speed_bench_rows))

    def test_get_dataset_unknown_dataset(self):
        args = make_args(dataset_name="not-a-dataset")
        with self.assertRaises(ValueError):
            get_dataset(args, self.tokenizer, model_id="dummy-model")

    # ------------------------------------------------------------------
    # Generated-shared-prefix Zipf sampling
    # ------------------------------------------------------------------

    def _run_gsp(
        self,
        *,
        mode="uniform",
        alpha=None,
        seed=42,
        num_groups=4,
        prompts_per_group=5,
        num_turns=1,
        send_routing_key=False,
        ordered=True,
        range_ratio=1.0,
        system_prompt_len=4,
        question_len=3,
        output_len=2,
        fast_prepare=True,
        global_seed=None,
    ):
        # GSP's own `seed` kwarg only feeds the cache filename; reproducibility
        # of compute_random_lens / gen_prompt comes from seeding the module
        # globals before calling. Tests must seed both random and numpy here.
        seed_for_globals = global_seed if global_seed is not None else seed
        random.seed(seed_for_globals)
        np.random.seed(seed_for_globals)
        return sample_generated_shared_prefix_requests(
            num_groups=num_groups,
            prompts_per_group=prompts_per_group,
            system_prompt_len=system_prompt_len,
            question_len=question_len,
            output_len=output_len,
            range_ratio=range_ratio,
            tokenizer=self.tokenizer,
            seed=seed,
            send_routing_key=send_routing_key,
            num_turns=num_turns,
            fast_prepare=fast_prepare,
            ordered=ordered,
            group_distribution=mode,
            zipf_alpha=alpha,
        )

    @staticmethod
    def _row_fields(rows):
        return [(r.prompt, r.prompt_len, r.output_len, r.routing_key) for r in rows]

    def test_gsp_uniform_default_unchanged(self):
        # Uniform mode returns the documented number of rows and is
        # bit-reproducible under fixed seeding of the global RNGs.
        rows_a = self._run_gsp(
            mode="uniform", num_groups=3, prompts_per_group=4, seed=7
        )
        rows_b = self._run_gsp(
            mode="uniform", num_groups=3, prompts_per_group=4, seed=7
        )
        self.assertEqual(len(rows_a), 3 * 4)
        self.assertEqual(self._row_fields(rows_a), self._row_fields(rows_b))

    def test_gsp_uniform_cache_path_format_unchanged(self):
        # The uniform-mode cache filename keeps its existing
        # gen_shared_prefix_<seed>_<N>_<P>_<sysL>_<qL>_<outL>_<TokenizerCls>.pkl
        # shape. The trailing class name is a transformers/tokenizers internal
        # detail (TokenizersBackend / PreTrainedTokenizerFast depending on
        # version), so we only pin the deterministic numeric portion.
        path = get_gen_prefix_cache_path(
            seed=7,
            num_groups=3,
            prompts_per_group=4,
            system_prompt_len=16,
            question_len=8,
            output_len=4,
            tokenizer=self.tokenizer,
        )
        self.assertTrue(path.name.startswith("gen_shared_prefix_7_3_4_16_8_4_"))
        self.assertTrue(path.name.endswith(".pkl"))
        self.assertEqual(path.parent, Path.home() / ".cache" / "sglang" / "benchmark")

    def test_zipf_group_probs_helper(self):
        # Rank-based probability vector: weight(rank) = 1 / rank ** alpha,
        # normalized to sum to 1, with rank starting at 1.
        probs_n3_a1 = _zipf_group_probs(3, 1.0)
        expected_n3_a1 = np.array([6.0, 3.0, 2.0]) / 11.0
        np.testing.assert_allclose(probs_n3_a1, expected_n3_a1, atol=1e-12)
        self.assertAlmostEqual(float(probs_n3_a1.sum()), 1.0, places=12)

        probs_n4_a15 = _zipf_group_probs(4, 1.5)
        ranks = np.arange(1, 5, dtype=np.float64)
        ref = 1.0 / ranks**1.5
        ref = ref / ref.sum()
        np.testing.assert_allclose(probs_n4_a15, ref, atol=1e-12)
        # Three-decimal pin against a hand-computable reference.
        np.testing.assert_allclose(
            np.round(probs_n4_a15, 3),
            np.array([0.598, 0.212, 0.115, 0.075]),
            atol=1e-3,
        )

    def test_zipf_group_probs_not_lora_skewed_formula(self):
        # The helper must NOT use the LoRA `skewed` alpha**-i exponential
        # formula; for alpha=1.5, N=4 the two formulas differ noticeably.
        actual = _zipf_group_probs(4, 1.5)
        lora_weights = np.array([1.5**-i for i in range(4)], dtype=np.float64)
        lora_probs = lora_weights / lora_weights.sum()
        self.assertFalse(
            np.allclose(actual, lora_probs, atol=1e-3),
            "Zipf helper must use rank-based 1/rank**alpha, not LoRA alpha**-i",
        )

    def test_zipf_reproducible_with_seed(self):
        # Same seed + same args -> identical rows, including order, under
        # both the in-order and shuffled paths.
        kwargs = dict(
            mode="zipf", alpha=1.7, seed=11, num_groups=4, prompts_per_group=10
        )
        rows_a = self._run_gsp(**kwargs)
        rows_b = self._run_gsp(**kwargs)
        self.assertEqual(len(rows_a), 4 * 10)
        self.assertEqual(self._row_fields(rows_a), self._row_fields(rows_b))

        # Also under the shuffled path.
        rows_c = self._run_gsp(ordered=False, **kwargs)
        rows_d = self._run_gsp(ordered=False, **kwargs)
        self.assertEqual(self._row_fields(rows_c), self._row_fields(rows_d))

    def test_zipf_different_seeds_differ(self):
        # Different seeds -> at least one differing slot under Zipf sampling.
        base = dict(mode="zipf", alpha=1.7, num_groups=4, prompts_per_group=10)
        rows_a = self._run_gsp(seed=11, **base)
        rows_b = self._run_gsp(seed=12, **base)
        self.assertEqual(len(rows_a), len(rows_b))
        self.assertNotEqual(self._row_fields(rows_a), self._row_fields(rows_b))

    def test_zipf_does_not_perturb_global_random_state(self):
        # The Zipf branch must consume zero draws from the global random /
        # numpy.random state. Therefore the per-slot generated questions and
        # system prompts under uniform and Zipf modes for the same args and
        # the same global seed are byte-equal.
        common = dict(
            num_groups=4,
            prompts_per_group=6,
            system_prompt_len=4,
            question_len=3,
            output_len=2,
            range_ratio=1.0,
            seed=99,
            ordered=True,
            send_routing_key=False,
            fast_prepare=True,
            global_seed=99,
        )
        uniform_rows = self._run_gsp(mode="uniform", **common)
        zipf_rows = self._run_gsp(mode="zipf", alpha=1.3, **common)

        # Slot i in uniform mode pairs system_prompts[i // P] with
        # questions[i // P][i % P], so the question substring after the
        # delimiter is exactly the i-th question. Same construction is used by
        # the Zipf branch (only the system prompt changes per slot), so the
        # question substrings must match slot-by-slot under the same global
        # seed.
        delim = "\n\n"

        def question_of(prompt):
            return prompt.split(delim, 1)[1]

        uniform_questions = [question_of(r.prompt) for r in uniform_rows]
        zipf_questions = [question_of(r.prompt) for r in zipf_rows]
        self.assertEqual(uniform_questions, zipf_questions)

        # The set of system prompts (which the gen_prompt path generates) must
        # also match between modes (set equality, since Zipf reuses prefixes).
        def system_of(prompt):
            return prompt.split(delim, 1)[0]

        self.assertEqual(
            set(system_of(r.prompt) for r in uniform_rows),
            set(system_of(r.prompt) for r in zipf_rows),
        )

    def test_zipf_deterministic_per_group_counts(self):
        # The per-group counts are deterministic and pinned for a known
        # (num_groups, prompts_per_group, alpha, seed) tuple. Any drift in
        # the Zipf sampling implementation will trip this assertion.
        rows = self._run_gsp(
            mode="zipf",
            alpha=2.0,
            seed=0,
            num_groups=4,
            prompts_per_group=25,
            send_routing_key=True,
            ordered=True,
        )
        self.assertEqual(len(rows), 4 * 25)
        # routing_key format is "<uuid8>_<timestamp>_<group_idx>".
        per_group = Counter(int(r.routing_key.rsplit("_", 1)[-1]) for r in rows)
        # Pinned counts derived from the implementation for
        # (N=4, P=25, alpha=2.0, seed=0) using numpy.random.default_rng(seed)
        # and rng.choice over _zipf_group_probs(N, alpha).
        self.assertEqual(
            dict(per_group),
            {0: 63, 1: 18, 2: 12, 3: 7},
        )
        # Independent skew sanity check: rank-1 (hottest) > rank-N (coldest).
        self.assertGreater(per_group[0], per_group[3])

    def test_zipf_uses_distinct_cache_from_uniform(self):
        # The on-disk cache key includes group_distribution and zipf_alpha,
        # so uniform mode, zipf alpha=1.0, and zipf alpha=2.0 each get their
        # own file. Uniform mode never reads a zipf cache and vice versa.
        from sglang.benchmark.datasets import generated_shared_prefix as gsp_mod

        fake_home = self.tmpdir_path / "fakehome"
        fake_home.mkdir()

        common = dict(
            num_groups=2,
            prompts_per_group=3,
            system_prompt_len=4,
            question_len=3,
            output_len=2,
            range_ratio=1.0,
            seed=5,
            send_routing_key=False,
            num_turns=1,
            fast_prepare=True,
            ordered=True,
        )

        with patch.object(gsp_mod.Path, "home", return_value=fake_home):
            uniform_path = get_gen_prefix_cache_path(
                seed=common["seed"],
                num_groups=common["num_groups"],
                prompts_per_group=common["prompts_per_group"],
                system_prompt_len=common["system_prompt_len"],
                question_len=common["question_len"],
                output_len=common["output_len"],
                tokenizer=self.tokenizer,
            )
            zipf_path_a = get_gen_prefix_cache_path(
                seed=common["seed"],
                num_groups=common["num_groups"],
                prompts_per_group=common["prompts_per_group"],
                system_prompt_len=common["system_prompt_len"],
                question_len=common["question_len"],
                output_len=common["output_len"],
                tokenizer=self.tokenizer,
                group_distribution="zipf",
                zipf_alpha=1.5,
            )
            zipf_path_b = get_gen_prefix_cache_path(
                seed=common["seed"],
                num_groups=common["num_groups"],
                prompts_per_group=common["prompts_per_group"],
                system_prompt_len=common["system_prompt_len"],
                question_len=common["question_len"],
                output_len=common["output_len"],
                tokenizer=self.tokenizer,
                group_distribution="zipf",
                zipf_alpha=2.0,
            )
            self.assertNotEqual(uniform_path, zipf_path_a)
            self.assertNotEqual(zipf_path_a, zipf_path_b)

            # Run each mode; each writes its own cache file.
            self._run_gsp(mode="uniform", **common)
            self._run_gsp(mode="zipf", alpha=1.5, **common)
            self._run_gsp(mode="zipf", alpha=2.0, **common)
            self.assertTrue(uniform_path.exists())
            self.assertTrue(zipf_path_a.exists())
            self.assertTrue(zipf_path_b.exists())

            # Sentinel into the uniform cache: zipf must not read it.
            sentinel = [DatasetRow(prompt="SENTINEL", prompt_len=1, output_len=1)]
            with open(uniform_path, "wb") as f:
                pickle.dump(sentinel, f)
            zipf_rows = self._run_gsp(mode="zipf", alpha=1.5, **common)
            self.assertNotEqual(zipf_rows, sentinel)

            # Second zipf call with same args must load from cache (no
            # regeneration). Mutate the zipf cache to a sentinel and confirm.
            zipf_sentinel = [
                DatasetRow(prompt="ZIPF_SENTINEL", prompt_len=1, output_len=1)
            ]
            with open(zipf_path_a, "wb") as f:
                pickle.dump(zipf_sentinel, f)
            reloaded = self._run_gsp(mode="zipf", alpha=1.5, **common)
            self.assertEqual(reloaded, zipf_sentinel)

    def test_zipf_total_rows_and_unique_prompts(self):
        # Total returned row count under Zipf equals num_groups *
        # prompts_per_group (identical to uniform mode) and every prompt
        # string is unique even when groups repeat.
        rows = self._run_gsp(
            mode="zipf",
            alpha=2.5,
            seed=3,
            num_groups=4,
            prompts_per_group=10,
            send_routing_key=False,
        )
        self.assertEqual(len(rows), 4 * 10)
        self.assertEqual(len({r.prompt for r in rows}), len(rows))

    def test_zipf_ordered_preserves_generation_order(self):
        # With ordered=True, output preserves the sampled order and matches
        # an independently re-derived group sequence from default_rng(seed).
        rows = self._run_gsp(
            mode="zipf",
            alpha=1.5,
            seed=21,
            num_groups=3,
            prompts_per_group=8,
            send_routing_key=True,
            ordered=True,
        )
        observed_groups = [int(r.routing_key.rsplit("_", 1)[-1]) for r in rows]

        # Independently reproduce the expected group sequence: an isolated
        # default_rng(seed) over _zipf_group_probs(N, alpha) sampling
        # N * P slots.
        expected_rng = np.random.default_rng(21)
        expected_probs = _zipf_group_probs(3, 1.5)
        expected_groups = expected_rng.choice(
            3, size=3 * 8, replace=True, p=expected_probs
        ).tolist()
        self.assertEqual(observed_groups, expected_groups)

    def test_zipf_shuffle_path_matches_uniform_shuffle(self):
        # When ordered=False, both modes go through random.shuffle on a list
        # of equal length, so the same global RNG seed yields the same
        # permutation. Verified indirectly: two Zipf calls with the same
        # global seed produce identical orderings.
        kwargs = dict(
            mode="zipf",
            alpha=1.2,
            seed=8,
            num_groups=4,
            prompts_per_group=6,
            send_routing_key=False,
            ordered=False,
            global_seed=8,
        )
        rows_a = self._run_gsp(**kwargs)
        rows_b = self._run_gsp(**kwargs)
        self.assertEqual(self._row_fields(rows_a), self._row_fields(rows_b))

    # ------------------------------------------------------------------
    # CLI / from_args validation
    # ------------------------------------------------------------------

    def test_from_args_rejects_invalid_distribution_and_alpha(self):
        # Defensive validation in from_args protects in-process callers
        # that build a Namespace by hand and bypass the argparse boundary
        # in bench_serving.py. Covers: unknown distribution, zipf without
        # alpha, uniform with alpha, and non-finite/non-positive alpha.
        cases = [
            {"gsp_group_distribution": "not-a-distribution", "gsp_zipf_alpha": None},
            {"gsp_group_distribution": "zipf", "gsp_zipf_alpha": None},
            {"gsp_group_distribution": "uniform", "gsp_zipf_alpha": 1.0},
            {"gsp_group_distribution": "zipf", "gsp_zipf_alpha": 0.0},
            {"gsp_group_distribution": "zipf", "gsp_zipf_alpha": -0.5},
            {"gsp_group_distribution": "zipf", "gsp_zipf_alpha": float("nan")},
            {"gsp_group_distribution": "zipf", "gsp_zipf_alpha": float("inf")},
            {"gsp_group_distribution": "zipf", "gsp_zipf_alpha": float("-inf")},
        ]
        for case in cases:
            args = make_args(dataset_name="generated-shared-prefix", **case)
            with self.assertRaises(ValueError, msg=f"case={case}"):
                GeneratedSharedPrefixDataset.from_args(args)

    def test_bench_serving_help_and_invalid_choice_argparse(self):
        # Subprocess-driven coverage of the live CLI: --help advertises both
        # flags with the rank-based Zipf formula and the alpha constraint,
        # and argparse rejects an unknown distribution choice.
        help_res = subprocess.run(
            [sys.executable, "-m", "sglang.benchmark.serving", "--help"],
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(help_res.returncode, 0, help_res.stderr)
        out = help_res.stdout
        # Both new flags appear.
        self.assertIn("--gsp-group-distribution", out)
        self.assertIn("--gsp-zipf-alpha", out)
        # Rank-based Zipf formula and alpha constraint are documented.
        self.assertIn("1/rank**alpha", out)
        self.assertIn("rank starts at 1", out)
        self.assertIn("finite float", out)

        # Argparse rejects unknown distribution choice.
        bad_choice_res = subprocess.run(
            [
                sys.executable,
                "-m",
                "sglang.benchmark.serving",
                "--dataset-name",
                "generated-shared-prefix",
                "--gsp-group-distribution",
                "invalid_name",
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertNotEqual(bad_choice_res.returncode, 0)
        self.assertIn("invalid choice", (bad_choice_res.stderr + bad_choice_res.stdout))

    def test_bench_serving_cli_rejects_zipf_without_alpha_before_server(self):
        # Malformed CLI combinations (zipf with no alpha) must fail at
        # argparse time so users see the GSP-flag error directly, not a
        # downstream connection or model-fetch failure.
        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "sglang.benchmark.serving",
                "--dataset-name",
                "generated-shared-prefix",
                "--gsp-group-distribution",
                "zipf",
                "--ready-check-timeout-sec",
                "0",
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
        # parser.error() exits with code 2 (argparse convention).
        self.assertEqual(res.returncode, 2, res.stderr)
        stderr = res.stderr + res.stdout
        self.assertIn("--gsp-group-distribution", stderr)
        self.assertIn("--gsp-zipf-alpha", stderr)
        # The error must mention the GSP flags directly, not a network or
        # model-discovery problem masquerading as the failure.
        for forbidden in [
            "HTTPConnectionPool",
            "HTTPSConnectionPool",
            "Connection refused",
            "Failed to fetch model",
            "Traceback",
        ]:
            self.assertNotIn(forbidden, stderr)

    def test_bench_serving_cli_rejects_uniform_with_alpha_before_server(self):
        # The complementary malformation: uniform distribution with an
        # explicit alpha value. Must also fail at argparse time.
        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "sglang.benchmark.serving",
                "--dataset-name",
                "generated-shared-prefix",
                "--gsp-group-distribution",
                "uniform",
                "--gsp-zipf-alpha",
                "1.0",
                "--ready-check-timeout-sec",
                "0",
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(res.returncode, 2, res.stderr)
        stderr = res.stderr + res.stdout
        self.assertIn("--gsp-group-distribution", stderr)
        self.assertIn("--gsp-zipf-alpha", stderr)
        for forbidden in [
            "HTTPConnectionPool",
            "HTTPSConnectionPool",
            "Connection refused",
            "Failed to fetch model",
            "Traceback",
        ]:
            self.assertNotIn(forbidden, stderr)


if __name__ == "__main__":
    unittest.main()
