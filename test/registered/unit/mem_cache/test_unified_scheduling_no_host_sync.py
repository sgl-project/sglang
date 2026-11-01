"""The unified pool's per-step allocation and release paths issue no host
sync. On the scheduler stream a sync waits for every queued op; behind the WAR
barrier that parks the host until the in-flight forward's shared reads end,
which for an extend is the whole forward."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    setup_allocator_context,
)

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

PAGE_SIZE = 4
CHAINS = {
    "full+window": dict(chain="swa", full_layer_ids=(0,), swa_layer_ids=(1,)),
    "full+state": dict(chain="mamba", full_layer_ids=(0,), state_layer_ids=(1,)),
    "full+window+state": dict(
        chain="tri", full_layer_ids=(0,), swa_layer_ids=(1,), state_layer_ids=(2,)
    ),
}


def _sync_error(fn):
    """The RuntimeError torch raises if `fn` synchronizes, or None."""
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        fn()
    except RuntimeError as exc:
        return exc
    finally:
        torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
    return None


def _cuda(values):
    return torch.tensor(values, dtype=torch.int64).cuda()


class _Step:
    """One scheduling step's allocator calls on a two-request batch: extend,
    decode, a window's tail-only extend, window retirement and release."""

    def __init__(self, allocator):
        self.allocator = allocator
        self.has_window = allocator.swa_attn_allocator is not None
        self.has_state = allocator.mamba_allocator is not None
        # Host lengths are computed before the checked call, as the scheduler
        # stages them; the device copies come from pinned-free CPU tensors.
        self.seq = torch.tensor([2 * PAGE_SIZE, PAGE_SIZE + 2], dtype=torch.int64)
        self.zero = torch.zeros(2, dtype=torch.int64)
        self.seq_d, self.zero_d, self.no_last = (
            _cuda(self.seq.tolist()),
            _cuda([0, 0]),
            _cuda([-1, -1]),
        )
        self.decode = self.seq + 1
        self.decode_d = _cuda(self.decode.tolist())
        self.tail_seq = torch.tensor([3 * PAGE_SIZE], dtype=torch.int64)
        self.tail_seq_d = _cuda(self.tail_seq.tolist())
        self.tail_prefix = torch.zeros(1, dtype=torch.int64)
        self.tail_prefix_d = _cuda([0])
        self.tail_last = _cuda([-1])

    def extend(self):
        self.out = self.allocator.alloc_extend(
            self.zero_d,
            self.zero,
            self.seq_d,
            self.seq,
            self.no_last,
            int(self.seq.sum()),
        )
        assert self.out is not None

    def decode_step(self):
        first = int(self.seq[0])
        self.last = self.out[torch.tensor([first - 1, int(self.seq.sum()) - 1])].cuda()
        return lambda: self.allocator.alloc_decode(
            self.decode_d, self.decode, self.last
        )

    def tail_extend(self):
        tokens = int(self.tail_seq[0])
        return self.allocator.alloc_extend_swa_tail(
            self.tail_prefix_d,
            self.tail_prefix,
            self.tail_seq_d,
            self.tail_seq,
            self.tail_last,
            tokens,
            window_tail_lens=dict.fromkeys(
                self.allocator.window_pool_names(), PAGE_SIZE + 1
            ),
        )

    def retire_window(self):
        self.allocator.free_window_segment(
            self.allocator.roles.window, self.out[:PAGE_SIZE], start_pos=0
        )

    def release(self, decoded):
        first = int(self.seq[0])
        # The first request's row with its decoded token: two whole pages and
        # one token of the third.
        row = torch.cat([self.out[:first], decoded[:1]])
        if self.has_window:
            # The retired first page left the window; the rest are still in it.
            self.allocator.free_full_segment(row[:PAGE_SIZE], start_pos=0)
            self.allocator.free_segment(row[PAGE_SIZE:], start_pos=PAGE_SIZE)
        else:
            self.allocator.free_segment(row, start_pos=0)

    def state_slots(self):
        slots = self.allocator.alloc_state_slots(2)
        assert slots is not None
        self.allocator.mamba_allocator.free(slots)


class TestUnifiedSchedulingNoHostSync(CustomTestCase):
    def _allocator(self, chain):
        setup_allocator_context()
        return build_unified_bundle(
            UnifiedGeometry(
                **CHAINS[chain],
                page_size=PAGE_SIZE,
                full_tokens=64 * PAGE_SIZE,
                swa_tokens=64 * PAGE_SIZE,
                device="cuda",
            )
        ).token_to_kv_pool_allocator

    def _run(self, step):
        """Every call once, checking each for a host sync; the second pass
        runs with every kernel loaded and the allocator's memory cached."""
        errors = {}
        for checked in (False, True):
            check = _sync_error if checked else (lambda fn: fn())
            errors["extend"] = check(step.extend)
            decode = step.decode_step()
            decoded = []
            errors["decode"] = check(lambda: decoded.append(decode()))
            if step.has_state:
                errors["state slots"] = check(step.state_slots)
            if step.has_window:
                tail = []
                errors["window tail extend"] = check(
                    lambda: tail.append(step.tail_extend())
                )
                errors["window retirement"] = check(step.retire_window)
            errors["release"] = check(lambda: step.release(decoded[0]))
            torch.cuda.synchronize()
            step.allocator.clear()
        return {name: error for name, error in errors.items() if error is not None}

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_per_step_paths_do_not_synchronize(self):
        for chain in CHAINS:
            with self.subTest(chain=chain):
                errors = self._run(_Step(self._allocator(chain)))
                self.assertEqual(errors, {}, f"{chain} synchronized: {errors}")


if __name__ == "__main__":
    unittest.main()
