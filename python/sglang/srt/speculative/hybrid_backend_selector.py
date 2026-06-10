"""Per-batch backend selector for HYBRID_SUFFIX_MTP speculative decoding.

HYBRID_SUFFIX_MTP routes each verify batch to one of three backends:

  - **SUFFIX** — arctic suffix-tree draft + EAGLE-style verify.
    Cheap CPU draft, wide chain (K = ``speculative_num_draft_tokens``,
    typically 16). Wins when the prompt or output history has repetitive
    structure the suffix tree can match.
  - **MTP** — EAGLE V2 draft model + verify. GPU draft, narrow chain
    (K = ``speculative_num_steps + 1``, typically 4). Wins when SUFFIX
    has no match but the draft model still produces useful tokens.
  - **NONE** — plain decode, no spec (K = 1). Wins when the batch is
    large enough that the extra K tokens of verify steal more time than
    the accepted drafts save.

Per-batch inputs to ``choose()``:

  - ``suffix_scores`` — per-request SUFFIX peek score (tree match strength).
  - ``mtp_scores``   — per-request MTP head top-1 probability (free signal,
    obtained without an extra forward pass).

The selector also maintains an online EMA of realized accept length per
backend (fed via ``note_accept`` after every step) to detect when SUFFIX
is paying off relative to MTP, and an ε-exploration knob so SUFFIX runs
at least ``explore_p`` of the time and its EMA stays fresh.

All thresholds are env-tunable. Defaults are set for DSv4 / GLM-5 on
H200 with agentic + long-prompt workloads; tune for other hardware.
"""

from __future__ import annotations

import logging
import os
import random
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class Backend(Enum):
    SUFFIX = "SUFFIX"
    MTP = "MTP"
    NONE = "NONE"


class HybridBackendSelector:
    """Picks SUFFIX / MTP / NONE per batch.

    Selection rule (default path; overridden by force / debug knobs below):

      1. If no suffix_scores (cold-start / empty batch): pick SUFFIX so
         the next step's keep-up populates Eagle warm state.
      2. Compute ``mean_score`` = mean of suffix_scores, ``bs`` = batch size,
         ``T_high`` = piecewise-linear threshold at this bs (see
         ``_t_high_for_bs``).
      3. Tentative tier-1: ``mean_score >= T_high`` → SUFFIX, else → MTP.
      4. If SUFFIX picked and both EMAs are warm and per-batch score is
         not overwhelming (< sfx_gate_override), demote to MTP unless
         ``ema_sfx - ema_mtp >= break_even(bs)``.
      5. NONE diversion: at ``bs >= NONE_FORCE_BS``, override pick to NONE
         (large bs saturates GPU; spec verify becomes net-negative).
         Optional legacy paths (tier-1 score-based + EMA absolute
         break-even) available below NONE_FORCE_BS, off by default.
      6. ε-explore: any would-be MTP pick is flipped to SUFFIX with
         probability ``explore_p`` so ``ema_sfx`` stays fresh.

    Debug knobs (``ARCTIC_HYBRID_SWITCH_DEBUG``):

      ``alternate``         — flip backend every step (hot-switch stress test)
      ``mtp_first_n=N``     — N MTP steps then SUFFIX (MTP→SUFFIX transition)
      ``suffix_first_n=N``  — N SUFFIX steps then MTP (SUFFIX→MTP transition)

    Force knob (``ARCTIC_HYBRID_FORCE_BACKEND=SUFFIX|MTP|NONE``) pins one
    backend for A/B testing.

    Env-tunable thresholds — see ``__init__`` for the full list. Selected
    ones:

      ARCTIC_HYBRID_T_HIGH_AT_BS{1,16,32}    piecewise SUFFIX-score cutoff
      ARCTIC_HYBRID_BREAK_EVEN_AT_BS{1,4,16,32}
                                             EMA Δaccept break-even
      ARCTIC_HYBRID_NONE_FORCE_BS=N          bs >= N → NONE (saturated GPU)
      ARCTIC_HYBRID_EMA_ALPHA=0.05           EMA smoothing factor
      ARCTIC_HYBRID_EXPLORE_P=0.01           ε-explore probability
      ARCTIC_HYBRID_SFX_GATE_OVERRIDE=5.0    score bypassing EMA gate
      ARCTIC_HYBRID_MTP_CONF_THRESHOLD=0.6   T_mtp (reserved; not used in
                                             current rule)
      ARCTIC_HYBRID_SUFFIX_LOW_THRESHOLD=1.5 T_low (reserved)
    """

    def __init__(self, server_args: ServerArgs):
        self.server_args = server_args
        self._forced = os.environ.get("ARCTIC_HYBRID_FORCE_BACKEND", "").upper()
        if self._forced and self._forced not in ("SUFFIX", "MTP", "NONE"):
            logger.warning(
                "ARCTIC_HYBRID_FORCE_BACKEND=%s ignored (expected SUFFIX|MTP|NONE)",
                self._forced,
            )
            self._forced = ""
        self._debug_mode = self._parse_debug_mode()
        self._step = 0

        def _envf(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, str(default)))
            except ValueError:
                logger.warning("%s invalid, falling back to %.2f", name, default)
                return default

        # Piecewise-linear T_high(bs): SUFFIX picked when mean_score >= T_high.
        # Threshold rises with bs because SUFFIX's wider K verify gets
        # progressively more expensive vs MTP's narrower K as the batch saturates.
        self.t_high_at_bs1 = _envf("ARCTIC_HYBRID_T_HIGH_AT_BS1", 2.0)
        self.t_high_at_bs16 = _envf("ARCTIC_HYBRID_T_HIGH_AT_BS16", 4.5)
        self.t_high_at_bs32 = _envf("ARCTIC_HYBRID_T_HIGH_AT_BS32", 7.5)
        logger.info(
            "HybridBackendSelector T_high anchors: bs1=%.2f bs16=%.2f bs32=%.2f",
            self.t_high_at_bs1,
            self.t_high_at_bs16,
            self.t_high_at_bs32,
        )

        # Periodic per-step decision logging.
        self._sel_log_every = max(
            1, int(os.environ.get("ARCTIC_HYBRID_SELECTOR_LOG_EVERY", "20"))
        )
        self._suffix_picks = 0
        self._mtp_picks = 0
        self._none_picks = 0
        self._sfx_override_picks = 0
        self._sum_mean_score = 0.0
        self._sum_mean_mtp = 0.0
        self._mtp_seen = 0

        # EMA-Δaccept gate with ε-exploration.
        # Tracks realized accept length per backend; SUFFIX is only picked
        # in the warm state if it has accumulated enough accept advantage
        # over MTP to clear a bs-dependent break-even (which approximates
        # the per-step verify-cost ratio).
        self._ema_alpha = _envf("ARCTIC_HYBRID_EMA_ALPHA", 0.05)
        self._explore_p = _envf("ARCTIC_HYBRID_EXPLORE_P", 0.01)
        self._warmup_min_samples = int(_envf("ARCTIC_HYBRID_EMA_WARMUP_N", 3))

        # When the per-batch SUFFIX score is overwhelmingly high, bypass the
        # EMA gate and pick SUFFIX. EMA is cumulative and can be stale; a
        # strong per-batch score is fresh and should override. Set <= 0 to
        # disable.
        self._sfx_gate_override = _envf("ARCTIC_HYBRID_SFX_GATE_OVERRIDE", 5.0)

        # Piecewise-linear break_even(bs) for the SUFFIX-vs-MTP EMA gate.
        # Δaccept_EMA (ema_sfx - ema_mtp) must clear this for SUFFIX to be
        # picked in the warm state. The bs=4 anchor is interpolated between
        # bs1 and bs16 by default; set it explicitly to add a kink in the
        # low-bs segment.
        self.be_at_bs1 = _envf("ARCTIC_HYBRID_BREAK_EVEN_AT_BS1", 2.0)
        self.be_at_bs16 = _envf("ARCTIC_HYBRID_BREAK_EVEN_AT_BS16", 6.0)
        self.be_at_bs32 = _envf("ARCTIC_HYBRID_BREAK_EVEN_AT_BS32", 6.0)
        _be4_default = self.be_at_bs1 + (self.be_at_bs16 - self.be_at_bs1) * 3.0 / 15.0
        self.be_at_bs4 = _envf("ARCTIC_HYBRID_BREAK_EVEN_AT_BS4", _be4_default)
        self._ema_sfx_accept = None
        self._ema_mtp_accept = None
        self._ema_sfx_n = 0
        self._ema_mtp_n = 0
        self._explore_picks = 0

        # NONE diversion — primary rule: simple bs cutoff. At large bs the
        # batch compute saturates GPU; the extra-K verify of any spec
        # backend steals work without producing enough accepts to pay back.
        # The cutoff is a function of model + hardware (not workload).
        # Set very high (e.g., 10000) to effectively disable NONE.
        self._none_force_bs = int(_envf("ARCTIC_HYBRID_NONE_FORCE_BS", 64))

        # Per-backend absolute EMA break-evens — secondary, opt-in NONE rule.
        # After the tier-1/EMA gate picks SUFFIX or MTP, if that backend's
        # EMA accept length is below its own break-even (verify cost ratio
        # vs baseline single-token decode), divert to NONE. Different
        # thresholds because K=32 SUFFIX verify is ~2× the cost of K=4 MTP
        # verify. Set the relevant threshold to 0 to disable this rule for
        # that backend.
        self._none_sfx_at_bs1 = _envf("ARCTIC_HYBRID_NONE_SFX_AT_BS1", 1.0)
        self._none_sfx_at_bs16 = _envf("ARCTIC_HYBRID_NONE_SFX_AT_BS16", 1.5)
        self._none_sfx_at_bs32 = _envf("ARCTIC_HYBRID_NONE_SFX_AT_BS32", 2.0)
        self._none_mtp_at_bs1 = _envf("ARCTIC_HYBRID_NONE_MTP_AT_BS1", 0.5)
        self._none_mtp_at_bs16 = _envf("ARCTIC_HYBRID_NONE_MTP_AT_BS16", 0.8)
        self._none_mtp_at_bs32 = _envf("ARCTIC_HYBRID_NONE_MTP_AT_BS32", 1.0)

        # Score-based fast NONE trigger — secondary, opt-in (default off).
        # When set (both > 0), at bs >= _none_min_bs, if BOTH mean SUFFIX
        # score < _none_tier1_sfx AND mean MTP confidence < _none_tier1_mtp,
        # divert to NONE without waiting for EMA warmup.
        self._none_min_bs = int(_envf("ARCTIC_HYBRID_NONE_MIN_BS", 16))
        self._none_tier1_sfx = _envf("ARCTIC_HYBRID_NONE_TIER1_SFX", 0.0)
        self._none_tier1_mtp = _envf("ARCTIC_HYBRID_NONE_TIER1_MTP", 0.0)

        # Single-threshold NONE rule — opt-in (default off). When > 0,
        # replaces per-backend break-evens with a max(ema_sfx, ema_mtp) <
        # _t_none check.
        self._t_none = _envf("ARCTIC_HYBRID_T_NONE", 0.0)

        logger.info(
            "HybridBackendSelector EMA gate: alpha=%.3f explore_p=%.3f warmup_n=%d "
            "break_even(SUFFIX-vs-MTP) anchors bs1=%.2f bs4=%.2f bs16=%.2f bs32=%.2f",
            self._ema_alpha,
            self._explore_p,
            self._warmup_min_samples,
            self.be_at_bs1,
            self.be_at_bs4,
            self.be_at_bs16,
            self.be_at_bs32,
        )
        logger.info(
            "HybridBackendSelector NONE: force_bs=%d (bs>=this → NONE); "
            "tier-1 sfx=%.2f mtp=%.2f (off if 0); "
            "per-backend EMA break-even SUFFIX bs1/16/32=%.2f/%.2f/%.2f "
            "MTP=%.2f/%.2f/%.2f; t_none=%.2f min_bs=%d",
            self._none_force_bs,
            self._none_tier1_sfx,
            self._none_tier1_mtp,
            self._none_sfx_at_bs1,
            self._none_sfx_at_bs16,
            self._none_sfx_at_bs32,
            self._none_mtp_at_bs1,
            self._none_mtp_at_bs16,
            self._none_mtp_at_bs32,
            self._t_none,
            self._none_min_bs,
        )

    def note_accept(self, backend, mean_accept):
        """Update the per-backend EMA of realized accept length. Called by
        the worker after each decode step (accept_lens is already
        materialized on CPU via the step's mandatory sync — free signal).

        NONE picks are intentionally not recorded: accept_len is trivially
        1, and feeding NONE outcomes into the SUFFIX/MTP EMAs would make
        NONE self-reinforcing.
        """
        a = self._ema_alpha
        if backend == Backend.SUFFIX:
            self._ema_sfx_accept = (
                mean_accept
                if self._ema_sfx_accept is None
                else (1 - a) * self._ema_sfx_accept + a * mean_accept
            )
            self._ema_sfx_n += 1
        elif backend == Backend.MTP:
            self._ema_mtp_accept = (
                mean_accept
                if self._ema_mtp_accept is None
                else (1 - a) * self._ema_mtp_accept + a * mean_accept
            )
            self._ema_mtp_n += 1

    def _none_threshold_sfx(self, bs: int) -> float:
        """SUFFIX absolute EMA break-even (floor below which SUFFIX is
        net-negative vs baseline at this bs). Piecewise-linear through
        ``(1, _none_sfx_at_bs1)``, ``(16, _none_sfx_at_bs16)``,
        ``(32, _none_sfx_at_bs32)``; extrapolates the (16, 32) slope past 32.
        """
        if bs <= 1:
            return self._none_sfx_at_bs1
        if bs <= 16:
            return (
                self._none_sfx_at_bs1
                + (self._none_sfx_at_bs16 - self._none_sfx_at_bs1) * (bs - 1) / 15.0
            )
        slope_hi = (self._none_sfx_at_bs32 - self._none_sfx_at_bs16) / 16.0
        return self._none_sfx_at_bs16 + slope_hi * (bs - 16)

    def _none_threshold_mtp(self, bs: int) -> float:
        """MTP absolute EMA break-even. Lower than SUFFIX's because K=4
        verify is cheaper than K=32. Piecewise-linear through the same bs
        anchors as ``_none_threshold_sfx``."""
        if bs <= 1:
            return self._none_mtp_at_bs1
        if bs <= 16:
            return (
                self._none_mtp_at_bs1
                + (self._none_mtp_at_bs16 - self._none_mtp_at_bs1) * (bs - 1) / 15.0
            )
        slope_hi = (self._none_mtp_at_bs32 - self._none_mtp_at_bs16) / 16.0
        return self._none_mtp_at_bs16 + slope_hi * (bs - 16)

    def _break_even_for_bs(self, bs: int) -> float:
        """Piecewise-linear break_even(bs) through ``(1, be_at_bs1)``,
        ``(4, be_at_bs4)``, ``(16, be_at_bs16)``, ``(32, be_at_bs32)``.
        ``ema_sfx - ema_mtp`` must clear this for SUFFIX to be picked in
        the warm state. The bs=4 anchor decouples low-bs from mid-bs so a
        steeper low-bs segment can suppress marginal SUFFIX picks at
        bs∈[1, 4] without affecting bs=8+ behavior."""
        if bs <= 1:
            return self.be_at_bs1
        if bs <= 4:
            return self.be_at_bs1 + (self.be_at_bs4 - self.be_at_bs1) * (bs - 1) / 3.0
        if bs <= 16:
            return self.be_at_bs4 + (self.be_at_bs16 - self.be_at_bs4) * (bs - 4) / 12.0
        slope_hi = (self.be_at_bs32 - self.be_at_bs16) / 16.0
        return self.be_at_bs16 + slope_hi * (bs - 16)

    def _t_high_for_bs(self, bs: int) -> float:
        """Piecewise-linear T_high(bs) through ``(1, t_high_at_bs1)``,
        ``(16, t_high_at_bs16)``, ``(32, t_high_at_bs32)``; extrapolates
        the (16, 32) slope past 32."""
        if bs <= 1:
            return self.t_high_at_bs1
        if bs <= 16:
            return (
                self.t_high_at_bs1
                + (self.t_high_at_bs16 - self.t_high_at_bs1) * (bs - 1) / 15.0
            )
        slope_hi = (self.t_high_at_bs32 - self.t_high_at_bs16) / 16.0
        return self.t_high_at_bs16 + slope_hi * (bs - 16)

    def _parse_debug_mode(self):
        raw = os.environ.get("ARCTIC_HYBRID_SWITCH_DEBUG", "").strip()
        if not raw:
            return None
        if raw == "alternate":
            return ("alternate", None)
        for prefix, who in (
            ("mtp_first_n=", Backend.MTP),
            ("suffix_first_n=", Backend.SUFFIX),
        ):
            if raw.startswith(prefix):
                try:
                    n = int(raw[len(prefix) :])
                    return (prefix.rstrip("="), (who, n))
                except ValueError:
                    pass
        logger.warning("ARCTIC_HYBRID_SWITCH_DEBUG=%s unrecognized", raw)
        return None

    def choose(
        self,
        batch: ScheduleBatch,
        suffix_scores: Optional[List[float]] = None,
        mtp_scores: Optional[List[float]] = None,
    ) -> Backend:
        if self._forced == "MTP":
            return Backend.MTP
        if self._forced == "SUFFIX":
            return Backend.SUFFIX
        if self._forced == "NONE":
            return Backend.NONE
        if self._debug_mode is not None:
            mode, payload = self._debug_mode
            step, self._step = self._step, self._step + 1
            if mode == "alternate":
                return Backend.MTP if (step % 2) else Backend.SUFFIX
            if mode in ("mtp_first_n", "suffix_first_n"):
                who, n = payload
                other = Backend.SUFFIX if who == Backend.MTP else Backend.MTP
                return who if step < n else other

        if not suffix_scores:
            # Cold-start / empty batch: no useful peek — pick SUFFIX so the
            # next step's keep-up fires (Eagle warm-up).
            return Backend.SUFFIX
        mean_score = sum(suffix_scores) / len(suffix_scores)
        mean_mtp = sum(mtp_scores) / len(mtp_scores) if mtp_scores else None
        bs = len(suffix_scores)
        t_high = self._t_high_for_bs(bs)

        # Tier-1 + EMA gate. SUFFIX requires mean_score >= T_high; once both
        # EMAs are warm, additionally requires ema_sfx - ema_mtp >=
        # break_even(bs) unless the per-batch score is overwhelmingly high.
        warmed = (
            self._ema_sfx_accept is not None
            and self._ema_mtp_accept is not None
            and self._ema_sfx_n >= self._warmup_min_samples
            and self._ema_mtp_n >= self._warmup_min_samples
        )
        if mean_score >= t_high:
            if not warmed:
                tentative = Backend.SUFFIX
            elif self._sfx_gate_override > 0 and mean_score >= self._sfx_gate_override:
                tentative = Backend.SUFFIX
                self._sfx_override_picks = getattr(self, "_sfx_override_picks", 0) + 1
            else:
                delta = self._ema_sfx_accept - self._ema_mtp_accept
                tentative = (
                    Backend.SUFFIX
                    if delta >= self._break_even_for_bs(bs)
                    else Backend.MTP
                )
        else:
            tentative = Backend.MTP

        # NONE diversion.
        #   Primary: bs >= _none_force_bs → NONE.
        #   Below that, optional secondary rules below _none_min_bs only fire
        #   when explicitly enabled (their thresholds > 0).
        pick = tentative
        if bs >= self._none_force_bs:
            pick = Backend.NONE
        elif bs >= self._none_min_bs:
            if (
                self._none_tier1_sfx > 0
                and self._none_tier1_mtp > 0
                and mean_score < self._none_tier1_sfx
                and mean_mtp is not None
                and mean_mtp < self._none_tier1_mtp
            ):
                pick = Backend.NONE
            elif warmed and self._t_none == 0:
                if tentative == Backend.SUFFIX:
                    sfx_be = self._none_threshold_sfx(bs)
                    if sfx_be > 0 and self._ema_sfx_accept < sfx_be:
                        pick = Backend.NONE
                else:
                    mtp_be = self._none_threshold_mtp(bs)
                    if mtp_be > 0 and self._ema_mtp_accept < mtp_be:
                        pick = Backend.NONE
            elif self._t_none > 0 and warmed and mean_score < t_high:
                if max(self._ema_sfx_accept, self._ema_mtp_accept) < self._t_none:
                    pick = Backend.NONE

        if pick == Backend.MTP and random.random() < self._explore_p:
            pick = Backend.SUFFIX
            self._explore_picks += 1

        # Per-step counters for periodic logging.
        self._step += 1
        self._sum_mean_score += mean_score
        if mean_mtp is not None:
            self._sum_mean_mtp += mean_mtp
            self._mtp_seen += 1
        if pick == Backend.SUFFIX:
            self._suffix_picks += 1
        elif pick == Backend.MTP:
            self._mtp_picks += 1
        else:
            self._none_picks += 1
        if self._step % self._sel_log_every == 0:
            tot = self._suffix_picks + self._mtp_picks + self._none_picks
            avg_score = self._sum_mean_score / tot if tot else 0.0
            avg_mtp = (
                self._sum_mean_mtp / self._mtp_seen if self._mtp_seen else float("nan")
            )
            ema_sfx_str = (
                f"{self._ema_sfx_accept:.2f}"
                if self._ema_sfx_accept is not None
                else "n/a"
            )
            ema_mtp_str = (
                f"{self._ema_mtp_accept:.2f}"
                if self._ema_mtp_accept is not None
                else "n/a"
            )
            logger.info(
                "[HYBRID_SELECTOR] step=%d bs=%d T_high=%.2f sfx=%.2f mtp=%.2f pick=%s | "
                "since-start: suffix=%d (%.1f%%) mtp=%d (%.1f%%) none=%d (%.1f%%) "
                "avg_sfx_score=%.2f avg_mtp_score=%.2f ema_sfx=%s ema_mtp=%s "
                "sfx_override=%d (%.1f%%)",
                self._step,
                bs,
                t_high,
                mean_score,
                mean_mtp if mean_mtp is not None else float("nan"),
                pick.value,
                self._suffix_picks,
                100.0 * self._suffix_picks / max(1, tot),
                self._mtp_picks,
                100.0 * self._mtp_picks / max(1, tot),
                self._none_picks,
                100.0 * self._none_picks / max(1, tot),
                avg_score,
                avg_mtp,
                ema_sfx_str,
                ema_mtp_str,
                self._sfx_override_picks,
                100.0 * self._sfx_override_picks / max(1, self._suffix_picks),
            )
        return pick
