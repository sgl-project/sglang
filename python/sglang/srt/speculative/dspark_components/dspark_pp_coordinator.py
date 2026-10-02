"""Synchronous PP stage execution for the static target-KV DSpark contract."""

import hashlib

import msgspec
import torch
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockResult,
    DraftProposal,
)
from sglang.srt.speculative.dspark_components.dspark_verify import AcceptOuts


def _token_digest(tokens):
    return hashlib.sha256(msgspec.msgpack.encode(tokens)).hexdigest()


def _integer_tensor(value, shape, device):
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != tuple(shape)
        or value.dtype not in (torch.int32, torch.int64)
        or value.device != device
    ):
        raise ValueError("invalid pipeline integer tensor shape, dtype or device")
    return value


class DSparkPPCoordinator:
    """Run one agreed batch through all stages before allowing another batch.

    Every DP=1, CP=1 world rank must call this on the same ordered scheduler
    batch, including idle turns (None). The caller owns scheduler field isolation
    and subsequent result processing. No other traffic may use these PP groups
    during the call. A failure requires worker restart, not a partial-step retry.
    This class does not implement request admission, retraction or P/D readiness.
    """

    def __init__(self, *, worker, world_group, pp_group, tp_group):
        self.worker = worker
        self.world = world_group
        self.pp = pp_group
        self.tp = tp_group
        self.owner = (pp_group.world_size - 1) * tp_group.world_size
        self.sequence = 0
        self.failed = False
        self._active = False

    @property
    def _is_owner(self):
        return self.world.rank_in_group == self.owner

    def _fence(self, action):
        value, error = None, None
        try:
            value = action()
        except Exception as failure:  # noqa: BLE001 - Propagate local phase failures.
            error = f"{type(failure).__name__}: {failure}"
        errors = self.world.all_gather_object(error)
        if any(item is not None for item in errors):
            details = "; ".join(
                f"rank {rank}: {item}"
                for rank, item in enumerate(errors)
                if item is not None
            )
            raise RuntimeError(f"DSpark PP phase failed: {details}")
        return value

    def _signature(self, batch):
        if self.failed or self._active:
            raise RuntimeError("DSpark PP coordinator requires worker restart")
        if (
            self.pp.world_size < 2
            or self.world.world_size != self.pp.world_size * self.tp.world_size
            or self.worker.ps.dp_size != 1
            or self.worker.ps.attn_dp_size != 1
            or self.worker.ps.attn_cp_size != 1
            or self.worker.ps.attn_dcp_size != 1
            or self.world.rank_in_group
            != self.pp.rank_in_group * self.tp.world_size + self.tp.rank_in_group
        ):
            raise ValueError("DSpark PP requires a PP-major/TP-minor DP1 CP1 world")
        worker = self.worker
        if (
            worker._target_kv_contract is None
            or worker._verify_planner.mode_value != "static"
            or worker.carries_confidence
        ):
            raise ValueError("DSpark PP requires a static target-KV draft")
        worker._check_no_pending_step()
        if batch is None:
            return (1, self.sequence, None)
        if (
            not batch.spec_algorithm.is_dspark()
            or batch.return_logprob
            or batch.return_hidden_states
            or not batch.reqs
        ):
            raise ValueError("unsupported DSpark PP batch or output mode")
        if batch.forward_mode.is_extend():
            phase = "prefill"
        elif batch.forward_mode.is_decode():
            phase = "decode"
        else:
            raise ValueError("DSpark PP requires prefill or decode")
        bs = len(batch.reqs)
        self.device = batch.seq_lens.device
        self.vocab_size = worker.model_runner.model_config.vocab_size
        self.stride = worker.verify_num_draft_tokens
        if self.stride < 2 or self.vocab_size < 1:
            raise ValueError("invalid DSpark PP vocabulary or verify stride")
        lengths = _integer_tensor(batch.seq_lens, (bs,), self.device).cpu()
        if not torch.equal(lengths, batch.seq_lens_cpu) or bool((lengths < 0).any()):
            raise ValueError("DSpark PP CPU/device prefix lengths disagree")
        requests = tuple(
            (
                req.rid,
                req.kv_committed_len,
                len(req.output_ids),
                _token_digest((req.origin_input_ids, req.output_ids)),
            )
            for req in batch.reqs
        )
        if len({item[0] for item in requests}) != bs:
            raise ValueError("duplicate request identity in DSpark PP batch")
        if phase == "decode":
            tokens = _integer_tensor(batch.spec_info.bonus_tokens, (bs,), self.device)
            if not torch.equal(batch.spec_info.new_seq_lens, batch.seq_lens):
                raise ValueError("DSpark PP draft prefix lengths disagree")
            ranges = None
        else:
            tokens = batch.input_ids
            _integer_tensor(tokens, (tokens.numel(),), self.device)
            if len(batch.prefix_lens) != bs or len(batch.extend_lens) != bs:
                raise ValueError("invalid DSpark PP prefill ranges")
            ranges = (tuple(batch.prefix_lens), tuple(batch.extend_lens))
            if (
                any(p < 0 or n < 0 for p, n in zip(*ranges))
                or sum(batch.extend_lens) != tokens.numel()
                or tuple(p + n for p, n in zip(*ranges)) != tuple(lengths.tolist())
            ):
                raise ValueError("DSpark PP prefill ranges disagree with token lengths")
        return (
            1,
            self.sequence,
            phase,
            requests,
            tuple(lengths.tolist()),
            str(batch.seq_lens.dtype),
            _token_digest(tokens.cpu().tolist()),
            ranges,
            self.stride,
            self.vocab_size,
        )

    def _broadcast(self, payload, specs):
        def allocate():
            if self._is_owner:
                return {
                    name: _integer_tensor(payload[name], shape, self.device)
                    .to(dtype=dtype)
                    .contiguous()
                    for name, (shape, dtype) in specs.items()
                }
            return {
                name: torch.empty(shape, dtype=dtype, device=self.device)
                for name, (shape, dtype) in specs.items()
            }

        tensors = self._fence(allocate)
        for value in tensors.values():
            self.world.broadcast(value, src=self.owner)
        return tensors

    def _align_projection_state(self, batch):
        injector = self.worker._kv_injector

        def inspect():
            version = injector.weight_version
            prefixes = []
            for req, end in zip(batch.reqs, batch.seq_lens_cpu.tolist(), strict=True):
                state = req.dspark_projected_context
                valid = state is None or (
                    state.weight_version == version and 0 <= state.end <= end
                )
                prefixes.append(
                    (state.end if state is not None and valid else 0, valid)
                )
            return injector.weights_digest, prefixes

        local = self._fence(inspect)
        records = self.world.all_gather_object(local)
        if any(record[0] != local[0] for record in records):
            raise RuntimeError("DSpark PP draft weights differ across stages")

        def align():
            for index, req in enumerate(batch.reqs):
                prefixes = [record[1][index] for record in records]
                if not all(valid for _, valid in prefixes) or len(set(prefixes)) != 1:
                    # Divergent cached prefixes would enter a different number
                    # of source-KV collectives. Rebuild this request on every rank.
                    injector.invalidate_projected_context(
                        req,
                        reason="pipeline_projection_alignment",
                        new_weight_version=injector.weight_version,
                    )

        self._fence(align)

    def _sync_proposal(self, proposal, batch):
        bs = len(batch.reqs)

        def validate():
            ids = proposal.draft_block_ids
            _integer_tensor(ids, (bs, self.stride - 1), self.device)
            if not torch.equal(ids[:, 0], batch.spec_info.bonus_tokens):
                raise ValueError("DSpark PP proposal anchor differs from its request")
            tokens = proposal.draft_block.draft_tokens
            _integer_tensor(tokens, (bs, self.stride - 1), self.device)
            if bool(((tokens < 0) | (tokens >= self.vocab_size)).any()):
                raise ValueError("DSpark PP proposal token is outside the vocabulary")

        self._fence(validate)
        tokens = self._broadcast(
            {"tokens": proposal.draft_block.draft_tokens},
            {"tokens": ((bs, self.stride - 1), torch.int64)},
        )["tokens"]
        if self._is_owner:
            return proposal
        # Only the owner accepts. Other stages must not retain q logits from
        # their independently sampled local proposal after replacing its tokens.
        return DraftProposal(
            draft_block_ids=proposal.draft_block_ids,
            draft_block=DraftBlockResult(
                draft_tokens=tokens,
                corrected_logits=None,
                greedy_mask=proposal.draft_block.greedy_mask,
                temperatures=proposal.draft_block.temperatures,
            ),
            draft_hidden=proposal.draft_hidden,
            confidence=proposal.confidence,
            confidence_tap=proposal.confidence_tap,
            folded=proposal.folded,
        )

    def _forward(self, batch, step, signature):
        proxy, result = None, None
        for stage in range(self.pp.world_size):

            def forward_stage(stage=stage, proxy=proxy):
                if self.pp.rank_in_group != stage:
                    return None
                local = (
                    self.worker.forward_prefill_stage(batch, proxy)
                    if step is None
                    else self.worker.forward_decode_stage(step, proxy)
                )
                if stage == self.pp.world_size - 1:
                    if local.logits_output is None:
                        raise ValueError("final DSpark PP stage did not produce logits")
                else:
                    outgoing = (
                        local.pp_hidden_states_proxy_tensors
                        if step is None
                        else local.pp_proxy_tensors
                    )
                    if not isinstance(outgoing, PPProxyTensors):
                        raise ValueError("DSpark PP stage did not produce activations")
                return local

            local_result = self._fence(forward_stage)
            if self.pp.rank_in_group == stage:
                result = local_result
            if stage == self.pp.world_size - 1:
                continue

            def pack(stage=stage, result=result):
                if self.pp.rank_in_group != stage:
                    return None
                outgoing = (
                    result.pp_hidden_states_proxy_tensors
                    if step is None
                    else result.pp_proxy_tensors
                )
                packet = {"frame": (signature, stage)}
                for name, value in outgoing.tensors.items():
                    if not isinstance(name, str) or (
                        value is not None and not isinstance(value, torch.Tensor)
                    ):
                        raise ValueError("invalid DSpark PP activation field")
                    packet[f"activation:{name}"] = (
                        value.contiguous() if value is not None else None
                    )
                if not isinstance(packet.get("activation:hidden_states"), torch.Tensor):
                    raise TypeError("DSpark PP activation lacks hidden states")
                return packet

            packet = self._fence(pack)

            def relay(stage=stage, packet=packet):
                if self.pp.rank_in_group == stage:
                    self.pp.send_tensor_dict(packet, dst=stage + 1)
                elif self.pp.rank_in_group == stage + 1:
                    received = self.pp.recv_tensor_dict(src=stage)
                    if received.pop("frame", None) != (signature, stage):
                        raise ValueError("DSpark PP activation belongs to another step")
                    if any(not name.startswith("activation:") for name in received):
                        raise ValueError("unexpected DSpark PP activation field")
                    return PPProxyTensors(
                        {
                            name.removeprefix("activation:"): v
                            for name, v in received.items()
                        }
                    )
                return None

            incoming = self._fence(relay)
            if self.pp.rank_in_group == stage + 1:
                proxy = incoming
        return result

    def _validate_acceptance(self, accept, step):
        bs = len(step.prefix_lens)
        for name in (
            "correct_len",
            "bonus",
            "cap_trim_lens",
            "commit_lens",
            "new_seq_lens",
        ):
            _integer_tensor(getattr(accept, name), (bs,), self.device)
        _integer_tensor(accept.out_tokens, (bs, self.stride), self.device)
        lengths = accept.commit_lens.long()
        if (
            bool(((lengths < 1) | (lengths > self.stride)).any())
            or not torch.equal(accept.correct_len.long() + 1, lengths)
            or bool((accept.cap_trim_lens != 0).any())
            or not torch.equal(accept.new_seq_lens, step.prefix_lens + lengths)
            or bool((accept.new_seq_lens < 0).any())
        ):
            raise ValueError("invalid static DSpark PP acceptance lengths")
        if bool(((accept.bonus < 0) | (accept.bonus >= self.vocab_size)).any()):
            raise ValueError("DSpark PP bonus is outside the vocabulary")
        if not torch.equal(
            accept.out_tokens.gather(1, (lengths - 1)[:, None])[:, 0], accept.bonus
        ):
            raise ValueError("DSpark PP accepted block has a different bonus")
        mask = (
            torch.arange(self.stride - 1, device=self.device)[None, :]
            < (lengths - 1)[:, None]
        )
        if not torch.equal(
            accept.out_tokens[:, :-1][mask],
            step.proposal.draft_block.draft_tokens[mask],
        ):
            raise ValueError(
                "DSpark PP accepted tokens differ from the agreed proposal"
            )

    def run_batch(self, batch, *, grammar_barrier=None):
        try:
            signature = self._fence(lambda: self._signature(batch))
            signatures = self.world.all_gather_object(signature)
            if any(item != signature for item in signatures):
                raise RuntimeError("DSpark PP ranks disagree on the request/step")
            self._active = True
            if batch is None:
                self.sequence += 1
                return None
            self._align_projection_state(batch)
            prefill = signature[2] == "prefill"
            step = None
            if not prefill:
                # Complete collective prefix projection before local proposal
                # preparation. Its repeated ensure_context then has no work.
                self._fence(lambda: self.worker._kv_injector.ensure_context(batch))
                step = self._fence(lambda: self.worker.prepare_decode_step(batch))
                self._fence(lambda: self._validate_step(step))
                proposal = self._sync_proposal(step.proposal, batch)
                self._fence(lambda: self.worker.set_decode_proposal(step, proposal))
            output = self._forward(batch, step, signature)
            bs = len(batch.reqs)
            if prefill:
                tokens = self._broadcast(
                    {"tokens": output.next_token_ids},
                    {"tokens": ((bs,), torch.int64)},
                )["tokens"]
                self._fence(lambda: self._validate_prefill_tokens(tokens))
                result = self._fence(
                    lambda: self.worker.commit_prefill_stage(
                        batch, output, next_token_ids=tokens
                    )
                )
            else:
                accept = self._fence(
                    lambda: (
                        self.worker.accept_decode_step(
                            step, grammar_barrier=grammar_barrier
                        )
                        if self._is_owner
                        else None
                    )
                )
                self._fence(
                    lambda: (
                        self._validate_acceptance(accept, step)
                        if self._is_owner
                        else None
                    )
                )
                specs = {
                    name: ((bs,), torch.int32)
                    for name in ("correct_len", "cap_trim_lens", "commit_lens")
                }
                specs.update(
                    bonus=((bs,), torch.int64),
                    new_seq_lens=((bs,), step.prefix_lens.dtype),
                    out_tokens=((bs, self.stride), torch.int64),
                )
                payload = (
                    {name: getattr(accept, name) for name in specs}
                    if self._is_owner
                    else None
                )
                remote = AcceptOuts(**self._broadcast(payload, specs))
                self._fence(lambda: self._validate_acceptance(remote, step))
                result = self._fence(
                    lambda: self.worker.commit_decode_step(
                        step, acceptance=accept if self._is_owner else remote
                    )
                )
            self.sequence += 1
            return result
        except Exception:
            self.failed = True
            raise
        finally:
            self._active = False

    @staticmethod
    def _validate_step(step):
        if step.run_compact or step.layout is not None or step.confidence is not None:
            raise ValueError("DSpark PP coordinator requires full static verification")

    def _validate_prefill_tokens(self, tokens):
        if bool(((tokens < 0) | (tokens >= self.vocab_size)).any()):
            raise ValueError("DSpark PP prefill sample is outside the vocabulary")
