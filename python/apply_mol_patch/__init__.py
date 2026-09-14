"""Apply mol-patch monkey-patches to SGLang 0.5.13 at startup."""
import logging, re
_logger = logging.getLogger("mol-patch")

def apply():
    """Apply all patches. Called after sglang is fully importable."""
    _patch_kv_reuse()
    _patch_scheduler()
    _patch_scheduler_hooks()
    _patch_lora_layers()
    _patch_tokenizer_control()
    _patch_serving()
    _patch_http_routes()
    _maybe_install_free_debug()
    _logger.info("mol-patch: all patches applied")


def _maybe_install_free_debug():
    """Debug-only: when MOL_FREE_DEBUG=1, wrap the paged allocator's free() to
    detect when a page is freed while already free (double-free). Logs the
    offending page ids and a short stack so we can pin the exact mechanism in
    ONE run. No-op unless the env flag is set."""
    import os
    if os.environ.get("MOL_FREE_DEBUG", "") not in ("1", "true", "True"):
        return
    try:
        import torch, traceback
        from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator as _P
    except Exception as _e:
        _logger.warning("mol-patch: free-debug not installed: %s", _e)
        return
    _orig_free = _P.free
    def free(self, free_index):
        try:
            if free_index is not None and free_index.numel() > 0 and self.is_not_in_free_group:
                pages = torch.unique(free_index // self.page_size)
                cur = self.free_pages
                rel = getattr(self, "release_pages", None)
                already = set()
                for p in pages.tolist():
                    if (cur.numel() and (cur == p).any().item()) or (
                        rel is not None and rel.numel() and (rel == p).any().item()):
                        already.add(p)
                if already:
                    _logger.error("mol-patch FREE-DEBUG: DOUBLE-FREE pages=%s "
                                  "(freeing %d slots) stack:\n%s",
                                  sorted(already), int(free_index.numel()),
                                  "".join(traceback.format_stack(limit=8)))
        except Exception:
            pass
        return _orig_free(self, free_index)
    _P.free = free
    _logger.info("mol-patch: free-debug installed (MOL_FREE_DEBUG=1)")


def _patch_lora_layers():
    """Install a delegating __getattr__ on BaseLayerWithLoRA.

    GLM/DeepSeek MoE code reads attributes (e.g. moe_runner_config) directly off
    mlp.experts, which becomes FusedMoEWithLoRA after wrapping. Base 0.5.13 has no
    __getattr__ on BaseLayerWithLoRA, so delegate misses to the wrapped
    base_layer. weight/bias are excluded so nn.Module.register_parameter's
    hasattr() probe during init doesn't break.
    """
    import torch.nn as nn
    from sglang.srt.lora.layers import BaseLayerWithLoRA

    if getattr(BaseLayerWithLoRA, "_mol_getattr_installed", False):
        return

    def _delegating_getattr(self, name):
        try:
            return nn.Module.__getattr__(self, name)
        except AttributeError as original_error:
            if name in {"weight", "bias"}:
                raise original_error
            modules = self.__dict__.get("_modules")
            base_layer = modules.get("base_layer") if modules is not None else None
            if base_layer is not None:
                try:
                    return getattr(base_layer, name)
                except AttributeError:
                    pass
            raise original_error

    BaseLayerWithLoRA.__getattr__ = _delegating_getattr
    BaseLayerWithLoRA._mol_getattr_installed = True
    _logger.info("mol-patch: lora layers delegation shim installed")


def _patch_scheduler_hooks():
    """Wire the ported route methods into the base request lifecycle.

    B: process_batch_result -> _maybe_route_lora_after_prefill / _after_decode
    G: _add_request_to_queue -> _maybe_mark_route_decode_request (idempotent)
    Both base methods funnel all generate paths, so wrapping them covers
    session and non-session requests without copying their large bodies.
    """
    from sglang.srt.managers.scheduler import Scheduler

    # --- B: post-call hook on process_batch_result ---
    _orig_pbr = Scheduler.process_batch_result
    def process_batch_result(self, batch, result, *a, **kw):
        out = _orig_pbr(self, batch, result, *a, **kw)
        try:
            if self.lora_router_mode != "token_interval" or self.lora_router_pool:
                fm = batch.forward_mode
                if fm.is_extend():
                    self._maybe_route_lora_after_prefill(batch)
                elif fm.is_decode():
                    self._maybe_route_lora_after_decode(batch)
        except Exception:
            _logger.exception("mol-patch: route hook in process_batch_result failed")
        return out
    Scheduler.process_batch_result = process_batch_result

    # --- G: idempotent mark on _add_request_to_queue ---
    _orig_aq = Scheduler._add_request_to_queue
    def _add_request_to_queue(self, req, is_retracted: bool = False):
        if not is_retracted:
            try:
                self._maybe_mark_route_decode_request(req)
            except Exception:
                _logger.exception("mol-patch: _maybe_mark_route_decode_request failed")
        return _orig_aq(self, req, is_retracted)
    Scheduler._add_request_to_queue = _add_request_to_queue

    # --- H: ensure router-pool adapters count as "already loaded" when scheduling.
    # OLD did this by adding lora_router_pool to running_loras inside
    # _get_new_batch_prefill_raw. The base passes running_loras into
    # _can_schedule_lora_req, so we inject the pool there instead of overriding
    # the large prefill method. No-op for route_decode/route_decode (matching
    # OLD: those modes re-prefill on the target adapter, so the pool must NOT be
    # pre-seeded).
    _orig_cslr = Scheduler._can_schedule_lora_req
    def _can_schedule_lora_req(self, req, running_loras):
        try:
            if (self._lora_router_enabled()
                    and self.lora_router_mode != "route_decode"
                    and self.lora_router_pool):
                running_loras = set(running_loras) | set(self.lora_router_pool)
        except Exception:
            _logger.exception("mol-patch: _can_schedule_lora_req pool-inject failed")
        return _orig_cslr(self, req, running_loras)
    Scheduler._can_schedule_lora_req = _can_schedule_lora_req

    # NOTE on overlap scheduling: route_decode trims req_to_token / kv_allocated_len
    # mid-flight. Under the overlap scheduler the in-flight (next) batch is built
    # and its result processed against the pre-trim slot layout, re-freeing pages
    # the trim already freed -> pool double-free -> invariant_checker kills the
    # scheduler (confirmed via MOL_FREE_DEBUG). A per-batch overlap toggle is NOT
    # viable here: flipping is_disable_overlap_for_batch desyncs the overlap loop's
    # result_queue (IndexError: pop from empty deque). So route_decode REQUIRES the
    # server to run with --disable-overlap-schedule. The launcher sets this by
    # default for KV-reuse, and configure_lora_router rejects route_decode modes
    # when overlap is still enabled (see _patch_scheduler.configure_lora_router).

    _logger.info("mol-patch: scheduler lifecycle hooks installed")


def _patch_kv_reuse():
    """Replace mem_cache.common.release_kv_cache with a route_decode-aware version.

    When a request switched LoRA mid-flight (route_decode), it carries
    req.lora_router_stale_kv_end; the trimmed/stale KV range must be freed
    directly instead of going through the normal cache_finished_req path.

    Strategy: [OLD stale_kv branch] + [base 0.5.13 tail copied verbatim].
    The tail and early-return are taken from the running base module so any
    base-side fixes / renames (e.g. mamba_pool -> mamba_allocator) are kept.
    """
    import sglang.srt.mem_cache.common as _common
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang.srt.server_args import get_global_server_args
    from sglang.srt.utils.common import ceil_align
    _log = _common.logger

    def release_kv_cache(req, tree_cache, is_insert: bool = True):
        # --- base early-return (verbatim from 0.5.13) ---
        if req.req_pool_idx is None:
            assert (
                tree_cache.supports_mamba()
            ), "Only MambaRadixCache allow freeing before alloc"
            if req.mamba_pool_idx is not None:
                tree_cache.req_to_token_pool.mamba_allocator.free(
                    req.mamba_pool_idx.unsqueeze(-1)
                )
                req.mamba_pool_idx = None
            return

        # --- mol-patch: route_decode stale-KV direct release ---
        stale_kv_end = int(getattr(req, "lora_router_stale_kv_end", 0) or 0)
        if stale_kv_end > 0:
            release_end = max(stale_kv_end, req.kv_allocated_len, req.seqlen)
            indices_to_free = tree_cache.req_to_token_pool.req_to_token[
                req.req_pool_idx
            ][req.kv_allocated_len:release_end]
            tree_cache.token_to_kv_pool_allocator.free(indices_to_free)
            _log.info(
                "lora_router route_decode directly released request kv request_id=%s "
                "seqlen=%d kv_allocated_len=%d release_end=%d released=%d",
                req.rid, req.seqlen, req.kv_allocated_len, release_end,
                int(indices_to_free.numel()),
            )
            req.lora_router_stale_kv_end = 0
            tree_cache.cache_finished_req(
                req,
                is_insert=is_insert
                and not getattr(req, "skip_radix_cache_insert", False),
            )
            if req.req_pool_idx is None:
                return
            if req.last_node is not None and hasattr(tree_cache, "dec_lock_ref"):
                tree_cache.dec_lock_ref(req.last_node)
            if isinstance(tree_cache.req_to_token_pool, HybridReqToTokenPool) and (
                not tree_cache.supports_mamba()
            ):
                assert req.mamba_pool_idx is not None, (
                    "mamba state is freed while the tree cache does not manage "
                    "mamba states"
                )
                tree_cache.req_to_token_pool.free_mamba_cache(req)
            tree_cache.req_to_token_pool.free(req)
            return

        # --- base tail (verbatim from 0.5.13) ---
        tree_cache.cache_finished_req(
            req,
            is_insert=is_insert and not getattr(req, "skip_radix_cache_insert", False),
        )
        if req.req_pool_idx is None:
            return
        start_p, end_p = req.pop_overallocated_kv_cache()
        global_server_args = get_global_server_args()
        page_size = global_server_args.page_size
        spec_algo = global_server_args.speculative_algorithm
        if spec_algo is None and not global_server_args.strip_thinking_cache:
            assert start_p == end_p, (
                f"Unexpected overallocated KV cache, {req.kv_committed_len=}, "
                f"{req.kv_allocated_len=}"
            )
        if page_size > 1:
            start_p = ceil_align(start_p, page_size)
        if start_p < end_p:
            indices_to_free = tree_cache.req_to_token_pool.req_to_token[
                req.req_pool_idx
            ][start_p:end_p]
            tree_cache.token_to_kv_pool_allocator.free(indices_to_free)
        if isinstance(tree_cache.req_to_token_pool, HybridReqToTokenPool) and (
            not tree_cache.supports_mamba()
        ):
            assert req.mamba_pool_idx is not None, (
                "mamba state is freed while the tree cache does not manage mamba states"
            )
            tree_cache.req_to_token_pool.free_mamba_cache(req)
        tree_cache.req_to_token_pool.free(req)

    _common.release_kv_cache = release_kv_cache
    # scheduler.py did `from ...common import release_kv_cache`, so rebind there too.
    try:
        import sglang.srt.managers.scheduler as _sched
        if hasattr(_sched, "release_kv_cache"):
            _sched.release_kv_cache = release_kv_cache
    except Exception as _e:
        _logger.warning("mol-patch: could not rebind scheduler.release_kv_cache: %s", _e)
    _logger.info("mol-patch: kv-reuse (release_kv_cache) patched")


def _patch_scheduler():
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.io_struct import (
        ConfigureLoRARouterReqInput, SwitchLoRAAdapterReqInput,
    )
    _orig_init = Scheduler.__init__
    def _new_init(self, *_a, **_kw):
        _orig_init(self, *_a, **_kw)
        # Router state consumed by the ported route_decode methods.
        self.lora_router_pool = []          # list of lora_ids (adapter ids), order matters
        self.lora_router_names = []         # parallel list of public adapter names
        self.lora_router_switch_every_n_tokens = 0
        self.lora_router_mode = "token_interval"
        self.lora_router_seed = 0
        # Register our request types into the scheduler's TypeBasedDispatcher.
        # The dispatcher is built once in _orig_init from a hardcoded list, so a
        # ConfigureLoRARouterReqInput reaching the scheduler would otherwise raise
        # "Invalid object" and crash the whole scheduler loop.
        disp = getattr(self, "_request_dispatcher", None)
        if disp is not None and hasattr(disp, "_mapping"):
            disp._mapping[ConfigureLoRARouterReqInput] = self.configure_lora_router
            disp._mapping[SwitchLoRAAdapterReqInput] = self._handle_switch_lora_adapter
            if hasattr(disp, "_mro_cache"):
                disp._mro_cache.clear()
    Scheduler.__init__ = _new_init

    # Install the ported route_decode / KV-reuse methods onto Scheduler.
    from . import _route_methods
    installed = _route_methods.install(Scheduler)
    _logger.info("mol-patch: installed %d route methods: %s",
                 len(installed), ",".join(installed))

    def configure_lora_router(self, recv_req):
        """Scheduler-side handler: validate and store router config.

        Mirrors the OLD overlay's Scheduler.configure_lora_router. lora_ids are
        already resolved on recv_req by the tokenizer-side configure_lora_router.
        """
        from sglang.srt.managers.io_struct import ConfigureLoRARouterReqOutput
        try:
            lora_ids = list(recv_req.lora_ids or [])
            disable_modes = ("prefill_decode", "prefill_decode_ba_interval",
                             "route_decode")
            if recv_req.switch_every_n_tokens <= 0 and recv_req.mode not in disable_modes:
                self.lora_router_pool = []
                self.lora_router_names = []
                self.lora_router_switch_every_n_tokens = 0
                self.lora_router_mode = "token_interval"
                self.lora_router_seed = getattr(recv_req, "seed", 0)
                return ConfigureLoRARouterReqOutput(success=True, message="LoRA router disabled.")

            if not getattr(self, "enable_lora", False):
                raise ValueError("LoRA is not enabled.")
            valid_modes = ("token_interval",) + disable_modes
            if recv_req.mode not in valid_modes:
                raise ValueError(f"Unsupported LoRA router mode: {recv_req.mode}")
            if recv_req.mode in ("prefill_decode", "prefill_decode_ba_interval") and len(lora_ids) != 2:
                raise ValueError(f"{recv_req.mode} requires exactly two LoRA adapters")
            if recv_req.mode == "route_decode" and len(lora_ids) < 1:
                raise ValueError(f"{recv_req.mode} requires at least one LoRA adapter")
            if recv_req.mode == "token_interval" and len(lora_ids) < 1:
                raise ValueError("token_interval requires at least one LoRA adapter")

            # route_decode trims KV mid-flight and is unsafe under the overlap
            # scheduler (in-flight batch frees pages the trim already freed ->
            # pool double-free). Reject up-front rather than crash the scheduler.
            if recv_req.mode == "route_decode" and getattr(
                    self, "enable_overlap", False):
                raise ValueError(
                    f"{recv_req.mode} requires the server to run with "
                    "--disable-overlap-schedule (KV-reuse trim is unsafe under "
                    "the overlap scheduler)")

            # Best-effort capacity validation (guarded: internal APIs may drift).
            try:
                lm = self.tp_worker.model_runner.lora_manager
                if recv_req.mode == "route_decode":
                    bad = [lid for lid in dict.fromkeys(lora_ids)
                           if not lm.validate_lora_batch({lid, None})]
                    if bad:
                        raise ValueError(
                            "One or more route_decode LoRA adapters cannot fit with the "
                            f"base model in the LoRA memory pool: {bad}")
                else:
                    unique = set(lora_ids + [None])
                    if len(unique) > self.max_loras_per_batch:
                        raise ValueError(
                            "Configured LoRA router pool exceeds max_loras_per_batch "
                            f"including base: pool={len(set(lora_ids))} "
                            f"max_loras_per_batch={self.max_loras_per_batch}")
                    if not lm.validate_lora_batch(unique):
                        raise ValueError("LoRA router pool cannot fit in the LoRA memory pool")
            except AttributeError:
                pass  # validate API not available; skip capacity check

            self.lora_router_pool = lora_ids
            self.lora_router_names = list(recv_req.lora_pool)
            self.lora_router_switch_every_n_tokens = recv_req.switch_every_n_tokens
            self.lora_router_mode = recv_req.mode
            self.lora_router_seed = getattr(recv_req, "seed", 0)
            _logger.info("mol-patch: configured LoRA router mode=%s names=%s ids=%s n=%d",
                         self.lora_router_mode, self.lora_router_names,
                         self.lora_router_pool, self.lora_router_switch_every_n_tokens)
            return ConfigureLoRARouterReqOutput(
                success=True, message="LoRA router configured.",
                mode=self.lora_router_mode, lora_pool=self.lora_router_names,
                switch_every_n_tokens=self.lora_router_switch_every_n_tokens)
        except ValueError as e:
            # Expected client/config validation errors -> return cleanly, no traceback.
            _logger.info("mol-patch: LoRA router config rejected: %s", e)
            return ConfigureLoRARouterReqOutput(success=False, message=str(e))
        except Exception as e:
            _logger.exception("mol-patch: failed to configure LoRA router")
            return ConfigureLoRARouterReqOutput(success=False, message=str(e))
    Scheduler.configure_lora_router = configure_lora_router

    def _handle_switch_lora_adapter(self, recv_req):
        from sglang.srt.managers.io_struct import SwitchLoRAAdapterReqOutput
        for req in self.running_batch.reqs if self.running_batch else []:
            if getattr(req, "rid", "") == recv_req.request_id:
                req.lora_id = recv_req.lora_id
                return SwitchLoRAAdapterReqOutput(success=True, message="OK")
        return SwitchLoRAAdapterReqOutput(success=False, message="Not found")
    Scheduler._handle_switch_lora_adapter = _handle_switch_lora_adapter

    _logger.info("mol-patch: scheduler patched")


def _patch_tokenizer_control():
    from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
    from sglang.srt.managers.communicator import FanOutCommunicator
    
    BASE = "base_model"

    async def _configure_lora_router(self, obj, _=None):
        self.auto_create_handle_loop()
        from sglang.srt.managers.io_struct import ConfigureLoRARouterReqOutput
        if not self.server_args.enable_lora:
            return ConfigureLoRARouterReqOutput(success=False, message="LoRA not enabled")
        prev_ids = list(getattr(self, "_lora_router_hold_ids", []))
        acq = []
        try:
            mode = getattr(obj, "mode", "")
            sw = getattr(obj, "switch_every_n_tokens", 0)
            valid = ("token_interval", "prefill_decode", "route_decode")
            if sw <= 0 and mode not in ("prefill_decode", "route_decode"):
                obj.lora_pool = []; obj.lora_ids = []
            elif mode in valid and obj.lora_pool:
                ids = []
                for name in obj.lora_pool:
                    lid = await self.lora_registry.acquire(name) if name != BASE else None
                    ids.append(lid)
                    if lid: acq.append(lid)
                obj.lora_ids = ids
            else:
                raise ValueError("Bad mode: " + str(mode))
            comm = getattr(self, "configure_lora_router_communicator", None)
            results = await comm(obj) if comm else [(False, "No comm")]
            ok, msg = FanOutCommunicator.merge_results(results)
            if ok:
                if prev_ids: await self.lora_registry.release(prev_ids)
                self._lora_router_hold_ids = [x for x in (obj.lora_ids or []) if x is not None]
                return ConfigureLoRARouterReqOutput(success=True, message=msg, mode=mode,
                    lora_pool=list(obj.lora_pool), switch_every_n_tokens=sw)
            if acq: await self.lora_registry.release(acq)
            return ConfigureLoRARouterReqOutput(success=False, message=msg)
        except Exception as e:
            if acq: await self.lora_registry.release(acq)
            return ConfigureLoRARouterReqOutput(success=False, message=str(e))

    async def _switch_lora_adapter(self, obj, _=None):
        self.auto_create_handle_loop()
        from sglang.srt.managers.io_struct import SwitchLoRAAdapterReqOutput
        if not self.server_args.enable_lora:
            return SwitchLoRAAdapterReqOutput(success=False, message="LoRA not enabled")
        try:
            obj.lora_id = await self.lora_registry.acquire(obj.lora_name)
            comm = getattr(self, "switch_lora_adapter_communicator", None)
            results = await comm(obj) if comm else [(False, "No comm")]
            ok, msg = FanOutCommunicator.merge_results(results)
            if not ok and obj.lora_id: await self.lora_registry.release(obj.lora_id)
            return SwitchLoRAAdapterReqOutput(success=ok, message=msg)
        except Exception as e:
            lid = getattr(obj, "lora_id", None)
            if lid: await self.lora_registry.release(lid)
            return SwitchLoRAAdapterReqOutput(success=False, message=str(e))

    setattr(TokenizerControlMixin, "configure_lora_router", _configure_lora_router)
    setattr(TokenizerControlMixin, "switch_lora_adapter", _switch_lora_adapter)

    try:
        from sglang.srt.managers.io_struct import ConfigureLoRARouterReqOutput, SwitchLoRAAdapterReqOutput
        import sglang.srt.managers.tokenizer_control_mixin as tcm
        specs = list(getattr(tcm, "_COMMUNICATOR_SPECS", []))
        exist = set(s[0] for s in specs)
        if "configure_lora_router" not in exist:
            specs.append(("configure_lora_router", ConfigureLoRARouterReqOutput))
        if "switch_lora_adapter" not in exist:
            specs.append(("switch_lora_adapter", SwitchLoRAAdapterReqOutput))
        tcm._COMMUNICATOR_SPECS = specs
    except Exception:
        pass
    _logger.info("mol-patch: tokenizer_control patched")


def _patch_serving():
    """Wrap OpenAIServingCompletion for MoL:
    D: map public model_id / custom_params['lora_adapter'] -> internal adapter
       before the base resolves lora_path.
    E: fold meta_info['lora_router_*'] into the completion response metadata.

    The public->internal model-id table is configurable via the env var
    MOL_MODEL_ID_MAP (JSON: {"public_id": "internal_adapter_or_empty"}). The
    special router id maps to MOL_ENTRY_ADAPTER. If unset, mapping is a no-op
    passthrough so arbitrary adapter names still work.
    """
    import os, json
    from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
    from . import _mol_routing

    router_id = os.environ.get(
        "MOL_ROUTER_MODEL_ID", "mindlab-research/Macaron-V1-Preview-749B")
    entry_adapter = os.environ.get("MOL_ENTRY_ADAPTER", "l0_chat")
    router_aliases = {router_id, "Macaron-V1-Preview"}  # legacy short id still routes
    auto_route = os.environ.get("MOL_AUTO_ROUTE", "1") not in ("0", "false", "False")
    router_max_tokens = int(os.environ.get("MOL_ROUTER_MAX_TOKENS", "16") or 16)
    decode_tokens = int(os.environ.get("MOL_DECODE_TOKENS", "32") or 32)
    # Global override for the route_decode KV-reuse flag (A/B garble debug).
    # When "0", auto-injected route_decode params set enable_kv_reuse=False so the
    # specialist re-prefills instead of reusing the entry-adapter prefix KV.
    kv_reuse_default = os.environ.get("MOL_KV_REUSE", "1") not in ("0", "false", "False")
    try:
        id_map = json.loads(os.environ.get("MOL_MODEL_ID_MAP", "") or "{}")
        if not isinstance(id_map, dict):
            id_map = {}
    except Exception:
        _logger.warning("mol-patch: MOL_MODEL_ID_MAP is not valid JSON; ignoring")
        id_map = {}

    # Lazy, cached routing state (library parse + route map). Built on first
    # router request so a malformed library doesn't break plain LoRA serving.
    _routing_state = {"tasks": None, "route_to_adapter": None, "failed": False}

    def _ensure_routing_state():
        if _routing_state["tasks"] is not None or _routing_state["failed"]:
            return _routing_state["tasks"] is not None
        try:
            tasks = _mol_routing.load_lora_library()
            _routing_state["tasks"] = tasks
            _routing_state["route_to_adapter"] = _mol_routing.build_route_to_adapter(tasks)
            _logger.info(
                "mol-patch: routing library loaded; routes=%s map=%s",
                list(tasks), _routing_state["route_to_adapter"])
            return True
        except Exception:
            _routing_state["failed"] = True
            _logger.exception("mol-patch: failed to load routing library; auto-route disabled")
            return False

    def _effective_internal_model(public_model_id):
        if public_model_id in router_aliases:
            return entry_adapter
        return id_map.get(public_model_id, public_model_id)

    # --- D: request mapping (wrap) ---
    _orig_convert = OpenAIServingCompletion._convert_to_internal_request
    def _convert_to_internal_request(self, request, raw_request=None):
        try:
            custom_params = request.custom_params if isinstance(
                request.custom_params, dict) else {}
            explicit = custom_params.get("lora_adapter")
            if explicit is not None and not isinstance(explicit, str):
                raise ValueError("Invalid lora_adapter: expected a string model_id.")
            public_model_id = request.model or router_id
            if public_model_id in ("", "default"):
                public_model_id = router_id
            if explicit is not None:
                public_model_id = explicit
            request.model = public_model_id

            # --- auto-route: router model_id + no client-supplied lora_router ---
            # Synthesize the route_decode KV-reuse params + router prompt so a
            # bare `model: <router_id>` call routes L0 -> L0..L4 each request.
            is_router = public_model_id in router_aliases
            already_routing = isinstance(custom_params.get("lora_router"), dict)
            if (is_router and auto_route and not already_routing
                    and isinstance(request.prompt, str) and request.prompt
                    and _ensure_routing_state()):
                _inject_route_decode(self, request, custom_params)

            effective = _effective_internal_model(public_model_id)
            # Rewrite request.model to the effective adapter so the base
            # _resolve_lora_path picks the right LoRA (entry adapter for the
            # router id), then prefill starts on L0 before the router decides.
            if effective != public_model_id:
                request.model = effective
        except Exception:
            _logger.exception("mol-patch: model-id mapping failed; passing through")
        return _orig_convert(self, request, raw_request)
    OpenAIServingCompletion._convert_to_internal_request = _convert_to_internal_request

    def _inject_route_decode(self, request, custom_params):
        """Mutate request in place: prepend router instruction, set lora_router
        params, ensure enough max_tokens for router + specialist decode."""
        tasks = _routing_state["tasks"]
        route_to_adapter = _routing_state["route_to_adapter"]
        user_text = request.prompt
        # keep_prefix = tokens of the reusable "User request:\n...\n\nAnswer:" block.
        prefix = _mol_routing.answer_prompt(user_text)
        try:
            tok = self.tokenizer_manager.tokenizer
            keep_prefix = len(tok.encode(prefix, add_special_tokens=False))
        except Exception:
            # Fallback: rough word estimate; never block routing on tokenizer.
            keep_prefix = max(1, len(prefix.split()))
        request.prompt = _mol_routing.build_router_prompt(user_text, tasks)
        lora_router = _mol_routing.build_lora_router_params(
            user_text, tasks,
            keep_prefix_token_count=keep_prefix,
            route_to_adapter=route_to_adapter,
            router_max_tokens=router_max_tokens,
            decode_tokens=decode_tokens,
            enable_kv_reuse=kv_reuse_default,
        )
        merged = dict(custom_params)
        # DIAG: force every route_decode request to a fixed specialist so all
        # concurrent reqs route to the SAME adapter (isolates "different adapters
        # co-resident in one batch" vs "mid-decode switch itself").
        _force_route = os.environ.get("MOL_ROUTE_FORCE")
        if _force_route:
            lora_router["deterministic_route"] = _force_route
        merged["lora_router"] = lora_router
        request.custom_params = merged
        # The router must actually emit the `model_id=Lx` token before EOS, else
        # L0 stops on its first (EOS) token and route_decode never parses a route.
        # ignore_eos forces generation past EOS so the router token is produced;
        # route_decode then trims and hands off to the specialist.
        request.ignore_eos = True
        # Guarantee headroom: router emits up to router_max_tokens before the
        # specialist decode begins; clients sizing only for the answer would
        # otherwise truncate the router token.
        need = router_max_tokens + decode_tokens
        if request.max_tokens is None or request.max_tokens < need:
            request.max_tokens = max(int(request.max_tokens or 0), need)

    # --- E: response metadata (wrap) ---
    _orig_build = OpenAIServingCompletion._build_completion_response
    def _build_completion_response(self, request, ret, created, *a, **kw):
        resp = _orig_build(self, request, ret, created, *a, **kw)
        try:
            meta_info = ret[0].get("meta_info", {}) if ret else {}
            extra = {}
            for key, value in meta_info.items():
                if not key.startswith("lora_router_"):
                    continue
                if isinstance(value, list):
                    value = value[-1] if value else None
                extra[key] = value
            if extra:
                md = dict(resp.metadata or {})
                md.update(extra)
                resp.metadata = md
        except Exception:
            _logger.exception("mol-patch: response metadata fold failed")
        return resp
    OpenAIServingCompletion._build_completion_response = _build_completion_response

    _logger.info("mol-patch: serving_completions patched (model-map + metadata)")


def _patch_http_routes():
    from sglang.srt.entrypoints.http_server import app
    from pydantic import BaseModel, Field
    from typing import List

    class CnfgLoraBody(BaseModel):
        lora_pool: List[str] = Field(default_factory=list)
        switch_every_n_tokens: int = 0
        seed: int = 0
        mode: str = "token_interval"

    class SwitchLoraBody(BaseModel):
        request_id: str = ""
        lora_name: str = ""

    app.router.routes = [r for r in app.router.routes
        if not (hasattr(r, "path") and r.path in ("/v1/configure_lora_router", "/v1/switch_lora_adapter"))]

    @app.post("/v1/configure_lora_router")
    async def _cnfg(body: CnfgLoraBody):
        from sglang.srt.managers.io_struct import ConfigureLoRARouterReqInput
        from sglang.srt.entrypoints.http_server import get_global_state
        sw = body.switch_every_n_tokens
        obj = ConfigureLoRARouterReqInput(
            lora_pool=body.lora_pool, switch_every_n_tokens=sw,
            seed=body.seed, mode=body.mode)
        tm = get_global_state().tokenizer_manager
        return await tm.configure_lora_router(obj)

    @app.post("/v1/switch_lora_adapter")
    async def _sw(body: SwitchLoraBody):
        from sglang.srt.managers.io_struct import SwitchLoRAAdapterReqInput
        from sglang.srt.entrypoints.http_server import get_global_state
        obj = SwitchLoRAAdapterReqInput(request_id=body.request_id, lora_name=body.lora_name)
        tm = get_global_state().tokenizer_manager
        return await tm.switch_lora_adapter(obj)

    # --- advertise the public router model id in /v1/models ---
    import os as _os
    router_id = _os.environ.get(
        "MOL_ROUTER_MODEL_ID", "mindlab-research/Macaron-V1-Preview-749B")
    app.router.routes = [r for r in app.router.routes
        if not (hasattr(r, "path") and r.path == "/v1/models"
                and "GET" in getattr(r, "methods", set()))]

    @app.get("/v1/models")
    async def _models():
        from sglang.srt.entrypoints.http_server import get_global_state
        from sglang.srt.entrypoints.openai.protocol import ModelCard, ModelList
        tm = get_global_state().tokenizer_manager
        served = tm.served_model_name
        cards = [ModelCard(id=router_id, root=served, parent=served,
                           max_model_len=tm.model_config.context_len)]
        cards.append(ModelCard(id=served, root=served,
                               max_model_len=tm.model_config.context_len))
        if tm.server_args.enable_lora:
            for _, ref in tm.lora_registry.get_all_adapters().items():
                cards.append(ModelCard(id=ref.lora_name, root=ref.lora_path,
                                        parent=served, max_model_len=None))
        return ModelList(data=cards)

    _logger.info("mol-patch: HTTP routes registered")
