#!/usr/bin/env python3
"""Portable, correctness-only GPU smoke for the opt-in fused decode plan.

Run this file directly; no fixture checkout or compact-opt package is required.
A default-resolvable ``deep_gemm`` is required for module-isolation checks.
It may come from the environment or a process-local PYTHONPATH pointing at an
unmodified package. Its resolved path and binaries are recorded. This script
does not install packages or replace that binding, and fails if its old API is
incompatible rather than silently skipping the comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
TOPK = 2048


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class Fingerprints:
    def __init__(self):
        self.files = {}

    def add(self, filename):
        path = Path(filename).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if str(path) in self.files:
            require(self.files[str(path)]['sha256'] == digest, f'file changed: {path}')
        self.files[str(path)] = dict(path=str(path), sha256=digest, bytes=path.stat().st_size)

    def manifest(self, directory):
        root = Path(directory).resolve()
        self.add(root / 'manifest.json')
        data = json.loads((root / 'manifest.json').read_text())
        if 'staged_files' in data:
            for name in data['staged_files']:
                self.add(root / name)
            self.add(root / data['library'])
        else:
            for route in ('small', 'large'):
                for field in ('source', 'library'):
                    self.add(root / data[route][field])
        return data

    def modules(self, prefix):
        for name, module in tuple(sys.modules.items()):
            if name == prefix or name.startswith(prefix + '.'):
                filename = getattr(module, '__file__', None)
                if filename and Path(filename).is_file():
                    self.add(filename)

    def check(self):
        changed = []
        for name, expected in self.files.items():
            path = Path(name)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
                changed.append(name)
        return dict(ok=not changed, files_checked=len(self.files), changed_or_missing=changed)


def load_api():
    # Direct-file use also works before installing SGLang and its large dependency set.
    name = '_litetopk_fused_smoke_api'
    spec = importlib.util.spec_from_file_location(name, HERE / 'fused.py')
    require(spec is not None and spec.loader is not None, 'cannot load sibling fused.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DefaultBinding:
    def __init__(self):
        try:
            self.module = importlib.import_module('deep_gemm')
            self.scorer = self.module.fp8_paged_mqa_logits
            self.metadata = self.module.get_paged_mqa_logits_metadata
            self.sms = self.module.get_num_sms
        except Exception as error:
            raise RuntimeError('A default-resolvable deep_gemm with the old paged scorer and metadata API is required; '
                               'the module-isolation test is not skipped.') from error
        self.modules = {name: obj for name, obj in sys.modules.items()
                        if name == 'deep_gemm' or name.startswith('deep_gemm.')}

    def check(self):
        require(sys.modules.get('deep_gemm') is self.module, 'default deep_gemm module was replaced')
        for name, module in self.modules.items():
            require(sys.modules.get(name) is module, f'default module identity changed: {name}')
        for name, expected in [('fp8_paged_mqa_logits', self.scorer),
                               ('get_paged_mqa_logits_metadata', self.metadata), ('get_num_sms', self.sms)]:
            require(getattr(self.module, name) is expected, f'default function was replaced: {name}')

    def record(self):
        return dict(module=self.module.__name__, file=getattr(self.module, '__file__', None),
                    module_identity=id(self.module), old_scorer_identity=id(self.scorer),
                    metadata_identity=id(self.metadata), num_sms_identity=id(self.sms),
                    module_identities={name: id(obj) for name, obj in self.modules.items()})


class Inputs:
    def __init__(self, torch, batch, length, maximum, device, seed):
        self.torch, self.batch, self.length, self.maximum = torch, batch, length, maximum
        self.device, self.seed = device, seed
        self.logical_pages = (length + 63) // 64
        self.physical_pages = self.logical_pages + 19
        self.q = torch.empty((batch, 1, 32, 128), device=device, dtype=torch.float8_e4m3fn)
        self.weights = torch.empty((batch, 32), device=device, dtype=torch.float32)
        self.cache = torch.empty((self.physical_pages, 64, 1, 132), device=device, dtype=torch.uint8)
        self.table = torch.empty((batch, (maximum + 63) // 64), device=device, dtype=torch.int32)
        self.lengths = torch.empty(batch, device=device, dtype=torch.int32)
        self.live = []

    def set(self, pattern, variant):
        t = self.torch
        gen = t.Generator(device=self.device).manual_seed(self.seed + self.batch * 1009 + variant * 7919)

        def finite_fp8(shape):
            magnitude = t.randint(0, 127, shape, device=self.device, dtype=t.uint8, generator=gen)
            sign = t.randint(0, 2, shape, device=self.device, dtype=t.uint8, generator=gen) * 128
            return magnitude | sign  # Excludes both FP8 NaN encodings.

        self.q.view(t.uint8).copy_(finite_fp8(self.q.shape))
        self.weights.copy_(t.randn(self.weights.shape, device=self.device, generator=gen) * .05)
        self.weights[:, 0] = -self.weights[:, 0].abs() - .01
        packed = self.cache.view(self.physical_pages, -1)
        packed[:, :64 * 128].copy_(finite_fp8((self.physical_pages, 64 * 128)))
        scales = t.exp2(t.randint(-12, -5, (self.physical_pages, 64), device=self.device, generator=gen).float())
        packed[:, 64 * 128:].copy_(scales.view(t.uint8))
        self.table.fill_(-1)  # Unused logical pages are deliberately invalid.
        for row in range(self.batch):
            pages = t.randperm(self.physical_pages, device=self.device, generator=gen)[:self.logical_pages]
            # Guarantee a nonidentity mapping even in the smallest one-page test.
            pages[0] = self.physical_pages - 1
            # Preserve uniqueness if the replacement already appeared later.
            duplicates = (pages[1:] == self.physical_pages - 1).nonzero().flatten()
            if duplicates.numel():
                used = set(pages.cpu().tolist())
                replacement = next(i for i in range(self.physical_pages) if i not in used)
                pages[int(duplicates[0]) + 1] = replacement
            self.table[row, :self.logical_pages].copy_(pages)
        short = [1, 63, 64, 65, 127, 128, 129, 255, 256, 257]
        boundary = [2047, 2048, 2049, 4095, 4096, 4097, 0, 1]
        if pattern in ('uniform', 'recovery'):
            live = [self.length] * self.batch
        elif pattern == 'zero':
            live = [0] * self.batch
        elif pattern == 'short':
            live = [min(self.length, short[r % len(short)]) for r in range(self.batch)]
        elif pattern == 'boundaries':
            live = [min(self.length, boundary[r % len(boundary)]) for r in range(self.batch)]
        elif pattern == 'mixed':
            live = [(r * 977 + 17) % (self.length + 1) for r in range(self.batch)]
            for r in (0, 31, 63, 95):
                if r < self.batch:
                    live[r] = 0
                if r + 1 < self.batch:
                    live[r + 1] = self.length
        else:
            raise ValueError(pattern)
        self.live = live
        self.lengths.copy_(t.tensor(live, dtype=t.int32, device=self.device))

    def pointers(self, schedule):
        return {name: tensor.data_ptr() for name, tensor in dict(q=self.q, cache=self.cache,
            weights=self.weights, table=self.table, lengths=self.lengths, schedule=schedule).items()}


def references(torch, scorer, metadata, sms, inputs, label):
    lens = inputs.lengths.reshape(inputs.batch, 1)
    try:
        schedule = metadata(lens, 64, sms(), indices=None)
        dense = scorer(inputs.q, inputs.cache, inputs.weights, lens, inputs.table, schedule,
                       inputs.maximum, clean_logits=False, indices=None)
        result = [dense[row, :n].detach().cpu().contiguous() for row, n in enumerate(inputs.live)]
    except Exception as error:
        raise RuntimeError(f'{label} old scorer is incompatible with the smoke inputs/API; no test was skipped.') from error
    require(all(bool(torch.isfinite(row).all()) for row in result), f'{label}: nonfinite reference score')
    return result


def score_hash(torch, tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def same_scores(torch, actual, expected, label):
    require(len(actual) == len(expected), f'{label}: row count')
    for row, (a, e) in enumerate(zip(actual, expected)):
        require(torch.equal(a.view(torch.int32), e.view(torch.int32)), f'{label}: FP32 score bits differ at row {row}')


def check_full(torch, plan, dense, inputs, refs):
    actual = [dense[row, :n].detach().cpu().contiguous() for row, n in enumerate(inputs.live)]
    same_scores(torch, actual, refs, 'fused scorer')
    table, output = inputs.table.cpu().long(), plan.output.cpu().long()
    for row, (n, reference) in enumerate(zip(inputs.live, refs)):
        k = min(n, TOPK)
        require(bool((output[row, k:] == -1).all()), f'row {row}: missing -1 padding')
        selected = output[row, :k]
        require(selected.numel() == k and torch.unique(selected).numel() == k, f'row {row}: duplicate physical IDs')
        if not k:
            continue
        require(bool((selected >= 0).all()), f'row {row}: negative physical ID')
        page_inverse = {int(table[row, page]): page for page in range((n + 63) // 64)}
        logical = []
        for physical in selected.tolist():
            page, offset = divmod(physical, 64)
            require(page in page_inverse, f'row {row}: unmapped physical page {page}')
            index = page_inverse[page] * 64 + offset
            require(index < n, f'row {row}: physical ID maps outside live length')
            logical.append(index)
        picked = torch.tensor(logical, dtype=torch.int64)
        require(torch.unique(picked).numel() == k, f'row {row}: duplicate logical IDs')
        threshold = torch.topk(reference, k, sorted=False).values.min()
        require(bool((reference[picked] >= threshold).all()), f'row {row}: value below exact torch.topk threshold')
        strict = set((reference > threshold).nonzero().flatten().tolist())
        require(strict.issubset(set(logical)), f'row {row}: missing strictly greater score')
    require(not bool(plan.histogram.any()), 'histogram was not restored')
    require(not bool(plan.workspace.any()), 'workspace was not restored to zero')
    return dict(full_live_score_bits_exact=True, physical_topk_threshold_exact=True,
                strict_set_complete=True, selected_ids_unique=True, physical_mapping_valid=True,
                short_row_padding_exact=True, histogram_restored=True, workspace_restored=True,
                selector_diagnostics=plan.selector_diag.cpu().tolist(),
                row_score_sha256=[score_hash(torch, row) for row in refs])


def check_histogram(torch, plan, inputs, refs, schedule):
    histogram = torch.zeros_like(plan.histogram)
    dense = plan.deepgemm.fp8_fp4_paged_mqa_logits(
        (inputs.q, None), inputs.cache, inputs.weights, inputs.lengths.reshape(inputs.batch, 1), inputs.table,
        schedule, inputs.maximum, clean_logits=False, indices=None, histogram=histogram)
    actual = [dense[row, :n].detach().cpu().contiguous() for row, n in enumerate(inputs.live)]
    same_scores(torch, actual, refs, 'histogram-only scorer')
    cpu_hist = histogram.cpu().long()
    for row, scores in enumerate(refs):
        # hybrid1024-fp16rn16-unit-overflow-v1: FP16-RN bins below 16, unit bins saturating at 223
        bits = scores.view(torch.int32).long() & 0xffffffff
        magnitude = bits & 0x7fffffff
        negative = ((bits >> 31) == 1) & (magnitude != 0)
        code = (scores.half().view(torch.int16).long() & 0x7fff) >> 6
        bounded = torch.clamp(magnitude - negative.long(), max=0x435f0000).int().view(torch.float32)
        code = torch.where(code >= 304, torch.clamp(bounded.floor().long() + 288, min=304), code)
        expected = torch.bincount(torch.where(negative, 512 + code, 511 - code), minlength=1024)
        require(torch.equal(cpu_hist[row], expected), f'row {row}: coarse histogram mismatch')
    require(cpu_hist.sum(1).tolist() == inputs.live, 'histogram totals mismatch')
    return dict(full_live_score_bits_exact=True, all_1024_bins_exact=True)


def run_batch(torch, api, baseline, args, batch, record, hashes):
    inputs = Inputs(torch, batch, args.length, args.max_context_len, torch.device('cuda', args.device), args.seed)
    inputs.set('uniform', 0)
    # This happens before the first private package import in the process.
    before = references(torch, baseline.scorer, baseline.metadata, baseline.sms, inputs, 'default deep_gemm before plan')
    baseline.check()
    plan = api.FusedDecodePlan(inputs.q, inputs.cache, inputs.weights, deepgemm_package=args.deepgemm_package,
                               selector_dir=args.selector_dir, max_context_len=args.max_context_len)
    baseline.check()
    require(plan.deepgemm is not baseline.module, 'private package unexpectedly aliases default deep_gemm')
    require(plan.deepgemm.__name__.startswith('_litetopk_deepgemm_'), 'private package name is not isolated')
    hashes.modules(plan.deepgemm.__name__)
    refs = references(torch, plan.deepgemm.fp8_paged_mqa_logits, plan.deepgemm.get_paged_mqa_logits_metadata,
                      plan.deepgemm.get_num_sms, inputs, 'private deep_gemm')
    same_scores(torch, before, refs, 'default-before-plan versus private-old')
    schedule = plan.metadata(inputs.lengths)
    plan(inputs.table, inputs.lengths, schedule)
    check_full(torch, plan, plan.dense, inputs, refs)
    stream = torch.cuda.Stream(device=args.device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            plan(inputs.table, inputs.lengths, schedule)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        plan(inputs.table, inputs.lengths, schedule)
    # Later eager calls may reassign plan.dense. Keep the actual graph-owned
    # score tensor, rather than accidentally checking the most recent eager one.
    graph_dense = plan.dense
    fixed = inputs.pointers(schedule)
    for variant, pattern in enumerate(('uniform', 'zero', 'short', 'boundaries', 'mixed', 'recovery')):
        inputs.set(pattern, variant)
        fresh_schedule = plan.metadata(inputs.lengths)
        require(fresh_schedule.shape == schedule.shape and fresh_schedule.dtype == schedule.dtype,
                'metadata shape/dtype changed; cannot update the captured graph in place')
        schedule.copy_(fresh_schedule)
        refs = references(torch, plan.deepgemm.fp8_paged_mqa_logits, plan.deepgemm.get_paged_mqa_logits_metadata,
                          plan.deepgemm.get_num_sms, inputs, 'private deep_gemm')
        baseline_after = references(torch, baseline.scorer, baseline.metadata, baseline.sms,
                                    inputs, 'default deep_gemm after private import')
        same_scores(torch, baseline_after, refs, 'default-after versus private-old')
        baseline.check()
        histogram_check = check_histogram(torch, plan, inputs, refs, schedule)
        plan.output.fill_(-123)
        plan(inputs.table, inputs.lengths, schedule)
        eager = check_full(torch, plan, plan.dense, inputs, refs)
        for _ in range(args.graph_replays):
            plan.output.fill_(-123)
            graph.replay()
            graph_check = check_full(torch, plan, graph_dense, inputs, refs)
        require(inputs.pointers(schedule) == fixed, 'captured input pointer changed')
        baseline.check()
        record(dict(batch=batch, pattern=pattern, live_lengths=inputs.live, variant=variant,
                    graph_replays=args.graph_replays, same_graph=True, same_input_pointers=True,
                    updated_contents=['q', 'cache', 'weights', 'table', 'lengths', 'schedule'],
                    negative_weights_each_row=True, nonidentity_unique_page_mapping=True,
                    private_module=plan.deepgemm.__name__, default_module_and_functions_preserved=True,
                    default_before_plan_initial_input_bits_exact=True, default_after_private_old_bits_exact=True,
                    histogram=histogram_check, eager=eager, graph=graph_check))
    torch.cuda.synchronize()
    del graph, graph_dense, plan, inputs, schedule, fresh_schedule
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deepgemm-package', type=Path, default=HERE / 'build/fused/deepgemm')
    parser.add_argument('--selector-dir', type=Path, default=HERE / 'build/fused')
    parser.add_argument('--batches', default='1,3,8,16,33,65,128')
    parser.add_argument('--all-batches', action='store_true', help='Test every actual batch in 1..128.')
    parser.add_argument('--length', type=int, default=4097)
    parser.add_argument('--max-context-len', type=int, default=1_048_576)
    parser.add_argument('--graph-replays', type=int, default=2)
    parser.add_argument('--seed', type=int, default=202609261)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    batches = list(range(1, 129)) if args.all_batches else [int(x) for x in args.batches.split(',')]
    if not batches or len(set(batches)) != len(batches) or any(b < 1 or b > 128 for b in batches):
        parser.error('batches must be unique integers in 1..128')
    if not 1 <= args.length <= args.max_context_len <= 1_048_576 or args.graph_replays < 1:
        parser.error('require 1 <= length <= max-context-len <= 1048576 and graph-replays >= 1')
    report = dict(status='running', arguments={**vars(args), 'batches': batches}, records=[],
        scope=['Correctness-only smoke; no latency or performance claims.',
               'Random finite FP8, signed weights, positive scales, random nonidentity unique physical page mapping.',
               'Score oracle is the private package old native scorer, also compared bitwise to the explicitly resolved default deep_gemm. This is a compatibility oracle, not independent CPU arithmetic.',
               'The default deep_gemm may be provided through process-local PYTHONPATH; no global installation or replacement is claimed. The first old-API call precedes the first private-package import.',
               'CPU torch.topk threshold and strict-set checks permit arbitrary ordering and valid choices among ties.',
               'Each plan has fixed actual B. Only tensor contents change in its captured graph; this does not test device-active batch switching.',
               'Valid lengths/pages/schedules only. No device-invalid fail-safe or all-input correctness claim.',
               'Package manifests, Python APIs and loaded DSOs are fingerprinted. Runtime-generated JIT cubins are not individually enumerated.'])
    hashes = Fingerprints()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report['artifact_hashes'] = hashes.files
        args.output.write_text(json.dumps(report, indent=2, default=str) + '\n')

    def record(item):
        report['records'].append(item)
        save()
        print(json.dumps(dict(batch=item['batch'], pattern=item['pattern'], status='ok')), flush=True)

    save()
    try:
        import torch
        torch.set_num_threads(1)
        torch.cuda.set_device(args.device)
        hashes.add(__file__)
        hashes.add(HERE / 'fused.py')
        report['deepgemm_manifest'] = hashes.manifest(args.deepgemm_package)
        report['selector_manifest'] = hashes.manifest(args.selector_dir)
        baseline = DefaultBinding()
        report['default_binding_before'] = baseline.record()
        hashes.modules('deep_gemm')
        api = load_api()
        prop = torch.cuda.get_device_properties(args.device)
        report['device'] = dict(name=prop.name, capability=[prop.major, prop.minor], sms=prop.multi_processor_count,
                                torch=torch.__version__, cuda=torch.version.cuda)
        for batch in batches:
            run_batch(torch, api, baseline, args, batch, record, hashes)
        baseline.check()
        hashes.modules('deep_gemm')
        report['default_binding_after'] = baseline.record()
        report['hash_guard'] = hashes.check()
        require(report['hash_guard']['ok'], 'artifact changed during smoke')
        report['status'] = 'ok'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
