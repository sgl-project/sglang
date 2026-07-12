# Task 7 Report: Atomic PD Flip Handoff

## Result

Implemented an atomic target handoff with separate commit and activation phases,
a whole-source delta quiesce barrier, selected-only source finish, abort resume,
and monotonic relayed output sequencing.

## Compatibility design

The existing implementation stores migration sessions and entries as dictionaries,
with target schedulable requests at `entry["decode_req"].req`. The implementation
keeps those containers and adds strict full-session RID validation. Commit changes
all held entries to `ready_to_activate` without scheduling; activation initializes
all non-drop requests first, then performs one `waiting_queue.extend`. Drop entries
participate in every batch-state check but are never enqueued.

Delta cutover is intentionally a two-request protocol because a scheduler control
handler cannot block its own event loop waiting for quiescence:

1. The first delta request records the exact RID set and returns `success=False`, a
   `quiesce pending` message, and no manifests.
2. Both decode loops finish an already launched result, stop launching new work,
   and continue polling control requests.
3. A retry with the same session/RIDs captures C1 and returns delta manifests.
   Further identical retries return the cached manifests without incrementing the
   delta generation. A different RID set is rejected.

Updating the controller to retry this explicit pending response is deferred to
Task 8, as requested.

## Key invariants covered

- Commit requires the complete target session and every entry in
  `transferred_held`; otherwise the whole target session aborts.
- Commit and activation exceptions cannot leave partial entry phases or a partial
  waiting queue.
- HiCache restore commitment is deferred for `prepare_only` transfers and occurs
  during commit, avoiding a double commit.
- Source delta never reads C1 while an overlap result is pending.
- Source finish marks migrated requests, filters the running batch, and resumes
  unselected requests. Source abort never filters the running batch and clears the
  quiesce state.
- Every manifest carries `last_emitted_output_seq`; the target increments an
  output sequence before relay, and the source updates per-RID `last_seen` before
  forwarding while dropping `output_seq <= last_seen`.
- Activate request type, HTTP route, tokenizer control communicator, scheduler
  dispatcher, and request-type convention coverage are connected.

## Verification

Passed:

```text
PYTHONUTF8=1 python -m pytest \
  test/srt/test_pd_flip_atomic_batch.py \
  test/srt/test_pd_flip_active_decode_handoff.py -q

16 passed, 4 skipped
```

When the SGLang runtime import is unavailable, the Task 7 test compiles the real
commit, activate, delta, and quiesce method AST nodes into a lightweight harness.
Nine atomic state-machine and control-path tests therefore execute production method
bodies on Windows rather than being skipped or copying the logic. Four tests that
need the full TokenizerManager/source-release runtime remain explicitly skipped.
Additional AST tests verify route/control/dispatcher wiring and decode-loop ordering.

Also passed `py_compile` for all seven production/test Python files and
`git diff --check`.

The brief's complete command could not collect:

```text
python -m pytest test/srt/test_pd_flip_atomic_batch.py \
  test/srt/test_pd_flip_active_decode_handoff.py \
  test/srt/test_pd_flip_migration_accounting.py -q
```

Windows failure: `ModuleNotFoundError: No module named 'orjson'` while importing
`sglang.srt.utils.common`. The default Windows `python.exe` is also a Microsoft
Store alias; tests used `C:\Users\Tianci J\anaconda3\python.exe`. WSL Python is
available but has no `torch`, so it cannot run the runtime suite either.

## Files

- `python/sglang/srt/managers/io_struct.py`
- `python/sglang/srt/entrypoints/http_server.py`
- `python/sglang/srt/managers/tokenizer_control_mixin.py`
- `python/sglang/srt/managers/tokenizer_manager.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/disaggregation/decode.py`
- `test/srt/test_pd_flip_atomic_batch.py`
