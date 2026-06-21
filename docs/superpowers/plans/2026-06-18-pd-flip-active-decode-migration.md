# PD Flip Active Decode Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python-only experimental D-to-P PD flip migration path that can export active decode requests from a source decode worker, prepare them on a target decode worker, expose status, and let `prepare_pd_flip` advance after migrated local requests are released.

**Architecture:** Add scheduler-owned migration session state and Python HTTP control endpoints under `/pd_flip/migration/*`. The first implementation keeps the protocol Python-only and demo-oriented: it snapshots running decode request manifests, tracks source/target migration status, and wires PD flip preparation to migration completion without touching Rust router code or compiled extensions.

**Tech Stack:** Python dataclasses, FastAPI endpoints, existing tokenizer-manager `FanOutCommunicator`, scheduler state machine, unittest/pytest.

---

### Task 1: Migration Control Types And Scheduler State

**Files:**
- Modify: `python/sglang/srt/managers/io_struct.py`
- Modify: `python/sglang/srt/managers/scheduler.py`
- Test: `test/srt/test_pd_flip_internal_state_update.py`

- [ ] Add migration request/response dataclasses to `io_struct.py`.
- [ ] Add scheduler handlers for source start, target prepare, status, source finish, and abort.
- [ ] Add status fields to `get_pd_flip_internal_state`.
- [ ] Write tests that source start exports running request manifests and status reports pending/released counts.
- [ ] Run `python -m pytest test/srt/test_pd_flip_internal_state_update.py -q`.

### Task 2: Migration-Aware Prepare Gate

**Files:**
- Modify: `python/sglang/srt/managers/scheduler.py`
- Modify: `python/sglang/srt/disaggregation/flip_state_machine.py`
- Test: `test/srt/test_pd_flip_internal_state_update.py`
- Test: `test/srt/test_pd_flip_state_machine.py`

- [ ] Write tests showing D-to-P prepare waits while migration is pending and can advance after source finish plus external prepare ack.
- [ ] Update status strategy to `decode_to_decode_kv_transfer` while migration is active.
- [ ] Keep P-to-D and drain-only behavior unchanged.
- [ ] Run `python -m pytest test/srt/test_pd_flip_internal_state_update.py test/srt/test_pd_flip_state_machine.py -q`.

### Task 3: HTTP And Tokenizer Control Wiring

**Files:**
- Modify: `python/sglang/srt/managers/tokenizer_control_mixin.py`
- Modify: `python/sglang/srt/entrypoints/http_server.py`
- Modify: `python/sglang/srt/managers/scheduler.py`
- Test: `test/srt/test_pd_flip_internal_state_update.py`

- [ ] Add migration communicators to tokenizer manager.
- [ ] Add `/pd_flip/migration/source/start`, `/pd_flip/migration/target/prepare`, `/pd_flip/migration/status`, `/pd_flip/migration/source/finish`, and `/pd_flip/migration/abort` endpoints.
- [ ] Add scheduler dispatcher entries for the new dataclasses.
- [ ] Add lightweight tests for tokenizer/scheduler handler objects where practical.
- [ ] Run `python -m pytest test/srt/test_pd_flip_internal_state_update.py -q`.

### Task 4: Experiment Script Driver

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_experiment.py`
- Test: `test/srt/test_pd_flip_experiment_script.py`

- [ ] Add HTTP helpers for migration endpoints.
- [ ] Add `--migration-target-url` to `run-once`.
- [ ] Drive source start, target prepare, status polling, source finish, and prepare ack.
- [ ] Write script tests with the existing `FakeClient`.
- [ ] Run `python -m pytest test/srt/test_pd_flip_experiment_script.py -q`.

### Task 5: Verification And Handoff

**Files:**
- Modify only files touched by earlier tasks if verification finds issues.

- [ ] Run targeted PD flip tests.
- [ ] Run `git status --short`.
- [ ] Summarize the Python-only demo limits and the next step for real backend KV transfer validation on the Ali server.

