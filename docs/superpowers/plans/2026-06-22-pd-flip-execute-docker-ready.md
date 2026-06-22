# PD Flip Execute Docker Ready Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the four-node PD flip experiment executable from Docker scripts, with controller-driven D->P active decode KV handoff and P->D idle role switch.

**Architecture:** Keep the existing worker/router admin APIs. Add an `execute` path to `pd_flip_controller.py` that runs the planned HTTP actions, polls source and target migration status, waits for source idle before role mutation, and performs best-effort cleanup if a step fails. Update the Docker harness so one physical 8-GPU node maps to one SGLang worker role by default.

**Tech Stack:** Python dataclasses/urllib controller, existing SGLang PD migration endpoints, Rust router admin endpoints, Bash Docker launch scripts, unittest-based controller tests.

---

## File Structure

- `scripts/playground/disaggregation/pd_flip_controller.py`: executable D->P/P->D orchestration, status polling, timing output, and cleanup.
- `scripts/playground/disaggregation/pd_flip_docker/run_controller.sh`: expose `execute`.
- `scripts/playground/disaggregation/pd_flip_docker/env.example`: make default GPU topology match one 8-GPU node.
- `scripts/playground/disaggregation/pd_flip_docker/README.md`: document real execution command and acceptance checks.
- `test/srt/test_pd_flip_controller.py`: execution order, manifest handoff, polling, and cleanup tests.
- `test/srt/test_pd_flip_experiment_script.py`: Docker harness checks for `execute` and GPU topology defaults.

## Acceptance Criteria

- `python3 test/srt/test_pd_flip_controller.py -v` passes.
- `python3 test/srt/test_pd_flip_experiment_script.py -v` passes.
- `python3 -m py_compile scripts/playground/disaggregation/pd_flip_controller.py` passes.
- Docker harness defaults consume 8 GPUs per physical node as `TP_SIZE * DP_SIZE == 8`.
- `run_controller.sh execute` maps to the controller `execute` subcommand with `DIRECTION` and `SOURCE_NAME`.
- The controller returns structured execution output containing action records, migration timing, total timing, source, target, and success/failure.
- Four-node GPU execution is not claimed complete unless run on actual servers; local acceptance proves the scripts and controller are ready to run there.

## Tasks

- [ ] Add controller execute RED tests.
- [ ] Implement controller execute with D->P migration and P->D idle switch.
- [ ] Add Docker harness tests for `execute` and 8-GPU topology.
- [ ] Update Docker scripts and docs.
- [ ] Run verification and record results.
