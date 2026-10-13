#!/usr/bin/env python3
"""Small live-server smoke test for DSpark FULL KV release.

Uses the native /generate SSE endpoint, not character-count estimates.
Defaults: DP0/DP1, one normal 2600-token request and one real client
disconnect after 2600 output tokens on each rank. Waits for every DP to
be idle for 60 seconds and verifies another inference after each case.
Requires a restarted server with the cache fix and no other workload.
Python standard library only. Does not change server configuration.
This checks the exercised cases, not full GPQA accuracy or all KV ownership.
"""

import argparse
import datetime as dt
import http.client
import json
from pathlib import Path
import sys
import time
import uuid
from urllib.parse import urlsplit


class Client:
    def __init__(self, base_url, api_key='', timeout=120):
        self.base_url = base_url.rstrip('/')
        self.url = urlsplit(self.base_url)
        if self.url.scheme not in ('http', 'https') or not self.url.hostname:
            raise ValueError('Use a server base URL, e.g. http://127.0.0.1:30100')
        if self.url.username or self.url.password or self.url.query or self.url.fragment:
            raise ValueError('URL credentials/query/fragment are unsupported; use --api-key.')
        self.api_key = api_key
        self.timeout = timeout

    def open(self, method, path, payload=None):
        cls = http.client.HTTPSConnection if self.url.scheme == 'https' else http.client.HTTPConnection
        conn = cls(self.url.hostname, self.url.port, timeout=self.timeout)
        headers = {'Connection': 'close'}
        body = None
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        try:
            conn.request(method, self.url.path.rstrip('/') + path, body=body, headers=headers)
            resp = conn.getresponse()
            if not 200 <= resp.status < 300:
                detail = resp.read(8192).decode('utf-8', errors='replace')
                raise RuntimeError(f'{method} {path}: HTTP {resp.status}: {detail}')
            return conn, resp
        except BaseException:
            conn.close()
            raise

    def get_json(self, path):
        conn, resp = self.open('GET', path)
        try:
            return json.load(resp)
        finally:
            resp.close()
            conn.close()


def stream_events(resp):
    """Parse SSE events and handle multi-line data fields."""
    data = []
    while True:
        line = resp.readline()
        if not line:
            if data:
                yield '\n'.join(data)
            return
        line = line.decode('utf-8').rstrip('\r\n')
        if not line:
            if data:
                yield '\n'.join(data)
                data = []
        elif line.startswith('data:'):
            data.append(line[5:].lstrip(' '))


def generate(client, rank, limit, *, cancel_after=None, temperature=1.0):
    rid = 'kv-release-' + uuid.uuid4().hex
    payload = {
        'rid': rid,
        'cache_salt': rid,
        'routed_dp_rank': rank,
        'text': '这是推理服务稳定性测试。请持续写一篇关于科学研究方法的长文。',
        'sampling_params': {
            'max_new_tokens': limit,
            'temperature': temperature,
            'top_p': 1.0,
            'ignore_eos': True,
        },
        'stream': True,
    }
    record = {
        'mode': 'client_disconnect' if cancel_after is not None else 'normal',
        'rid': rid, 'requested_dp_rank': rank, 'limit': limit,
        'cancel_after': cancel_after,
    }
    start = time.monotonic()
    conn, resp = client.open('POST', '/generate', payload)
    last = None
    cancelled = False
    next_progress = 512
    try:
        if 'text/event-stream' not in (resp.getheader('Content-Type') or ''):
            raise RuntimeError('Server did not return SSE; cannot verify a streaming disconnect.')
        for event in stream_events(resp):
            if event == '[DONE]':
                break
            obj = json.loads(event)
            if not isinstance(obj, dict) or 'error' in obj:
                raise RuntimeError(f'Stream error: {str(obj)[:1500]}')
            meta = obj.get('meta_info')
            if not isinstance(meta, dict) or not isinstance(meta.get('completion_tokens'), int):
                raise RuntimeError('Stream lacks real completion_tokens; threshold cannot be verified.')
            last = meta
            count = meta['completion_tokens']
            if meta.get('dp_rank') is not None and meta['dp_rank'] != rank:
                raise RuntimeError(f'Request was routed to DP{meta["dp_rank"]}, expected DP{rank}.')
            if count >= next_progress:
                print(f'  DP{rank}: completion_tokens={count}, elapsed={time.monotonic()-start:.1f}s', flush=True)
                next_progress = (count // 512 + 1) * 512
            if cancel_after is not None and count >= cancel_after:
                if meta.get('finish_reason') is not None or count >= limit:
                    raise RuntimeError('Request finished before disconnect; cancellation case was not exercised.')
                cancelled = True
                print(f'  CLOSE CLIENT CONNECTION at {count} generated tokens', flush=True)
                break
    finally:
        # Closing this live SSE response is the client cancellation under test.
        resp.close()
        conn.close()
    if last is None:
        raise RuntimeError('No generation metadata received.')
    if cancel_after is not None:
        if not cancelled:
            raise RuntimeError('Generation ended before the cancellation threshold.')
    else:
        reason = last.get('finish_reason')
        reason_type = reason.get('type') if isinstance(reason, dict) else reason
        if reason_type != 'length' or last['completion_tokens'] != limit:
            raise RuntimeError(f'Expected {limit} tokens and length finish, got {last["completion_tokens"]}, {reason}.')
    record.update(
        meta_info=last,
        elapsed_seconds=round(time.monotonic() - start, 3),
        disconnected=cancelled,
    )
    print(f'  {record["mode"]}: observed completion_tokens={last["completion_tokens"]}, prompt_tokens={last.get("prompt_tokens")}', flush=True)
    return record


def check_loads(obj, dp_size):
    loads = obj.get('loads') if isinstance(obj, dict) else None
    if not isinstance(loads, list) or len(loads) != dp_size:
        raise RuntimeError(f'Expected {dp_size} DP snapshots at /v1/loads, got {len(loads) if isinstance(loads, list) else None}.')
    if {x.get('dp_rank') for x in loads} != set(range(dp_size)):
        raise RuntimeError('DP snapshots are duplicated or incomplete.')
    for load in loads:
        for field in ('num_running_reqs', 'num_waiting_reqs'):
            if field not in load or not isinstance(load[field], int) or load[field] < 0:
                raise RuntimeError(f'Invalid {field} on DP{load.get("dp_rank")}.')
        if load.get('num_used_tokens', 0) < 0 or load.get('token_usage', 0) < 0:
            raise RuntimeError(f'Negative KV usage reported on DP{load["dp_rank"]}: {load}')
    idle = all(
        x['num_running_reqs'] == 0 and x['num_waiting_reqs'] == 0
        and all(v == 0 for v in (x.get('queues') or {}).values())
        for x in loads
    )
    return idle, loads


def wait_idle(client, dp_size, hold_seconds, timeout_seconds=180):
    start = time.monotonic()
    idle_since = None
    first_idle = None
    while True:
        obj = client.get_json('/v1/loads?include=core,queues')
        idle, loads = check_loads(obj, dp_size)
        now = time.monotonic()
        if idle:
            if idle_since is None:
                idle_since = now
                first_idle = obj
                print(f'  All {dp_size} DP ranks idle; observe {hold_seconds:g}s', flush=True)
            # This also detects a dead HTTP process behind stale load snapshots.
            models = client.get_json('/v1/models')
            if not isinstance(models, dict) or not isinstance(models.get('data'), list):
                raise RuntimeError('Invalid /v1/models response during idle observation.')
            if now - idle_since >= hold_seconds:
                print('  Idle observation completed; server still responds', flush=True)
                return {'first_idle': first_idle, 'last_idle': obj, 'observed_seconds': round(now-idle_since, 3)}
        else:
            idle_since = None
        if now - start > timeout_seconds + hold_seconds:
            active = [(x['dp_rank'], x['num_running_reqs'], x['num_waiting_reqs']) for x in loads if x['num_running_reqs'] or x['num_waiting_reqs']]
            raise RuntimeError(f'Idle not sustained; outstanding requests or failed cancellation: {active}. Stop other workloads before testing.')
        time.sleep(min(2.0, max(0.02, hold_seconds / 3)))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--url', default='http://127.0.0.1:30100')
    p.add_argument('--api-key', default='')
    p.add_argument('--dp-size', type=int, default=16)
    p.add_argument('--ranks', default='0,1', help='Comma-separated DP ranks; each gets one normal and one cancelled long request.')
    p.add_argument('--tokens', type=int, default=2600, help='Normal completion tokens and cancellation threshold; must exceed 2304.')
    p.add_argument('--cancel-limit', type=int, default=4096)
    p.add_argument('--idle-seconds', type=float, default=60)
    p.add_argument('--socket-timeout', type=float, default=120)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    try:
        args.ranks = [int(x.strip()) for x in args.ranks.split(',')]
    except ValueError:
        p.error('--ranks must be comma-separated integers')
    if not args.ranks or len(set(args.ranks)) != len(args.ranks) or any(x < 0 or x >= args.dp_size for x in args.ranks):
        p.error('Ranks must be unique and within --dp-size')
    if args.tokens <= 2304 or args.cancel_limit <= args.tokens + 128:
        p.error('--tokens must exceed 2304; --cancel-limit must exceed --tokens by more than 128')
    if args.idle_seconds < 0 or args.socket_timeout <= 0:
        p.error('Invalid timing option')
    return args


def main():
    args = parse_args()
    path = args.output or Path('dspark_kv_release_check_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S') + '.json')
    report = {'url': args.url, 'dp_size': args.dp_size, 'ranks': args.ranks, 'tokens': args.tokens, 'idle_seconds': args.idle_seconds, 'cases': [], 'status': 'RUNNING'}
    code = 0
    try:
        client = Client(args.url, args.api_key, args.socket_timeout)
        print('Precheck: server must be restarted with the fix and have no other workload.', flush=True)
        report['initial_idle'] = wait_idle(client, args.dp_size, 2)
        for rank in args.ranks:
            for mode in ('normal', 'client_disconnect'):
                print(f'\nSTART DP{rank} {mode}', flush=True)
                case = generate(client, rank, args.tokens if mode == 'normal' else args.cancel_limit, cancel_after=args.tokens if mode != 'normal' else None)
                report['cases'].append(case)
                case['idle_after_long_request'] = wait_idle(client, args.dp_size, args.idle_seconds)
                case['short_inference_probe'] = generate(client, rank, 16, temperature=0.0)
                case['idle_after_probe'] = wait_idle(client, args.dp_size, 2)
        report['status'] = 'PASS'
        print('\nRESULT: PASS for these normal/disconnect cases. Check server logs for memory-leak warnings; this is not a full GPQA accuracy result.', flush=True)
    except KeyboardInterrupt:
        code = 130
        report.update(status='INTERRUPTED', error='Test script interrupted by user; active stream closed.')
        print('\nRESULT: INTERRUPTED', flush=True)
    except Exception as e:
        code = 1
        report.update(status='FAIL', error=f'{type(e).__name__}: {e}')
        print('\nRESULT: FAIL:', report['error'], flush=True)
    finally:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('REPORT:', path.resolve(), flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
