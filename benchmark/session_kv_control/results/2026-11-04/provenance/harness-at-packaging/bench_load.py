"""Opt-in client background inference. No traffic shaping or service mutation."""
import hashlib
import threading
import time


class BackgroundLoad:
    """Closed-loop concurrency; throughput counts only completions in the window.

    Outstanding requests drain after the cutoff and are reported separately.
    Fresh salts prevent background traffic hitting the foreground target.
    """
    def __init__(self, http, tokens, output_tokens, concurrency, trial, seed, salt_namespace=""):
        self.http, self.tokens, self.output_tokens = http, tokens, output_tokens
        self.concurrency, self.trial, self.seed = concurrency, trial, seed
        self.salt_namespace = salt_namespace
        self.stop_event, self.lock, self.started_event = threading.Event(), threading.Lock(), threading.Event()
        self.rows, self.threads = [], []
        self.started_at = self.cutoff = None

    def start(self):
        self.started_at = time.monotonic()
        for worker in range(self.concurrency):
            thread = threading.Thread(target=self._worker, args=(worker,), daemon=True)
            self.threads.append(thread)
            thread.start()
        return self

    def _worker(self, worker):
        sequence = 0
        while not self.stop_event.is_set():
            key = f"{self.trial}:{worker}:{sequence}"
            sequence += 1
            salt = "background-"+hashlib.sha256((self.salt_namespace+":"+key).encode()).hexdigest()[:40]
            row = {"id": key, "started_at": time.monotonic(), "worker": worker}
            self.started_event.set()
            try:
                response = self.http.generate(self.tokens, salt, self.output_tokens,
                    trial=self.trial, seed=self.seed, origin="background")
                row.update(success=True, output_tokens=response["meta_info"]["completion_tokens"],
                           ttft_ms=response["ttft_ms"], latency_ms=response["latency_ms"])
            except Exception as exc:
                row.update(success=False, error=f"{type(exc).__name__}: {exc}", output_tokens=0)
            row["completed_at"] = time.monotonic()
            with self.lock:
                self.rows.append(row)

    def stop(self):
        if self.cutoff is None:
            self.cutoff = time.monotonic()
            self.stop_event.set()
        for thread in self.threads:
            thread.join(self.http.timeout*2+5)
        if any(t.is_alive() for t in self.threads):
            raise RuntimeError("background request drain remains unknown after bounded timeout")
        duration = self.cutoff-self.started_at
        in_window = [x for x in self.rows if x["completed_at"] <= self.cutoff]
        successful = [x for x in in_window if x["success"]]
        return {"concurrency": self.concurrency, "load_model": "closed_loop_fresh_salt",
                "started_at": self.started_at, "cutoff_at": self.cutoff, "measurement_s": duration,
                "started_requests": len(self.rows), "in_window_completed": len(in_window),
                "in_window_successful": len(successful), "in_window_failures": len(in_window)-len(successful),
                "drained_after_cutoff": len(self.rows)-len(in_window),
                "total_failures": sum(not row["success"] for row in self.rows),
                "drained_failures": sum(not row["success"] for row in self.rows if row["completed_at"] > self.cutoff),
                "completion_cohort_requests_per_s": len(successful)/duration if duration else None,
                "completion_cohort_output_tokens_per_s": sum(x["output_tokens"] for x in successful)/duration if duration else None,
                "drain_ms": (time.monotonic()-self.cutoff)*1000,
                "throughput_definition": "successful requests whose completion is inside measurement window; boundary requests drain separately",
                "requests": self.rows}
