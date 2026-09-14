"""Benchmark harness: quantify NPU vs CPU. These numbers are your resume/proposal headline.

Produces the "3-5x lower latency on NPU" claim with real measurements instead of hand-waving.
Run against any ONNX model with representative inputs.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable

from .runtime import EdgeSession


@dataclass
class BenchResult:
    provider: str
    runs: int
    mean_ms: float
    p50_ms: float
    p90_ms: float

    def __str__(self) -> str:
        return (f"{self.provider:<24} mean={self.mean_ms:7.1f}ms "
                f"p50={self.p50_ms:7.1f}ms p90={self.p90_ms:7.1f}ms  (n={self.runs})")


def _time_session(sess: EdgeSession, make_feeds: Callable[[], dict[str, Any]],
                  runs: int, warmup: int) -> BenchResult:
    for _ in range(warmup):
        sess.run(make_feeds())
    samples: list[float] = []
    for _ in range(runs):
        feeds = make_feeds()
        t0 = time.perf_counter()
        sess.run(feeds)
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return BenchResult(
        provider=sess.active_provider,
        runs=runs,
        mean_ms=statistics.fmean(samples),
        p50_ms=samples[len(samples) // 2],
        p90_ms=samples[int(len(samples) * 0.9)],
    )


def compare_npu_vs_cpu(model_path: str, make_feeds: Callable[[], dict[str, Any]],
                       runs: int = 30, warmup: int = 3) -> dict[str, BenchResult]:
    """Benchmark the same model on NPU (if available) and on CPU. Returns both results."""
    results: dict[str, BenchResult] = {}

    npu = EdgeSession(model_path, prefer_npu=True)
    results[npu.active_provider] = _time_session(npu, make_feeds, runs, warmup)

    if npu.on_npu:  # only worth a second run if the first one used the NPU
        cpu = EdgeSession(model_path, prefer_npu=False)
        results[cpu.active_provider] = _time_session(cpu, make_feeds, runs, warmup)

    return results


def print_comparison(results: dict[str, BenchResult]) -> None:
    for r in results.values():
        print(r)
    if len(results) == 2:
        vals = list(results.values())
        speedup = vals[1].mean_ms / vals[0].mean_ms
        faster, slower = (vals[0], vals[1]) if vals[0].mean_ms < vals[1].mean_ms else (vals[1], vals[0])
        print(f"\n→ {faster.provider} is {slower.mean_ms / faster.mean_ms:.1f}x faster than {slower.provider}")
