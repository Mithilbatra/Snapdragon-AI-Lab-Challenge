"""Latency benchmarking for the NPU vs CPU comparison in the proposal."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable

from .runtime import EdgeSession

Feeds = Callable[[], dict[str, Any]]


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


def _measure(session: EdgeSession, make_feeds: Feeds, runs: int, warmup: int) -> BenchResult:
    for _ in range(warmup):
        session.run(make_feeds())

    samples = []
    for _ in range(runs):
        feeds = make_feeds()
        start = time.perf_counter()
        session.run(feeds)
        samples.append((time.perf_counter() - start) * 1000)

    samples.sort()
    return BenchResult(
        provider=session.active_provider,
        runs=runs,
        mean_ms=statistics.fmean(samples),
        p50_ms=samples[len(samples) // 2],
        p90_ms=samples[int(len(samples) * 0.9)],
    )


def compare_npu_vs_cpu(model_path: str, make_feeds: Feeds,
                       runs: int = 30, warmup: int = 3) -> dict[str, BenchResult]:
    npu = EdgeSession(model_path, prefer_npu=True)
    results = {npu.active_provider: _measure(npu, make_feeds, runs, warmup)}

    if npu.on_npu:
        cpu = EdgeSession(model_path, prefer_npu=False)
        results[cpu.active_provider] = _measure(cpu, make_feeds, runs, warmup)

    return results


def print_comparison(results: dict[str, BenchResult]) -> None:
    for result in results.values():
        print(result)
    if len(results) == 2:
        fast, slow = sorted(results.values(), key=lambda r: r.mean_ms)
        print(f"\n{fast.provider} is {slow.mean_ms / fast.mean_ms:.1f}x faster than {slow.provider}")
