"""Rolling count of unmapped coded values.

Whichever unmapped_policy is chosen, the RATE is the thing worth watching. A
rising unmapped rate is how a vocabulary update that broke a binding becomes
visible: under 'reject' it shows up as a rising 422 rate that looks like a
caller bug, and under 'map_to_other' it does not show up at all — every episode
still scores, just on a rhythm value the model never really saw.

Deliberately in-process and unbounded-free: a counter, not a time series. The
audit log is the durable record; this exists so /readyz can surface the rate
without anyone having to run a log query first.
"""
from __future__ import annotations

import threading
from collections import Counter, deque

_LOCK = threading.Lock()
_WINDOW = 500

_totals: Counter = Counter()
_recent: deque = deque(maxlen=_WINDOW)
_unmapped_codes: Counter = Counter()


def record(feature: str, unmapped: bool, code: str | None = None) -> None:
    with _LOCK:
        _totals[f"{feature}:total"] += 1
        _recent.append(1 if unmapped else 0)
        if unmapped:
            _totals[f"{feature}:unmapped"] += 1
            if code:
                # The CODE is schema, not patient data: knowing that 426749004
                # keeps arriving unmapped is exactly what a terminology team
                # needs, and it says nothing about any individual.
                _unmapped_codes[f"{feature}:{code}"] += 1


def snapshot() -> dict:
    with _LOCK:
        n = len(_recent)
        return {
            "requests_seen": sum(v for k, v in _totals.items() if k.endswith(":total")),
            "unmapped_total": sum(v for k, v in _totals.items() if k.endswith(":unmapped")),
            "unmapped_rate_recent": round(sum(_recent) / n, 4) if n else 0.0,
            "recent_window": n,
            "top_unmapped_codes": dict(_unmapped_codes.most_common(5)),
        }


def reset() -> None:
    with _LOCK:
        _totals.clear()
        _recent.clear()
        _unmapped_codes.clear()
