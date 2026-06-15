from __future__ import annotations

import os
import sys


class MemoryLimitExceeded(RuntimeError):
    pass


def _process_memory_mb() -> float | None:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1024**2
    except Exception:
        pass

    if sys.platform.startswith("linux") or sys.platform == "darwin":
        try:
            import resource

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return rss / 1024 if sys.platform.startswith("linux") else rss / 1024**2
        except Exception:
            return None
    return None


def _available_memory_gb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().available / 1024**3
    except Exception:
        return None


def log_memory(label: str) -> None:
    rss = _process_memory_mb()
    available = _available_memory_gb()
    parts = [f"[memory] {label}"]
    if rss is not None:
        parts.append(f"rss={rss:.1f} MB")
    if available is not None:
        parts.append(f"available={available:.1f} GB")
    print(" | ".join(parts))


def check_memory(
    label: str,
    max_rss_gb: float = 30.0,
    min_available_gb: float = 8.0,
) -> None:
    rss_mb = _process_memory_mb()
    available_gb = _available_memory_gb()
    log_memory(label)

    if rss_mb is not None and rss_mb / 1024 > max_rss_gb:
        raise MemoryLimitExceeded(
            f"Memory guard stopped at {label}: process RSS {rss_mb / 1024:.1f} GB "
            f"> max_rss_gb={max_rss_gb:.1f} GB"
        )
    if available_gb is not None and available_gb < min_available_gb:
        raise MemoryLimitExceeded(
            f"Memory guard stopped at {label}: available memory {available_gb:.1f} GB "
            f"< min_available_gb={min_available_gb:.1f} GB"
        )
