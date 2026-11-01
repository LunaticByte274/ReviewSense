"""
ReviewSense - Enterprise Logging & Diagnostics Layer
Provides structured logging, timing decorators, and system telemetry.
"""

import os
import sys
import time
import logging
import functools
from typing import Dict, Any, Optional

# Standard Logger Configuration
LOG_LEVEL_STR = os.getenv("REVIEWSENSE_LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("ReviewSense")


def log_execution_time(func):
    """Decorator to log performance execution timing for compute operations."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_t = time.perf_counter()
        result = func(*args, **kwargs)
        duration_ms = (time.perf_counter() - start_t) * 1000
        logger.info(f"{func.__name__} executed in {duration_ms:.2f}ms")
        return result
    return wrapper


def get_system_telemetry() -> Dict[str, Any]:
    """Retrieves live memory footprint and runtime system health metrics."""
    import platform
    telemetry = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "process_id": os.getpid(),
    }
    
    # Calculate process memory RSS
    try:
        import resource
        # ru_maxrss is in bytes on macOS, kilobytes on Linux
        raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            mem_mb = raw_rss / (1024 * 1024)
        else:
            mem_mb = raw_rss / 1024
        telemetry["memory_rss_mb"] = round(mem_mb, 2)
    except Exception:
        telemetry["memory_rss_mb"] = "N/A"

    return telemetry

