"""
Unit and integration tests for ReviewSense CLI runner and System Telemetry.
"""

import sys
import subprocess
import pytest
from pathlib import Path
from src.config import PROJECT_ROOT, SAMPLE_REVIEWS_CSV
from src.logger import get_system_telemetry, log_execution_time


def test_system_telemetry():
    telemetry = get_system_telemetry()
    assert "python_version" in telemetry
    assert "platform" in telemetry
    assert "memory_rss_mb" in telemetry
    assert telemetry["process_id"] > 0


def test_log_execution_time_decorator():
    @log_execution_time
    def dummy_compute(x, y):
        return x + y

    assert dummy_compute(10, 20) == 30


def test_cli_execution_via_subprocess(tmp_path):
    cli_script = PROJECT_ROOT / "cli.py"
    assert cli_script.is_file()

    out_csv = tmp_path / "enriched.csv"
    out_report = tmp_path / "report.html"

    cmd = [
        sys.executable,
        str(cli_script),
        "--input", str(SAMPLE_REVIEWS_CSV),
        "--output", str(out_csv),
        "--report", str(out_report)
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI failed: {res.stderr}"
    assert out_csv.is_file()
    assert out_report.is_file()

    # Verify output CSV content
    with open(out_csv, "r", encoding="utf-8") as f:
        header = f.readline()
        assert "tb_polarity" in header
        assert "tb_sentiment" in header

    # Verify output HTML report content
    with open(out_report, "r", encoding="utf-8") as f:
        html = f.read()
        assert "ReviewSense Executive Report" in html
        assert "Net Sentiment Score" in html
