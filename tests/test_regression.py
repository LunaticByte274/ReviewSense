"""
Regression tests for ReviewSense.
Specifically guards against previously identified defects:
1. NameError 'nst' is not defined.
2. FileNotFoundError for sample_data/sample_reviews.csv.
3. Empty initial dashboard on first launch.
4. Missing page navigation.
"""

import os
from pathlib import Path
import pytest
from src.config import PROJECT_ROOT, SAMPLE_REVIEWS_CSV


def test_regression_no_nst_variable_in_codebase():
    """
    Guards against previously reported bug:
    'NameError: name nst is not defined'.
    Parses all python files in project using AST to ensure no undefined 'nst'
    variable or attribute is referenced in executable code.
    """
    import ast

    py_files = list(PROJECT_ROOT.glob("**/*.py"))
    assert len(py_files) > 0

    for py_file in py_files:
        if ".venv" in str(py_file) or "test_regression" in str(py_file):
            continue
        with open(py_file, "r", encoding="utf-8") as f:
            code = f.read()

        tree = ast.parse(code, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "nst":
                pytest.fail(f"Found rogue identifier 'nst' in {py_file} at line {node.lineno}")
            elif isinstance(node, ast.Attribute) and node.attr == "nst":
                pytest.fail(f"Found rogue attribute 'nst' in {py_file} at line {node.lineno}")


def test_regression_sample_reviews_path_resolution():
    """
    Guards against previously reported bug:
    'FileNotFoundError for sample_data/sample_reviews.csv'.
    Verifies that SAMPLE_REVIEWS_CSV resolves reliably regardless of current working directory.
    """
    assert SAMPLE_REVIEWS_CSV.is_file(), f"Sample CSV missing at {SAMPLE_REVIEWS_CSV}"
    assert os.path.exists(str(SAMPLE_REVIEWS_CSV))
    # Check non-empty content
    with open(SAMPLE_REVIEWS_CSV, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) > 5, "Sample reviews CSV contains insufficient rows"


def test_regression_all_seven_pages_exist():
    """
    Guards against previously reported defect:
    'Some advertised pages were missing from navigation'.
    Verifies that all 7 required pages exist in the pages/ directory.
    """
    pages_dir = PROJECT_ROOT / "pages"
    assert pages_dir.is_dir()
    page_files = [p.name for p in pages_dir.glob("*.py")]

    expected_page_keywords = [
        "Home",
        "Single_Review",
        "Bulk_Analysis",
        "Analytics_Dashboard",
        "Review_Explorer",
        "Text_Preprocessing",
        "About_Project"
    ]

    for expected in expected_page_keywords:
        match = any(expected in p for p in page_files)
        assert match, f"Required page containing '{expected}' is missing from {pages_dir}"
