"""
Smoke tests for all Streamlit pages using Streamlit's official AppTest framework.
Ensures that app.py and all pages in pages/ compile, initialize session state,
and render without raising unhandled exceptions.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from src.config import PROJECT_ROOT


def test_app_entrypoint_smoke():
    app_file = PROJECT_ROOT / "app.py"
    assert app_file.is_file()
    at = AppTest.from_file(str(app_file), default_timeout=20)
    at.run()
    assert not at.exception, f"app.py raised unhandled exception: {at.exception}"


@pytest.mark.parametrize("page_name", [
    "1_🏠_Home.py",
    "2_🔍_Single_Review.py",
    "3_📁_Bulk_Analysis.py",
    "4_📊_Analytics_Dashboard.py",
    "5_🔎_Review_Explorer.py",
    "6_⚙️_Text_Preprocessing.py",
    "7_📖_About_Project.py",
])
def test_all_pages_render_without_exception(page_name):
    page_path = PROJECT_ROOT / "pages" / page_name
    assert page_path.is_file(), f"Page file {page_name} does not exist!"
    at = AppTest.from_file(str(page_path), default_timeout=20)
    at.run()
    assert not at.exception, f"Page {page_name} raised unhandled exception: {at.exception}"

