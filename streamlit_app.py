"""
ReviewSense - Streamlit Community Cloud Default Entry Point.
Serves as an alias to app.py so deployments succeed whether
'app.py' or 'streamlit_app.py' is entered in the deployment modal.
"""

from pathlib import Path
import runpy

_APP_ENTRYPOINT = Path(__file__).parent / "app.py"
runpy.run_path(str(_APP_ENTRYPOINT), run_name="__main__")

