"""ReviewSense Source Package."""
import sys
import importlib


def reload_src_modules():
    """Hot-reloads all previously imported src.* modules from disk.
    Prevents stale in-memory module caching across Streamlit page transitions.
    """
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("src.") and mod_name != "src":
            try:
                importlib.reload(sys.modules[mod_name])
            except Exception:
                pass
