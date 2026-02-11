import sys
from pathlib import Path

# Add project root to PYTHONPATH for pytest
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# tests/conftest.py
import pytest
import importlib.util
import sys
from pathlib import Path

# Absolute path to app.py
APP_PATH = Path(__file__).resolve().parent.parent / "app.py"

# Load app.py as a real module
spec = importlib.util.spec_from_file_location("flask_entrypoint", APP_PATH)
flask_entrypoint = importlib.util.module_from_spec(spec)

# 🔑 REGISTER THE MODULE (this is what was missing)
sys.modules["flask_entrypoint"] = flask_entrypoint

spec.loader.exec_module(flask_entrypoint)


@pytest.fixture
def client():
    flask_app = flask_entrypoint.app
    flask_app.config["TESTING"] = True

    with flask_app.test_client() as client:
        yield client