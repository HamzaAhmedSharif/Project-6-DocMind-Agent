from pathlib import Path

import pytest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def test_app_explains_how_to_build_the_index_when_none_exists(tmp_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.chdir(tmp_path)  # a directory without data/faiss_index.bin
    at = testing.AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    assert not at.exception
    assert any("No index found" in e.value for e in at.error)
