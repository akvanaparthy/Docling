import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "uploads").mkdir()
    (tmp_path / "outputs").mkdir()
    (tmp_path / "static").mkdir()
    (tmp_path / "static" / "index.html").write_text("<html/>")
    with TestClient(app) as c:
        yield c
