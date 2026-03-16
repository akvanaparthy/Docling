# tests/test_convert.py
import io
import pytest
from fastapi.testclient import TestClient


def test_neither_file_nor_url(client):
    r = client.post("/convert", data={"pipeline": "standard", "ocr": "true", "format": "md"})
    assert r.status_code == 400


def test_both_file_and_url(client):
    r = client.post("/convert", data={"url": "http://example.com/doc.pdf", "pipeline": "standard", "ocr": "true", "format": "md"},
                    files={"file": ("test.pdf", b"%PDF-1.4", "application/pdf")})
    assert r.status_code == 400


def test_invalid_url_scheme(client):
    r = client.post("/convert", data={"url": "ftp://example.com/doc.pdf", "pipeline": "standard", "ocr": "true", "format": "md"})
    assert r.status_code == 400


def test_file_upload_returns_job_id(client, monkeypatch):
    # mock the executor submit so no actual conversion runs
    monkeypatch.setattr("app.executor.submit", lambda fn, *a, **kw: None)
    r = client.post("/convert",
                    data={"pipeline": "standard", "ocr": "true", "format": "md"},
                    files={"file": ("test.pdf", b"%PDF-1.4", "application/pdf")})
    assert r.status_code == 200
    assert "job_id" in r.json()


def test_vlm_without_cuda(client, monkeypatch):
    monkeypatch.setattr("app.torch_cuda_available", lambda: False)
    r = client.post("/convert",
                    data={"pipeline": "vlm", "ocr": "true", "format": "md"},
                    files={"file": ("test.pdf", b"%PDF-1.4", "application/pdf")})
    assert r.status_code == 400
    assert "CUDA" in r.json()["detail"]
