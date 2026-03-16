# tests/test_result.py
from pathlib import Path
from app import jobs, _make_job, OUTPUTS


def _inject_done_job_with_file(fmt="md", content="# Hello"):
    job_id, job = _make_job(fmt)
    job["status"] = "done"
    out = OUTPUTS / job_id
    out.mkdir(parents=True, exist_ok=True)
    ext = {"md": ".md", "html": ".html", "json": ".json", "doctags": ".doctags"}[fmt]
    p = out / f"result{ext}"
    p.write_text(content, encoding="utf-8")
    job["result_path"] = p
    jobs[job_id] = job
    return job_id


def test_result_not_found(client):
    r = client.get("/result/nonexistent")
    assert r.status_code == 404


def test_result_pending_job(client):
    job_id, job = _make_job("md")
    job["status"] = "pending"
    jobs[job_id] = job
    r = client.get(f"/result/{job_id}")
    assert r.status_code == 404


def test_result_done(client):
    job_id = _inject_done_job_with_file("md", "# Hello World")
    r = client.get(f"/result/{job_id}")
    assert r.status_code == 200
    data = r.json()
    assert data["format"] == "md"
    assert "Hello World" in data["content"]


def test_download_not_found(client):
    r = client.get("/download/nonexistent")
    assert r.status_code == 404


def test_download_done(client):
    job_id = _inject_done_job_with_file("md", "# Download Test")
    r = client.get(f"/download/{job_id}")
    assert r.status_code == 200
    assert "attachment" in r.headers.get("content-disposition", "")
