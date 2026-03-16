# tests/test_stream.py
import pytest
from app import jobs, _make_job


def _inject_done_job(fmt="md"):
    job_id, job = _make_job(fmt)
    job["status"] = "done"
    job["queue"].put("Converting page 1")
    job["queue"].put(None)
    jobs[job_id] = job
    return job_id


def _inject_error_job():
    job_id, job = _make_job("md")
    job["status"] = "error"
    job["error"] = "Something went wrong"
    job["queue"].put(None)
    jobs[job_id] = job
    return job_id


def test_stream_not_found(client):
    r = client.get("/stream/nonexistent-id")
    assert r.status_code == 404


def test_stream_done_job(client):
    job_id = _inject_done_job()
    r = client.get(f"/stream/{job_id}")
    assert r.status_code == 200
    text = r.text
    assert "event: log" in text
    assert "Converting page 1" in text
    assert "event: done" in text


def test_stream_error_job(client):
    job_id = _inject_error_job()
    r = client.get(f"/stream/{job_id}")
    assert r.status_code == 200
    text = r.text
    assert "event: error" in text
    assert "Something went wrong" in text
