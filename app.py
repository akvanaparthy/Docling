import asyncio
import json as _json
import logging
import queue
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

import httpx
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

UPLOADS = Path("uploads")
OUTPUTS = Path("outputs")
MAX_UPLOAD_MB = int(__import__("os").environ.get("MAX_UPLOAD_MB", 200))
MAX_BYTES = MAX_UPLOAD_MB * 1024 * 1024

jobs: dict[str, dict] = {}

_cleanup_task = None  # module-level ref prevents GC


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cleanup_task
    UPLOADS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    _cleanup_task = asyncio.create_task(cleanup_loop())
    yield
    _cleanup_task.cancel()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


async def cleanup_loop():
    while True:
        await asyncio.sleep(300)
        cutoff = datetime.utcnow() - timedelta(hours=1)
        for job_id, job in list(jobs.items()):
            if job["status"] in ("done", "error") and job["created_at"] < cutoff:
                shutil.rmtree(OUTPUTS / job_id, ignore_errors=True)
                for f in UPLOADS.glob(f"{job_id}*"):
                    f.unlink(missing_ok=True)
                del jobs[job_id]


executor = ThreadPoolExecutor(max_workers=2)


EXT_MAP = {"md": ".md", "html": ".html", "json": ".json", "doctags": ".doctags"}

_thread_local = threading.local()  # stores current job_id per worker thread


def _export_result(result, fmt: str) -> str:
    doc = result.document
    if fmt == "md":
        return doc.export_to_markdown()
    elif fmt == "html":
        import markdown as md_lib
        return md_lib.markdown(doc.export_to_markdown())
    elif fmt == "json":
        import json
        return json.dumps(doc.model_dump(), indent=2)
    elif fmt == "doctags":
        return doc.export_to_doctags()
    raise ValueError(f"Unknown format: {fmt}")


class _JobFilter(logging.Filter):
    """Only passes log records emitted from the thread owning this job."""
    def __init__(self, job_id: str):
        super().__init__()
        self._job_id = job_id

    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(_thread_local, "job_id", None) == self._job_id


class _QueueHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord):
        try:
            self.q.put_nowait(self.format(record))
        except Exception:
            pass


def _run_conversion(job_id: str, source: str, pipeline: str, ocr: bool, fmt: str):
    _thread_local.job_id = job_id  # mark this thread with the job_id for log filtering
    job = jobs[job_id]
    job["status"] = "running"

    handler = _QueueHandler(job["queue"])
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    handler.addFilter(_JobFilter(job_id))  # only capture this thread's logs
    docling_logger = logging.getLogger("docling")
    docling_logger.addHandler(handler)

    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PipelineOptions

        if pipeline == "vlm":
            try:
                from docling.datamodel.pipeline_options import VlmPipelineOptions
                opts = VlmPipelineOptions(do_ocr=ocr)
            except ImportError:
                opts = PipelineOptions(do_ocr=ocr)
        else:
            opts = PipelineOptions(do_ocr=ocr)

        converter = DocumentConverter()
        result = converter.convert(source)

        text = _export_result(result, fmt)
        out_dir = OUTPUTS / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = EXT_MAP[fmt]
        out_path = out_dir / f"result{ext}"
        out_path.write_text(text, encoding="utf-8")

        job["result_path"] = out_path
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
    finally:
        docling_logger.removeHandler(handler)
        job["queue"].put(None)  # sentinel


def torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _make_job(fmt: str) -> tuple[str, dict]:
    job_id = str(uuid.uuid4())
    job = {
        "status": "pending",
        "queue": queue.Queue(),
        "result_path": None,
        "format": fmt,
        "error": None,
        "created_at": datetime.utcnow(),
    }
    return job_id, job


@app.post("/convert")
async def convert(
    file: UploadFile | None = None,
    url: str | None = Form(default=None),
    pipeline: str = Form(default="standard"),
    ocr: bool = Form(default=True),
    format: str = Form(default="md"),
):
    # Validate input
    if file is None and not url:
        raise HTTPException(400, "Provide either a file or a URL.")
    if file is not None and url:
        raise HTTPException(400, "Provide either a file or a URL, not both.")
    if url and not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(400, "URL must start with http:// or https://")
    if pipeline == "vlm" and not torch_cuda_available():
        raise HTTPException(400, "VLM pipeline requires a CUDA GPU. Switch to standard pipeline.")

    job_id, job = _make_job(format)
    jobs[job_id] = job

    if file is not None:
        suffix = Path(file.filename or "upload").suffix or ".bin"
        upload_path = UPLOADS / f"{job_id}{suffix}"
        content = await file.read()
        if len(content) > MAX_BYTES:
            del jobs[job_id]
            raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB}MB limit.")
        upload_path.write_bytes(content)
        source = str(upload_path)
    else:
        suffix = Path(url.split("?")[0]).suffix or ".bin"
        upload_path = UPLOADS / f"{job_id}{suffix}"
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                async with client.stream("GET", url) as resp:
                    cl = resp.headers.get("content-length")
                    if cl and int(cl) > MAX_BYTES:
                        del jobs[job_id]
                        raise HTTPException(413, f"Remote file exceeds {MAX_UPLOAD_MB}MB limit.")
                    total = 0
                    with open(upload_path, "wb") as f:
                        async for chunk in resp.aiter_bytes(65536):
                            total += len(chunk)
                            if total > MAX_BYTES:
                                upload_path.unlink(missing_ok=True)
                                del jobs[job_id]
                                raise HTTPException(413, f"Remote file exceeds {MAX_UPLOAD_MB}MB limit.")
                            f.write(chunk)
        except httpx.TimeoutException:
            upload_path.unlink(missing_ok=True)
            del jobs[job_id]
            raise HTTPException(400, "URL fetch timed out after 30 seconds.")
        source = str(upload_path)

    executor.submit(_run_conversion, job_id, source, pipeline, ocr, format)
    return {"job_id": job_id}


@app.get("/stream/{job_id}")
async def stream(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found.")

    async def generator():
        job = jobs[job_id]
        loop = asyncio.get_running_loop()
        while True:
            try:
                # run_in_executor so blocking queue.get doesn't block the event loop
                msg = await loop.run_in_executor(
                    None, lambda: job["queue"].get(timeout=0.1)
                )
            except queue.Empty:
                if job["status"] in ("done", "error"):
                    # queue fully drained, job already finished before client connected
                    if job["status"] == "error":
                        yield f"event: error\ndata: {job['error']}\n\n"
                    else:
                        yield f"event: done\ndata: {_json.dumps({'format': job['format']})}\n\n"
                    return
                continue
            else:
                if msg is None:  # sentinel
                    if job["status"] == "error":
                        yield f"event: error\ndata: {job['error']}\n\n"
                    else:
                        yield f"event: done\ndata: {_json.dumps({'format': job['format']})}\n\n"
                    return
                yield f"event: log\ndata: {msg}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


MIME_MAP = {
    ".md": "text/markdown",
    ".html": "text/html",
    ".json": "application/json",
    ".doctags": "text/plain",
}


@app.get("/result/{job_id}")
def result(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Result not available.")
    content = job["result_path"].read_text(encoding="utf-8")
    return {"content": content, "format": job["format"]}


@app.get("/download/{job_id}")
def download(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Result not available.")
    path = job["result_path"]
    mime = MIME_MAP.get(path.suffix, "text/plain")
    return FileResponse(
        path,
        media_type=mime,
        filename=f"result{path.suffix}",
        headers={"Content-Disposition": f"attachment; filename=result{path.suffix}"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
