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

PIC_DESC_PRESETS = {
    "smolvlm":        {"label": "SmolVLM-256M",       "repo_id": "HuggingFaceTB/SmolVLM-256M-Instruct",    "preset_id": "smolvlm"},
    "granite_vision": {"label": "Granite-Vision-3.3-2B", "repo_id": "ibm-granite/granite-vision-3.3-2b",   "preset_id": "granite_vision"},
    "pixtral":        {"label": "Pixtral-12B",         "repo_id": "mistral-community/pixtral-12b",          "preset_id": "pixtral"},
    "qwen":           {"label": "Qwen2.5-VL-3B",       "repo_id": "Qwen/Qwen2.5-VL-3B-Instruct",           "preset_id": "qwen"},
}

VLM_PRESETS = {
    "GRANITEDOCLING": {"label": "Granite-Docling-258M", "repo_id": "ibm-granite/granite-docling-258M",           "spec": "GRANITEDOCLING_TRANSFORMERS"},
    "SMOLDOCLING":    {"label": "SmolDocling-256M",      "repo_id": "docling-project/SmolDocling-256M-preview",   "spec": "SMOLDOCLING_TRANSFORMERS"},
    "GRANITE_VISION": {"label": "Granite-Vision-3.2-2B", "repo_id": "ibm-granite/granite-vision-3.2-2b",         "spec": "GRANITE_VISION_TRANSFORMERS"},
    "PHI4":           {"label": "Phi-4-multimodal",       "repo_id": "microsoft/Phi-4-multimodal-instruct",       "spec": "PHI4_TRANSFORMERS"},
    "DOLPHIN":        {"label": "Dolphin",                "repo_id": "ByteDance/Dolphin",                         "spec": "DOLPHIN_TRANSFORMERS"},
    "GOT2":           {"label": "GOT-OCR-2.0",            "repo_id": "stepfun-ai/GOT-OCR-2.0-hf",                "spec": "GOT2_TRANSFORMERS"},
}

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
        return json.dumps(doc.model_dump(mode='json'), indent=2)
    elif fmt == "doctags":
        return doc.export_to_doctags()
    raise ValueError(f"Unknown format: {fmt}")


class _JobFilter(logging.Filter):
    """Passes log records from the owning thread OR from Docling pipeline stage threads."""
    def __init__(self, job_id: str):
        super().__init__()
        self._job_id = job_id

    def filter(self, record: logging.LogRecord) -> bool:
        # Allow logs from our executor thread (has job_id set)
        if getattr(_thread_local, "job_id", None) == self._job_id:
            return True
        # Allow logs from Docling's internal pipeline stage threads
        # (Stage-preprocess, Stage-ocr, Stage-layout, Stage-table, Stage-assemble)
        if record.threadName and record.threadName.startswith("Stage-"):
            return True
        return False


class _QueueHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord):
        try:
            self.q.put_nowait(self.format(record))
        except Exception:
            pass


def _run_conversion(job_id: str, source: str, pipeline: str, ocr: bool, fmt: str,
                    do_picture_description: bool = False, pic_desc_model: str = "smolvlm",
                    vlm_model: str = "GRANITEDOCLING",
                    do_chunk: bool = False, chunk_max_tokens: int = 256,
                    page_from: int = 1, page_to: int = 0, pdf_backend: str = "docling"):
    import time, json as _json2
    _thread_local.job_id = job_id
    job = jobs[job_id]
    job["status"] = "running"
    t_total_start = time.perf_counter()

    handler = _QueueHandler(job["queue"])
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    handler.addFilter(_JobFilter(job_id))
    docling_logger = logging.getLogger("docling")
    docling_logger.setLevel(logging.DEBUG)
    docling_logger.addHandler(handler)

    def send_timing(data: dict):
        job["queue"].put(f"__TIMING__:{_json2.dumps(data)}")

    def send_info(data: dict):
        job["queue"].put(f"__INFO__:{_json2.dumps(data)}")

    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.settings import settings
        from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
        _backend_cls = PyPdfiumDocumentBackend if pdf_backend == "pypdfium" else DoclingParseV4DocumentBackend

        settings.debug.profile_pipeline_timings = True
        device = "cuda" if torch_cuda_available() else "cpu"

        def _pic_desc_options():
            from docling.datamodel.pipeline_options import PictureDescriptionVlmEngineOptions
            preset_id = PIC_DESC_PRESETS.get(pic_desc_model, PIC_DESC_PRESETS["smolvlm"])["preset_id"]
            return PictureDescriptionVlmEngineOptions.from_preset(preset_id)

        def _apply_pic_opts(base: dict) -> dict:
            if do_picture_description:
                base["picture_description_options"] = _pic_desc_options()
                base["generate_picture_images"] = True
            return base

        t0 = time.perf_counter()
        if pipeline == "vlm":
            try:
                from docling.datamodel.pipeline_options import VlmPipelineOptions
                from docling.pipeline.vlm_pipeline import VlmPipeline
                from docling.datamodel import vlm_model_specs
                preset_info = VLM_PRESETS.get(vlm_model, VLM_PRESETS["GRANITEDOCLING"])
                preset = getattr(vlm_model_specs, preset_info["spec"])
                pic_opts = _apply_pic_opts({"do_picture_description": do_picture_description})
                opts = VlmPipelineOptions(vlm_options=preset, **pic_opts)
                send_info({"pipeline": "vlm", "model": preset_info["label"], "device": device,
                           "do_picture_description": do_picture_description,
                           "pic_desc_model": PIC_DESC_PRESETS.get(pic_desc_model, {}).get("label") if do_picture_description else None})
                converter = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_cls=VlmPipeline, backend=_backend_cls, pipeline_options=opts)}
                )
            except ImportError:
                pic_opts = _apply_pic_opts({"do_picture_description": do_picture_description})
                opts = PdfPipelineOptions(do_ocr=ocr, **pic_opts)
                send_info({"pipeline": "standard", "model": None, "device": device,
                           "do_picture_description": do_picture_description, "ocr": ocr,
                           "pic_desc_model": PIC_DESC_PRESETS.get(pic_desc_model, {}).get("label") if do_picture_description else None})
                converter = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(backend=_backend_cls, pipeline_options=opts)}
                )
        else:
            pic_opts = _apply_pic_opts({"do_picture_description": do_picture_description})
            opts = PdfPipelineOptions(do_ocr=ocr, **pic_opts)
            send_info({"pipeline": "standard", "model": None, "device": device,
                       "do_picture_description": do_picture_description, "ocr": ocr,
                       "pic_desc_model": PIC_DESC_PRESETS.get(pic_desc_model, {}).get("label") if do_picture_description else None})
            converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(backend=_backend_cls, pipeline_options=opts)}
            )
        send_timing({"stage": "pipeline_init", "duration": round(time.perf_counter() - t0, 2)})

        import sys as _sys
        p_range = (page_from, page_to if page_to > 0 else _sys.maxsize)
        t0 = time.perf_counter()
        result = converter.convert(source, page_range=p_range)
        conversion_time = round(time.perf_counter() - t0, 2)

        # Extract Docling's built-in per-stage profiling data
        timings = {}
        for key, item in (result.timings or {}).items():
            timings[key] = {
                "total": round(item.total(), 2),
                "avg": round(float(item.avg()), 2) if item.count > 0 else 0,
                "count": item.count,
                "scope": item.scope.value,
            }

        page_count = len(result.document.pages) if result.document.pages else 0
        job["page_count"] = page_count
        send_timing({
            "stage": "conversion_done",
            "duration": conversion_time,
            "page_count": page_count,
            "timings": timings,
        })

        t0 = time.perf_counter()
        text = _export_result(result, fmt)
        send_timing({"stage": "export_done", "duration": round(time.perf_counter() - t0, 2)})

        out_dir = OUTPUTS / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = EXT_MAP[fmt]
        out_path = out_dir / f"result{ext}"
        out_path.write_text(text, encoding="utf-8")

        if do_chunk:
            t0 = time.perf_counter()
            from docling.chunking import HybridChunker
            import json as _json3
            chunker = HybridChunker(max_tokens=chunk_max_tokens)
            chunks_display = []
            chunks_full = []
            for i, chunk in enumerate(chunker.chunk(result.document)):
                page = None
                if chunk.meta.doc_items:
                    prov = getattr(chunk.meta.doc_items[0], 'prov', None)
                    if prov:
                        page = prov[0].page_no
                chunks_display.append({
                    "index": i,
                    "text": chunk.text,
                    "headings": chunk.meta.headings or [],
                    "page": page,
                })
                chunks_full.append(chunk.model_dump(mode='json'))
            job["chunks"] = chunks_display
            # write full chunks to file for download
            chunks_path = out_dir / "chunks.json"
            chunks_path.write_text(_json3.dumps(chunks_full, indent=2), encoding="utf-8")
            job["chunks_path"] = chunks_path
            send_timing({"stage": "chunking_done", "duration": round(time.perf_counter() - t0, 2),
                         "chunk_count": len(chunks_display)})

        send_timing({"stage": "total", "duration": round(time.perf_counter() - t_total_start, 2)})
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
        "chunks": None,
        "chunks_path": None,
        "page_count": 0,
        "error": None,
        "created_at": datetime.utcnow(),
    }
    return job_id, job


@app.get("/model-status")
def get_model_status():
    import os
    from pathlib import Path as _Path
    hf_cache = _Path(os.environ.get("HF_HOME", str(_Path.home() / ".cache" / "huggingface"))) / "hub"
    def cached(repo_id):
        return (hf_cache / ("models--" + repo_id.replace("/", "--"))).exists()
    return {
        "vlm":      {k: cached(v["repo_id"]) for k, v in VLM_PRESETS.items()},
        "pic_desc": {k: cached(v["repo_id"]) for k, v in PIC_DESC_PRESETS.items()},
    }


@app.post("/convert")
async def convert(
    file: UploadFile | None = None,
    url: str | None = Form(default=None),
    pipeline: str = Form(default="standard"),
    ocr: bool = Form(default=True),
    format: str = Form(default="md"),
    do_picture_description: bool = Form(default=False),
    pic_desc_model: str = Form(default="smolvlm"),
    vlm_model: str = Form(default="GRANITEDOCLING"),
    do_chunk: bool = Form(default=False),
    chunk_max_tokens: int = Form(default=256),
    page_from: int = Form(default=1),
    page_to: int = Form(default=0),
    pdf_backend: str = Form(default="docling"),
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

    executor.submit(_run_conversion, job_id, source, pipeline, ocr, format, do_picture_description, pic_desc_model, vlm_model, do_chunk, chunk_max_tokens, page_from, page_to, pdf_backend)
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
                safe_msg = msg.replace("\n", " ")
                if safe_msg.startswith("__TIMING__:"):
                    yield f"event: timing\ndata: {safe_msg[11:]}\n\n"
                elif safe_msg.startswith("__INFO__:"):
                    yield f"event: info\ndata: {safe_msg[9:]}\n\n"
                else:
                    yield f"event: log\ndata: {safe_msg}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


MIME_MAP = {
    ".md": "text/markdown",
    ".html": "text/html",
    ".json": "application/json",
    ".doctags": "text/plain",
}


@app.get("/chunks/{job_id}")
def get_chunks(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Result not available.")
    if not job["chunks"]:
        raise HTTPException(404, "Chunking was not enabled for this job.")
    return {"chunks": job["chunks"], "count": len(job["chunks"])}


@app.get("/chunks/{job_id}/download")
def download_chunks(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done" or not job["chunks_path"]:
        raise HTTPException(404, "Chunks not available.")
    return FileResponse(
        job["chunks_path"],
        media_type="application/json",
        filename="chunks.json",
        headers={"Content-Disposition": "attachment; filename=chunks.json"},
    )


@app.get("/result/{job_id}")
def result(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Result not available.")
    page_count = job.get("page_count", 0)
    if page_count > 20:
        return {"content": "", "format": job["format"], "page_count": page_count}
    content = job["result_path"].read_text(encoding="utf-8")
    return {"content": content, "format": job["format"], "page_count": page_count}


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
