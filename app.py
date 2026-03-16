import asyncio
import logging
import queue
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

import httpx
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
