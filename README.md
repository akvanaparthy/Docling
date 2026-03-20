# Docling UI

A web-based wrapper around [Docling](https://github.com/DS4SD/docling) for converting PDFs (and other documents) to structured formats (JSON, Markdown, HTML, YAML, text) with a dark-themed UI.

## Features

- Single and multi-file upload (up to 20 files)
- URL-based document input
- Export to JSON, Markdown, HTML, YAML, DocTags, text
- Configurable pipelines: `standard`, `vlm`, `simple`
- OCR toggle with selectable backends
- Page range selection and batch processing
- Parallel multi-file conversion via `convert_all()`
- Reading order correction (spatial reorder)
- Document chunking (HybridChunker)
- Picture descriptions via VLM models
- Gemini enrichment
- Per-file timing metrics via SSE streaming
- Summary report generation (tables, figures, sections, pages)
- ZIP download for multi-file results
- Auto-cleanup of old jobs (1 hour TTL)

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Run

```bash
uvicorn app:app --reload --port 8000
```

Then open `http://localhost:8000`.

## Project Structure

```
app.py              # FastAPI backend (all endpoints + conversion logic)
bundler.py          # Standalone bundler utility
static/
  index.html        # Frontend HTML
  app.js            # Frontend JavaScript
  style.css         # Dark theme CSS
tests/
  conftest.py       # Pytest fixtures
  test_convert.py   # Conversion tests
  test_stream.py    # SSE stream tests
  test_result.py    # Result endpoint tests
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_UPLOAD_MB` | `200` | Max upload size in MB |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves the UI |
| `GET` | `/model-status` | CUDA/GPU availability check |
| `POST` | `/convert` | Start a conversion job |
| `GET` | `/stream/{job_id}` | SSE stream for progress/timing |
| `GET` | `/result/{job_id}` | Get conversion result |
| `GET` | `/result/{job_id}/{index}` | Get result for a specific file (multi) |
| `GET` | `/download/{job_id}` | Download result file or ZIP |
| `GET` | `/download/{job_id}/{index}` | Download specific file (multi) |
| `GET` | `/chunks/{job_id}` | Get chunks JSON |
| `GET` | `/chunks/{job_id}/download` | Download chunks file |
| `GET` | `/chunks/{job_id}/{index}` | Get chunks for specific file (multi) |
| `GET` | `/chunks/{job_id}/{index}/download` | Download chunks for specific file |
| `GET` | `/prompts/{job_id}/download` | Download prompts file |

## Tests

```bash
pytest
```
