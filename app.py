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


def _build_report(doc) -> dict:
    """Extract all countable metadata from a DoclingDocument."""
    from collections import Counter

    report = {}

    # --- Document overview ---
    overview = {}
    if doc.origin:
        overview["filename"] = doc.origin.filename or "—"
        overview["mimetype"] = doc.origin.mimetype or "—"
    overview["pages"] = len(doc.pages) if doc.pages else 0
    # Page dimensions
    if doc.pages:
        sizes = set()
        for p in doc.pages.values():
            if p.size:
                sizes.add(f"{p.size.width:.0f}×{p.size.height:.0f}")
        overview["page_dimensions"] = ", ".join(sorted(sizes)) if sizes else "—"
        # Pages with rendered image
        pages_with_image = sum(1 for p in doc.pages.values() if p.image is not None)
        if pages_with_image:
            overview["pages_with_image"] = pages_with_image
    report["overview"] = overview

    # --- Collect all elements for counting ---
    all_items = []  # (label_str, item)
    label_counts = Counter()

    for t in doc.texts:
        lbl = t.label.value if hasattr(t.label, 'value') else str(t.label)
        label_counts[lbl] += 1
        all_items.append((lbl, t))
    for p in doc.pictures:
        lbl = p.label.value if hasattr(p.label, 'value') else str(p.label)
        label_counts[lbl] += 1
        all_items.append((lbl, p))
    for tb in doc.tables:
        lbl = tb.label.value if hasattr(tb.label, 'value') else str(tb.label)
        label_counts[lbl] += 1
        all_items.append((lbl, tb))
    for kv in doc.key_value_items:
        lbl = kv.label.value if hasattr(kv.label, 'value') else str(kv.label)
        label_counts[lbl] += 1
        all_items.append((lbl, kv))
    for fi in doc.form_items:
        lbl = fi.label.value if hasattr(fi.label, 'value') else str(fi.label)
        label_counts[lbl] += 1
        all_items.append((lbl, fi))
    for fr in getattr(doc, 'field_regions', []):
        lbl = fr.label.value if hasattr(fr.label, 'value') else str(fr.label)
        label_counts[lbl] += 1
        all_items.append((lbl, fr))
    for fitem in getattr(doc, 'field_items', []):
        lbl = fitem.label.value if hasattr(fitem.label, 'value') else str(fitem.label)
        label_counts[lbl] += 1
        all_items.append((lbl, fitem))

    total_elements = len(all_items)
    report["total_elements"] = total_elements

    # Elements by label (sorted by count desc, only non-zero)
    report["elements_by_label"] = dict(label_counts.most_common())

    # --- Structure ---
    structure = {}
    with_bbox = 0
    with_parent = 0
    parent_ids = set()
    content_layers = Counter()
    for _lbl, item in all_items:
        if hasattr(item, 'prov') and item.prov:
            for pv in item.prov:
                if pv.bbox:
                    with_bbox += 1
                    break
        if hasattr(item, 'parent') and item.parent:
            with_parent += 1
            parent_ids.add(item.parent.cref if hasattr(item.parent, 'cref') else str(item.parent))
        if hasattr(item, 'content_layer') and item.content_layer:
            cl = item.content_layer.value if hasattr(item.content_layer, 'value') else str(item.content_layer)
            content_layers[cl] += 1

    structure["with_bbox"] = f"{with_bbox} / {total_elements}"
    structure["with_parent"] = f"{with_parent} / {total_elements}"
    structure["unique_parents"] = len(parent_ids)
    if content_layers:
        structure["content_layers"] = dict(content_layers.most_common())
    # Groups
    if doc.groups:
        group_labels = Counter()
        for g in doc.groups:
            gl = g.label.value if hasattr(g.label, 'value') else str(g.label)
            group_labels[gl] += 1
        structure["groups"] = dict(group_labels.most_common())
        structure["groups_total"] = len(doc.groups)
    report["structure"] = structure

    # --- Section header levels ---
    level_counts = Counter()
    for t in doc.texts:
        lbl = t.label.value if hasattr(t.label, 'value') else str(t.label)
        if lbl == "section_header" and hasattr(t, 'level'):
            level_counts[f"L{t.level}"] += 1
    if level_counts:
        report["heading_levels"] = dict(sorted(level_counts.items()))

    # --- Text formatting ---
    fmt_counts = {"bold": 0, "italic": 0, "underline": 0, "strikethrough": 0, "hyperlinks": 0}
    for t in doc.texts:
        if hasattr(t, 'formatting') and t.formatting:
            if t.formatting.bold: fmt_counts["bold"] += 1
            if t.formatting.italic: fmt_counts["italic"] += 1
            if t.formatting.underline: fmt_counts["underline"] += 1
            if t.formatting.strikethrough: fmt_counts["strikethrough"] += 1
        if hasattr(t, 'hyperlink') and t.hyperlink:
            fmt_counts["hyperlinks"] += 1
    # Only include non-zero
    fmt_counts = {k: v for k, v in fmt_counts.items() if v > 0}
    if fmt_counts:
        report["text_formatting"] = fmt_counts

    # --- List items ---
    enum_count = 0
    bullet_count = 0
    for t in doc.texts:
        lbl = t.label.value if hasattr(t.label, 'value') else str(t.label)
        if lbl == "list_item" and hasattr(t, 'enumerated'):
            if t.enumerated:
                enum_count += 1
            else:
                bullet_count += 1
    if enum_count or bullet_count:
        report["list_items"] = {"enumerated": enum_count, "bulleted": bullet_count}

    # --- Tables detail ---
    if doc.tables:
        total_cells = 0
        header_cells = 0
        row_header_cells = 0
        merged_cells = 0
        fillable_cells = 0
        table_sizes = []
        tables_with_caption = 0
        for tb in doc.tables:
            if hasattr(tb, 'captions') and tb.captions:
                tables_with_caption += 1
            if tb.data:
                table_sizes.append((tb.data.num_rows, tb.data.num_cols))
                for cell in tb.data.table_cells:
                    total_cells += 1
                    if cell.column_header:
                        header_cells += 1
                    if cell.row_header:
                        row_header_cells += 1
                    if cell.row_span > 1 or cell.col_span > 1:
                        merged_cells += 1
                    if cell.fillable:
                        fillable_cells += 1
        td = {"total_cells": total_cells}
        if header_cells: td["column_header_cells"] = header_cells
        if row_header_cells: td["row_header_cells"] = row_header_cells
        if merged_cells: td["merged_cells"] = merged_cells
        if fillable_cells: td["fillable_cells"] = fillable_cells
        if tables_with_caption: td["with_caption"] = tables_with_caption
        if table_sizes:
            avg_r = sum(r for r, c in table_sizes) / len(table_sizes)
            avg_c = sum(c for r, c in table_sizes) / len(table_sizes)
            td["avg_size"] = f"{avg_r:.1f}×{avg_c:.1f}"
        report["tables_detail"] = td

    # --- Pictures detail ---
    if doc.pictures:
        pd = {}
        with_image = sum(1 for p in doc.pictures if p.image is not None)
        with_caption = sum(1 for p in doc.pictures if hasattr(p, 'captions') and p.captions)
        with_desc = 0
        with_classification = 0
        class_labels = Counter()
        for p in doc.pictures:
            if hasattr(p, 'meta') and p.meta:
                if hasattr(p.meta, 'description') and p.meta.description:
                    with_desc += 1
                if hasattr(p.meta, 'classification') and p.meta.classification:
                    with_classification += 1
                    for pred in p.meta.classification.predictions:
                        class_labels[pred.class_name] += 1
        pd["with_image"] = f"{with_image} / {len(doc.pictures)}"
        if with_caption: pd["with_caption"] = with_caption
        if with_desc: pd["with_description"] = with_desc
        if with_classification: pd["with_classification"] = with_classification
        if class_labels:
            pd["classification_labels"] = dict(class_labels.most_common())
        report["pictures_detail"] = pd

    # --- Code languages ---
    code_langs = Counter()
    for t in doc.texts:
        lbl = t.label.value if hasattr(t.label, 'value') else str(t.label)
        if lbl == "code" and hasattr(t, 'code_language') and t.code_language:
            cl = t.code_language.value if hasattr(t.code_language, 'value') else str(t.code_language)
            code_langs[cl] += 1
    if code_langs:
        report["code_languages"] = dict(code_langs.most_common())

    # --- Forms / Key-value ---
    if doc.key_value_items:
        total_kv_cells = sum(len(kv.graph.cells) for kv in doc.key_value_items if kv.graph)
        report["key_value_detail"] = {"regions": len(doc.key_value_items), "cells": total_kv_cells}
    if doc.form_items:
        total_form_cells = sum(len(f.graph.cells) for f in doc.form_items if f.graph)
        report["form_detail"] = {"forms": len(doc.form_items), "cells": total_form_cells}

    # --- Pages coverage ---
    pages_with_content = Counter()
    for _lbl, item in all_items:
        if hasattr(item, 'prov') and item.prov:
            for pv in item.prov:
                pages_with_content[pv.page_no] += 1
    if pages_with_content:
        total_pages = len(doc.pages) if doc.pages else 0
        avg_per_page = total_elements / total_pages if total_pages else 0
        report["pages_coverage"] = {
            "pages_with_content": f"{len(pages_with_content)} / {total_pages}",
            "avg_elements_per_page": round(avg_per_page, 1),
        }

    return report


def _reorder_body_children(doc) -> None:
    """Sort body.children in-place by (page ASC, t DESC, l ASC).

    Coords are BOTTOMLEFT origin: higher t = higher on page = comes first.
    Groups are atomic units — positioned by their first leaf element's bbox.
    Recurses through nested groups to find the first prov-bearing element.
    """
    ref_map = {}
    for lst in [doc.texts, doc.tables, doc.pictures,
                doc.key_value_items, doc.form_items,
                getattr(doc, 'field_items', []), getattr(doc, 'field_regions', [])]:
        for item in lst:
            if hasattr(item, 'self_ref'):
                ref_map[item.self_ref] = item
    for g in doc.groups:
        ref_map[g.self_ref] = g

    def first_prov(cref: str):
        item = ref_map.get(cref)
        if item is None:
            return None
        if hasattr(item, 'prov') and item.prov:
            return item.prov[0]
        if hasattr(item, 'children') and item.children:
            return first_prov(item.children[0].cref)
        return None

    def sort_key(child):
        prov = first_prov(child.cref)
        if prov is None:
            return (float('inf'), 0.0, 0.0)
        page = prov.page_no if prov.page_no is not None else float('inf')
        bbox = prov.bbox
        t = bbox.t if bbox else 0.0
        l = bbox.l if bbox else 0.0
        return (page, -t, l)  # page asc, top-to-bottom (-t asc), left-to-right (l asc)

    doc.body.children.sort(key=sort_key)


def _export_result(result, fmt: str) -> str:
    return _export_result_from_doc(result.document, fmt)


def _export_result_from_doc(doc, fmt: str) -> str:
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
                    page_from: int = 1, page_to: int = 0, pdf_backend: str = "docling",
                    queue_max_size: int = 100, batch_size: int = 0, reorder: bool = True):
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

    def send_report(data: dict):
        job["queue"].put(f"__REPORT__:{_json2.dumps(data)}")

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
                opts = PdfPipelineOptions(do_ocr=ocr, queue_max_size=queue_max_size, **pic_opts)
                send_info({"pipeline": "standard", "model": None, "device": device,
                           "do_picture_description": do_picture_description, "ocr": ocr,
                           "pic_desc_model": PIC_DESC_PRESETS.get(pic_desc_model, {}).get("label") if do_picture_description else None})
                converter = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(backend=_backend_cls, pipeline_options=opts)}
                )
        else:
            pic_opts = _apply_pic_opts({"do_picture_description": do_picture_description})
            opts = PdfPipelineOptions(do_ocr=ocr, queue_max_size=queue_max_size, **pic_opts)
            send_info({"pipeline": "standard", "model": None, "device": device,
                       "do_picture_description": do_picture_description, "ocr": ocr,
                       "pic_desc_model": PIC_DESC_PRESETS.get(pic_desc_model, {}).get("label") if do_picture_description else None})
            converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(backend=_backend_cls, pipeline_options=opts)}
            )
        send_timing({"stage": "pipeline_init", "duration": round(time.perf_counter() - t0, 2)})

        # Determine if we should use batched conversion
        is_pdf = source.lower().endswith(".pdf")
        use_batching = is_pdf and batch_size > 0

        if use_batching:
            text, page_count, all_timings, all_docs = _run_batched_conversion(
                source, converter, fmt, batch_size, page_from, page_to,
                send_timing, job, do_chunk, send_report, reorder=reorder
            )
        else:
            import sys as _sys
            p_range = (page_from, page_to if page_to > 0 else _sys.maxsize)
            t0 = time.perf_counter()
            result = converter.convert(source, page_range=p_range)
            conversion_time = round(time.perf_counter() - t0, 2)

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

            try:
                report = _build_report(result.document)
                send_report(report)
            except Exception:
                pass  # report is non-critical

            t0 = time.perf_counter()
            if reorder:
                _reorder_body_children(result.document)
            text = _export_result(result, fmt)
            send_timing({"stage": "export_done", "duration": round(time.perf_counter() - t0, 2)})
            all_docs = [result.document] if do_chunk else []

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
            chunk_idx = 0
            for doc in all_docs:
                for chunk in chunker.chunk(doc):
                    page = None
                    if chunk.meta.doc_items:
                        prov = getattr(chunk.meta.doc_items[0], 'prov', None)
                        if prov:
                            page = prov[0].page_no
                    chunks_display.append({
                        "index": chunk_idx,
                        "text": chunk.text,
                        "headings": chunk.meta.headings or [],
                        "page": page,
                    })
                    chunks_full.append(chunk.model_dump(mode='json'))
                    chunk_idx += 1
            job["chunks"] = chunks_display
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


def _run_batched_conversion(source, converter, fmt, batch_size, page_from, page_to,
                            send_timing, job, do_chunk, send_report=None, reorder=True):
    """Split PDF into batches using pypdfium2, convert each as a separate document.

    For JSON format and chunking, uses DoclingDocument.concatenate() to merge
    batch documents with correct page numbers and internal references.
    For MD/HTML/DocTags, concatenates exported text strings directly.
    """
    import time
    from io import BytesIO
    import pypdfium2 as pdfium
    from docling.datamodel.document import DocumentStream, DoclingDocument

    src_pdf = pdfium.PdfDocument(source)
    total_pages = len(src_pdf)

    # Apply user page range
    effective_from = max(page_from, 1) - 1  # 0-indexed
    effective_to = min(page_to, total_pages) if page_to > 0 else total_pages
    pages_to_process = list(range(effective_from, effective_to))

    # We need the full document for JSON export and chunking
    need_merged_doc = fmt == "json" or do_chunk

    texts = []
    batch_docs = []
    total_page_count = 0
    t0_all = time.perf_counter()

    for batch_start in range(0, len(pages_to_process), batch_size):
        batch_pages = pages_to_process[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        real_from = batch_pages[0] + 1  # back to 1-indexed for display
        real_to = batch_pages[-1] + 1

        job["queue"].put(f"__INFO_BATCH__:Processing batch {batch_num}: pages {real_from}-{real_to}")

        # Create a sub-PDF with just these pages
        new_pdf = pdfium.PdfDocument.new()
        new_pdf.import_pages(src_pdf, batch_pages)
        buf = BytesIO()
        new_pdf.save(buf)
        new_pdf.close()
        buf.seek(0)

        stream = DocumentStream(name=f"batch_{real_from}_{real_to}.pdf", stream=buf)

        t0 = time.perf_counter()
        result = converter.convert(stream)
        batch_time = round(time.perf_counter() - t0, 2)

        batch_page_count = len(result.document.pages) if result.document.pages else 0
        total_page_count += batch_page_count

        send_timing({
            "stage": "conversion_done",
            "duration": batch_time,
            "page_count": batch_page_count,
            "batch": f"{real_from}-{real_to}",
            "timings": {},
        })

        if need_merged_doc:
            batch_docs.append(result.document)
        else:
            if reorder:
                _reorder_body_children(result.document)
            texts.append(_export_result(result, fmt))

    src_pdf.close()
    job["page_count"] = total_page_count

    t0 = time.perf_counter()
    if need_merged_doc and batch_docs:
        # Merge all batch documents into one with correct page numbers and refs
        merged_doc = DoclingDocument.concatenate(batch_docs)
        if reorder:
            _reorder_body_children(merged_doc)
        text = _export_result_from_doc(merged_doc, fmt)
        all_docs = [merged_doc]
    else:
        separator = "\n\n" if fmt in ("md", "doctags") else "\n"
        text = separator.join(texts)
        all_docs = []
    send_timing({"stage": "export_done", "duration": round(time.perf_counter() - t0, 2)})

    # Build report from merged doc or first batch
    if send_report:
        try:
            report_doc = all_docs[0] if all_docs else (batch_docs[0] if batch_docs else None)
            if report_doc:
                send_report(_build_report(report_doc))
        except Exception:
            pass

    return text, total_page_count, {}, all_docs


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
    queue_max_size: int = Form(default=100),
    batch_size: int = Form(default=0),
    reorder: bool = Form(default=True),
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

    executor.submit(_run_conversion, job_id, source, pipeline, ocr, format, do_picture_description, pic_desc_model, vlm_model, do_chunk, chunk_max_tokens, page_from, page_to, pdf_backend, queue_max_size, batch_size, reorder)
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
                elif safe_msg.startswith("__REPORT__:"):
                    yield f"event: report\ndata: {safe_msg[11:]}\n\n"
                elif safe_msg.startswith("__INFO_BATCH__:"):
                    yield f"event: log\ndata: {safe_msg[15:]}\n\n"
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
