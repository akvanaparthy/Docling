import asyncio
import json as _json
import logging
import queue
import shutil
import threading
import uuid
import zipfile
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
MAX_MULTI_FILES = 20

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
            if job["status"] in ("done", "error", "partial") and job["created_at"] < cutoff:
                shutil.rmtree(OUTPUTS / job_id, ignore_errors=True)
                shutil.rmtree(UPLOADS / job_id, ignore_errors=True)
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


def _merge_orphaned_list_descriptions(doc, y_tolerance: float = 3.0) -> None:
    """Find body-level text elements that spatially belong to list items and merge them.

    Detects text elements parented to body whose top-Y matches a list item's top-Y
    (within tolerance) and whose left-X is to the right of the list item. These are
    description texts that Docling incorrectly parsed as separate body elements
    instead of part of the list item.

    Merges the orphan's text and provenance into the matching list item, then deletes
    the orphan from the document tree.
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

    # Build a lookup: (page, rounded_top_y) -> list_item for all list items in list groups
    list_item_by_pos = {}
    for g in doc.groups:
        lbl = g.label.value if hasattr(g.label, 'value') else str(g.label)
        if lbl != 'list':
            continue
        for ch in g.children:
            item = ref_map.get(ch.cref)
            if item is None or not hasattr(item, 'prov') or not item.prov:
                continue
            pv = item.prov[0]
            if pv.bbox:
                key = (pv.page_no, round(pv.bbox.t, 0))
                list_item_by_pos[key] = (item, pv.bbox)

    if not list_item_by_pos:
        return

    # Find body-level text orphans that match a list item's Y position
    orphans_to_merge = []  # (orphan_item, matching_list_item)
    body_ref = doc.body.self_ref if hasattr(doc.body, 'self_ref') else '#/body'
    for t in doc.texts:
        parent_cref = t.parent.cref if hasattr(t, 'parent') and t.parent else None
        if parent_cref != body_ref:
            continue
        if not t.prov:
            continue
        pv = t.prov[0]
        if not pv.bbox:
            continue
        # Check for matching list item at same Y
        key = (pv.page_no, round(pv.bbox.t, 0))
        match = list_item_by_pos.get(key)
        if match is None:
            # Try nearby Y values within tolerance
            for dy in range(-int(y_tolerance), int(y_tolerance) + 1):
                alt_key = (pv.page_no, round(pv.bbox.t, 0) + dy)
                match = list_item_by_pos.get(alt_key)
                if match:
                    break
        if match is None:
            continue
        list_item, list_bbox = match
        # Orphan must be to the right of the list item
        if pv.bbox.l > list_bbox.r:
            orphans_to_merge.append((t, list_item))

    # Merge orphans into their matching list items
    for orphan, list_item in orphans_to_merge:
        # Merge text content
        list_item.text = list_item.text + " " + orphan.text
        if hasattr(list_item, 'orig') and hasattr(orphan, 'orig') and orphan.orig:
            list_item.orig = (list_item.orig or list_item.text) + " " + orphan.orig
        # Merge provenance (preserves both bboxes)
        list_item.prov.extend(orphan.prov)
        # Remove orphan from document
        doc.delete_items(node_items=[orphan])


def _reorder_body_children(doc) -> None:
    """Sort body.children in-place by (page ASC, t DESC, l ASC).

    Coords are BOTTOMLEFT origin: higher t = higher on page = comes first.
    Groups are atomic units — positioned by their first leaf element's bbox.
    Recurses through nested groups to find the first prov-bearing element.
    """
    # First, merge orphaned list descriptions into their matching list items
    try:
        _merge_orphaned_list_descriptions(doc)
    except Exception:
        pass  # non-critical, continue with reorder

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


def _reindex_json_reading_order(data: dict) -> dict:
    """Reorder texts/tables/pictures/groups arrays to match body.children
    reading order, and rewrite all #/type/N references throughout the document.

    Walks body tree depth-first to collect the canonical order for each array,
    then reindexes and rewrites every JSON-pointer reference in the document.
    Assumes body.children is already sorted (by _reorder_body_children).
    """
    ARRAY_KEYS = ['texts', 'tables', 'pictures', 'groups',
                  'key_value_items', 'form_items']

    # Build ref → item lookup
    ref_items = {}
    for key in ARRAY_KEYS:
        for item in data.get(key, []):
            sr = item.get('self_ref', '')
            if sr:
                ref_items[sr] = item

    # Walk tree depth-first to collect reading order per array
    order = {key: [] for key in ARRAY_KEYS}
    seen = set()

    def collect(ref: str):
        if not ref or ref in seen:
            return
        seen.add(ref)
        parts = ref.lstrip('#').lstrip('/').split('/')
        if len(parts) == 2 and parts[0] in order:
            try:
                order[parts[0]].append(int(parts[1]))
            except ValueError:
                pass
        item = ref_items.get(ref)
        if item:
            for child in item.get('children', []):
                collect(child.get('cref', ''))

    for child in data.get('body', {}).get('children', []):
        collect(child.get('cref', ''))
    for child in data.get('furniture', {}).get('children', []):
        collect(child.get('cref', ''))

    # Append orphans (not reachable from body/furniture) at end
    for key in ARRAY_KEYS:
        visited_set = set(order[key])
        for idx in range(len(data.get(key, []))):
            if idx not in visited_set:
                order[key].append(idx)

    # Build old→new ref mapping
    ref_map = {}
    for key in ARRAY_KEYS:
        for new_idx, old_idx in enumerate(order[key]):
            if old_idx != new_idx:
                ref_map[f'#/{key}/{old_idx}'] = f'#/{key}/{new_idx}'

    # Reorder arrays
    for key in ARRAY_KEYS:
        arr = data.get(key, [])
        if arr and order[key]:
            data[key] = [arr[old_idx] for old_idx in order[key]]

    # Rewrite all references recursively
    def rewrite(obj):
        if isinstance(obj, str):
            return ref_map.get(obj, obj)
        if isinstance(obj, dict):
            return {k: rewrite(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [rewrite(v) for v in obj]
        return obj

    return rewrite(data)


def _render_pdf_pages(source_path: str, page_nos: set, scale: float = 2.0) -> dict:
    """Render specific pages from a PDF using pypdfium2. Returns {page_no: PIL.Image}."""
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(source_path)
        result = {}
        for page_no in page_nos:
            idx = page_no - 1  # pypdfium2 is 0-indexed
            if 0 <= idx < len(pdf):
                bitmap = pdf[idx].render(scale=scale)
                result[page_no] = bitmap.to_pil()
        pdf.close()
        return result
    except Exception:
        return {}


def _crop_from_page(pil_page, bbox, page_size) -> "PIL.Image.Image | None":
    """Crop a region from a rendered page image using a Docling bbox."""
    try:
        page_w = page_size.width
        page_h = page_size.height
        img_w, img_h = pil_page.size
        sx = img_w / page_w
        sy = img_h / page_h

        l, t, r, b = bbox.l, bbox.t, bbox.r, bbox.b
        origin = str(getattr(bbox, "coord_origin", "")).upper()
        if "BOTTOMLEFT" in origin:
            # convert to top-left origin
            t, b = page_h - bbox.t, page_h - bbox.b

        left   = min(l, r) * sx
        right  = max(l, r) * sx
        top    = min(t, b) * sy
        bottom = max(t, b) * sy

        if right <= left or bottom <= top:
            return None
        return pil_page.crop((left, top, right, bottom))
    except Exception:
        return None


def _save_item_images(doc, out_dir: Path, source_path: str = None) -> dict:
    """Crop and save images for all pictures and tables.

    Pictures: uses pic.get_image(doc) which reads from the stored picture image.
    Tables:   re-renders the source PDF page via pypdfium2 and crops via bbox,
              since page.image is not persisted in the DoclingDocument after conversion.
    Returns {self_ref: relative_path_str} for items saved successfully.
    """
    import logging as _log
    _logger = _log.getLogger(__name__)

    pics_dir = out_dir / "pictures"
    tabs_dir = out_dir / "tables"
    pics_dir.mkdir(parents=True, exist_ok=True)
    tabs_dir.mkdir(parents=True, exist_ok=True)
    path_map = {}

    # ── Pictures (use stored image from Docling pipeline) ──────────────────────
    for pic in doc.pictures:
        try:
            img = pic.get_image(doc)
            if img:
                idx = pic.self_ref.rsplit("/", 1)[-1]
                fname = f"picture_{idx}.png"
                img.convert("RGB").save(pics_dir / fname, format="PNG")
                path_map[pic.self_ref] = f"pictures/{fname}"
        except Exception as e:
            _logger.warning("Failed to save picture %s: %s", pic.self_ref, e)

    # ── Tables (re-render PDF pages via pypdfium2 and crop by bbox) ────────────
    if not doc.tables:
        return path_map

    is_pdf = source_path and source_path.lower().endswith(".pdf")
    if not is_pdf:
        _logger.debug("Skipping table images — source is not a PDF")
        return path_map

    # Collect which pages we need
    pages_needed = set()
    for table in doc.tables:
        if table.prov:
            pages_needed.add(table.prov[0].page_no)

    page_images = _render_pdf_pages(source_path, pages_needed, scale=2.0)
    if not page_images:
        _logger.warning("Could not render PDF pages for table images")
        return path_map

    for table in doc.tables:
        try:
            if not table.prov:
                continue
            prov    = table.prov[0]
            page_no = prov.page_no
            pil_page = page_images.get(page_no)
            doc_page = doc.pages.get(page_no)
            if pil_page is None or doc_page is None or doc_page.size is None:
                continue
            img = _crop_from_page(pil_page, prov.bbox, doc_page.size)
            if img:
                idx = table.self_ref.rsplit("/", 1)[-1]
                fname = f"table_{idx}.png"
                img.convert("RGB").save(tabs_dir / fname, format="PNG")
                path_map[table.self_ref] = f"tables/{fname}"
        except Exception as e:
            _logger.warning("Failed to save table %s: %s", table.self_ref, e)

    return path_map


def _export_result(result, fmt: str, reorder: bool = False,
                   image_path_map: dict = None) -> str:
    return _export_result_from_doc(result.document, fmt, reorder=reorder,
                                   image_path_map=image_path_map)


def _export_result_from_doc(doc, fmt: str, reorder: bool = False,
                            image_path_map: dict = None) -> str:
    if fmt == "md":
        return doc.export_to_markdown()
    elif fmt == "html":
        import markdown as md_lib
        return md_lib.markdown(doc.export_to_markdown())
    elif fmt == "json":
        import json
        d = doc.model_dump(mode='json')
        if reorder:
            d = _reindex_json_reading_order(d)
        if image_path_map:
            for entry in d.get("pictures", []):
                ref = entry.get("self_ref")
                if ref in image_path_map:
                    entry["image_path"] = image_path_map[ref]
            for entry in d.get("tables", []):
                ref = entry.get("self_ref")
                if ref in image_path_map:
                    entry["image_path"] = image_path_map[ref]
        return json.dumps(d, indent=2)
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


def _build_converter(pipeline, ocr, pdf_backend, queue_max_size,
                     do_picture_description, pic_desc_model, vlm_model,
                     layout_batch_size, table_batch_size, ocr_batch_size,
                     table_mode, accelerator, send_info, send_timing,
                     gemini_enrich: bool = False, save_images: bool = False):
    """Build and return a DocumentConverter with the given options.

    Shared by _run_conversion and _run_multi_conversion to avoid duplicating
    the complex converter construction logic.
    Returns (converter, device).
    """
    import time

    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

    _backend_cls = PyPdfiumDocumentBackend if pdf_backend == "pypdfium" else DoclingParseV4DocumentBackend

    if accelerator == "auto":
        device = "cuda" if torch_cuda_available() else "cpu"
    else:
        device = accelerator

    def _pic_desc_options():
        from docling.datamodel.pipeline_options import PictureDescriptionVlmEngineOptions
        preset_id = PIC_DESC_PRESETS.get(pic_desc_model, PIC_DESC_PRESETS["smolvlm"])["preset_id"]
        return PictureDescriptionVlmEngineOptions.from_preset(preset_id)

    def _apply_pic_opts(base: dict) -> dict:
        if do_picture_description:
            base["picture_description_options"] = _pic_desc_options()
            base["generate_picture_images"] = True
        elif gemini_enrich or save_images:
            base["generate_picture_images"] = True
        return base

    # Build optional batch-size kwargs (0 = use Docling defaults)
    _batch_kw = {}
    if layout_batch_size > 0:
        _batch_kw["layout_batch_size"] = layout_batch_size
    if table_batch_size > 0:
        _batch_kw["table_batch_size"] = table_batch_size
    if ocr_batch_size > 0:
        _batch_kw["ocr_batch_size"] = ocr_batch_size

    # Accelerator device (auto / cpu / cuda)
    if accelerator != "auto":
        from docling.datamodel.pipeline_options import AcceleratorOptions, AcceleratorDevice
        _dev = AcceleratorDevice.CUDA if accelerator == "cuda" else AcceleratorDevice.CPU
        _batch_kw["accelerator_options"] = AcceleratorOptions(device=_dev)

    # Table structure mode (fast / accurate)
    if table_mode == "fast":
        from docling.datamodel.pipeline_options import TableStructureOptions, TableFormerMode
        _batch_kw["table_structure_options"] = TableStructureOptions(mode=TableFormerMode.FAST)
    elif table_mode == "accurate":
        from docling.datamodel.pipeline_options import TableStructureOptions, TableFormerMode
        _batch_kw["table_structure_options"] = TableStructureOptions(mode=TableFormerMode.ACCURATE)

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
                       "pic_desc_model": PIC_DESC_PRESETS.get(pic_desc_model, {}).get("label") if do_picture_description else None,
                       "gemini_enrich": gemini_enrich})
            converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_cls=VlmPipeline, backend=_backend_cls, pipeline_options=opts)}
            )
        except ImportError:
            pic_opts = _apply_pic_opts({"do_picture_description": do_picture_description})
            opts = PdfPipelineOptions(do_ocr=ocr, queue_max_size=queue_max_size, **_batch_kw, **pic_opts)
            send_info({"pipeline": "standard", "model": None, "device": device,
                       "do_picture_description": do_picture_description, "ocr": ocr,
                       "pic_desc_model": PIC_DESC_PRESETS.get(pic_desc_model, {}).get("label") if do_picture_description else None,
                       "gemini_enrich": gemini_enrich})
            converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(backend=_backend_cls, pipeline_options=opts)}
            )
    else:
        pic_opts = _apply_pic_opts({"do_picture_description": do_picture_description})
        opts = PdfPipelineOptions(do_ocr=ocr, queue_max_size=queue_max_size, **_batch_kw, **pic_opts)
        send_info({"pipeline": "standard", "model": None, "device": device,
                   "do_picture_description": do_picture_description, "ocr": ocr,
                   "pic_desc_model": PIC_DESC_PRESETS.get(pic_desc_model, {}).get("label") if do_picture_description else None})
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(backend=_backend_cls, pipeline_options=opts)}
        )
    send_timing({"stage": "pipeline_init", "duration": round(time.perf_counter() - t0, 2)})

    return converter, device


def _run_conversion(job_id: str, source: str, pipeline: str, ocr: bool, fmt: str,
                    do_picture_description: bool = False, pic_desc_model: str = "smolvlm",
                    vlm_model: str = "GRANITEDOCLING",
                    do_chunk: bool = False, chunk_max_tokens: int = 256,
                    page_from: int = 1, page_to: int = 0, pdf_backend: str = "docling",
                    queue_max_size: int = 100, batch_size: int = 0, reorder: bool = True,
                    layout_batch_size: int = 0, table_batch_size: int = 0, ocr_batch_size: int = 0,
                    table_mode: str = "accurate", accelerator: str = "auto",
                    free_vram: bool = False, gemini_enrich: bool = False,
                    save_images: bool = False):
    import time, json as _json2
    _thread_local.job_id = job_id
    job = jobs[job_id]
    job["status"] = "running"
    out_dir = OUTPUTS / job_id
    t_total_start = time.perf_counter()

    handler = _QueueHandler(job["queue"])
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    handler.addFilter(_JobFilter(job_id))
    docling_logger = logging.getLogger("docling")
    docling_logger.setLevel(logging.DEBUG)
    docling_logger.addHandler(handler)

    gemini_handler = _QueueHandler(job["queue"])
    gemini_handler.setFormatter(logging.Formatter("__GEMINI__:%(message)s"))
    gemini_handler.addFilter(_JobFilter(job_id))
    enricher_logger = logging.getLogger("enricher")
    enricher_logger.setLevel(logging.DEBUG)
    enricher_logger.addHandler(gemini_handler)

    def send_timing(data: dict):
        job["queue"].put(f"__TIMING__:{_json2.dumps(data)}")

    def send_info(data: dict):
        job["queue"].put(f"__INFO__:{_json2.dumps(data)}")

    def send_report(data: dict):
        job["queue"].put(f"__REPORT__:{_json2.dumps(data)}")

    try:
        from docling.datamodel.settings import settings
        settings.debug.profile_pipeline_timings = True

        converter, device = _build_converter(
            pipeline, ocr, pdf_backend, queue_max_size,
            do_picture_description, pic_desc_model, vlm_model,
            layout_batch_size, table_batch_size, ocr_batch_size,
            table_mode, accelerator, send_info, send_timing,
            gemini_enrich=gemini_enrich, save_images=save_images
        )

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
                t0 = time.perf_counter()
                report = _build_report(result.document)
                send_report(report)
                send_timing({"stage": "report_done", "duration": round(time.perf_counter() - t0, 2)})
            except Exception:
                pass  # report is non-critical

            if reorder:
                t0 = time.perf_counter()
                _reorder_body_children(result.document)
                send_timing({"stage": "reorder_done", "duration": round(time.perf_counter() - t0, 2)})

            # Run Gemini enrichment + image saving in parallel
            out_dir.mkdir(parents=True, exist_ok=True)
            image_path_map = {}
            if gemini_enrich:
                import json as _json_e
                from enricher import enrich as _gemini_enrich
                from concurrent.futures import ThreadPoolExecutor
                _fname = Path(source).name
                t0 = time.perf_counter()
                try:
                    with ThreadPoolExecutor(max_workers=2) as _pool:
                        _f_enrich = _pool.submit(_gemini_enrich, result.document, _fname)
                        _f_images = _pool.submit(_save_item_images, result.document, out_dir, source)
                        image_path_map = _f_images.result()
                        n_enriched, prompt_log, token_stats = _f_enrich.result()

                    n_pics = sum(1 for v in image_path_map.values() if "picture_" in v)
                    n_tabs = sum(1 for v in image_path_map.values() if "table_" in v)
                    _dur = round(time.perf_counter() - t0, 2)

                    prompts_path = out_dir / "prompts.json"
                    prompts_path.write_text(_json_e.dumps(prompt_log, indent=2), encoding="utf-8")
                    job["prompts_path"] = prompts_path

                    send_timing({
                        "stage": "gemini_enrich_done",
                        "duration": _dur,
                        "pictures": n_enriched,
                        "input_tokens": token_stats["input_tokens"],
                        "output_tokens": token_stats["output_tokens"],
                        "cost_usd": token_stats["cost_usd"],
                        "images_saved_pictures": n_pics,
                        "images_saved_tables": n_tabs,
                    })
                    job["queue"].put(
                        f"__GEMINI__:Done — {n_enriched} pic(s) in {_dur}s | "
                        f"in:{token_stats['input_tokens']} out:{token_stats['output_tokens']} "
                        f"tokens | cost: ${token_stats['cost_usd']:.6f}"
                    )
                    job["queue"].put(f"__GEMINI__:__PROMPTS_READY__")
                except Exception as _ee:
                    _dur = round(time.perf_counter() - t0, 2)
                    send_timing({"stage": "gemini_enrich_done", "duration": _dur, "error": str(_ee)})
                    job["queue"].put(f"__GEMINI__:ERROR after {_dur}s: {_ee}")
            elif save_images:
                # Save images without Gemini
                t0 = time.perf_counter()
                image_path_map = _save_item_images(result.document, out_dir, source)
                n_pics = sum(1 for v in image_path_map.values() if "picture_" in v)
                n_tabs = sum(1 for v in image_path_map.values() if "table_" in v)
                send_timing({"stage": "images_saved", "duration": round(time.perf_counter() - t0, 2),
                             "pictures": n_pics, "tables": n_tabs})

            t0 = time.perf_counter()
            text = _export_result(result, fmt, reorder=reorder, image_path_map=image_path_map)
            send_timing({"stage": "export_done", "duration": round(time.perf_counter() - t0, 2)})
            all_docs = [result.document] if do_chunk else []
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = EXT_MAP[fmt]
        out_path = out_dir / f"result{ext}"
        t0 = time.perf_counter()
        out_path.write_text(text, encoding="utf-8")
        send_timing({"stage": "file_write_done", "duration": round(time.perf_counter() - t0, 2),
                     "size_kb": round(len(text.encode("utf-8")) / 1024, 1)})

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
        if free_vram and torch_cuda_available():
            import gc, torch
            # Delete converter and its cached pipelines to release model tensors
            try:
                converter.initialized_pipelines.clear()
            except Exception:
                pass
            try:
                del converter
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
        docling_logger.removeHandler(handler)
        enricher_logger.removeHandler(gemini_handler)
        job["queue"].put(None)  # sentinel


def _run_multi_conversion(job_id, sources, filenames, pipeline, ocr, fmt,
                          do_picture_description, pic_desc_model, vlm_model,
                          do_chunk, chunk_max_tokens, pdf_backend, queue_max_size,
                          reorder, layout_batch_size, table_batch_size, ocr_batch_size,
                          table_mode, accelerator, doc_concurrency, doc_batch_size_setting,
                          free_vram: bool = False, gemini_enrich: bool = False,
                          save_images: bool = False):
    """Convert multiple files in parallel using converter.convert_all()."""
    import time
    import json as _json2

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

    def send_file_status(data: dict):
        job["queue"].put(f"__FILE_STATUS__:{_json2.dumps(data)}")

    try:
        from docling.datamodel.settings import settings
        from docling.datamodel.base_models import ConversionStatus

        settings.debug.profile_pipeline_timings = True
        settings.perf.doc_batch_size = doc_batch_size_setting
        settings.perf.doc_batch_concurrency = doc_concurrency

        # Send initial pending status for all files
        for i, fname in enumerate(filenames):
            send_file_status({"file_index": i, "file_name": fname, "status": "pending"})

        converter, device = _build_converter(
            pipeline, ocr, pdf_backend, queue_max_size,
            do_picture_description, pic_desc_model, vlm_model,
            layout_batch_size, table_batch_size, ocr_batch_size,
            table_mode, accelerator, send_info, send_timing,
            gemini_enrich=gemini_enrich, save_images=save_images
        )

        send_info({"multi": True, "file_count": len(sources),
                    "doc_concurrency": doc_concurrency,
                    "doc_batch_size": doc_batch_size_setting})

        out_dir = OUTPUTS / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = EXT_MAP[fmt]

        # Build a mapping from source path to file index
        source_to_idx = {}
        for i, src in enumerate(sources):
            source_to_idx[str(src)] = i

        success_count = 0
        fail_count = 0

        for conv_result in converter.convert_all(sources, raises_on_error=False):
            # Determine which file this result corresponds to
            input_path = str(conv_result.input.file) if conv_result.input and conv_result.input.file else ""
            # Try to match by path
            file_idx = None
            for src_path, idx in source_to_idx.items():
                if input_path.endswith(Path(src_path).name) or input_path == src_path:
                    file_idx = idx
                    break
            if file_idx is None:
                # Fallback: assign by order of successful results
                file_idx = success_count + fail_count

            fname = filenames[file_idx] if file_idx < len(filenames) else f"file_{file_idx}"

            if conv_result.status == ConversionStatus.SUCCESS:
                t0 = time.perf_counter()

                # Get page count
                page_count = len(conv_result.document.pages) if conv_result.document.pages else 0

                # Timings
                timings = {}
                for key, item in (conv_result.timings or {}).items():
                    timings[key] = {
                        "total": round(item.total(), 2),
                        "avg": round(float(item.avg()), 2) if item.count > 0 else 0,
                        "count": item.count,
                        "scope": item.scope.value,
                    }

                send_timing({
                    "stage": "conversion_done",
                    "file_index": file_idx,
                    "file_name": fname,
                    "duration": round(time.perf_counter() - t0, 2),
                    "page_count": page_count,
                    "timings": timings,
                })

                # Reorder if enabled
                if reorder:
                    _reorder_body_children(conv_result.document)

                # Build report
                try:
                    report = _build_report(conv_result.document)
                    send_report({"file_index": file_idx, "file_name": fname, "report": report})
                except Exception:
                    pass

                # Export
                text = _export_result(conv_result, fmt, reorder=reorder)
                result_path = out_dir / f"result_{file_idx}{ext}"
                result_path.write_text(text, encoding="utf-8")

                # Chunking per file
                file_chunks = None
                file_chunks_path = None
                if do_chunk:
                    from docling.chunking import HybridChunker
                    import json as _json3
                    chunker = HybridChunker(max_tokens=chunk_max_tokens)
                    chunks_display = []
                    chunks_full = []
                    chunk_idx = 0
                    for chunk in chunker.chunk(conv_result.document):
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
                    file_chunks = chunks_display
                    file_chunks_path = out_dir / f"chunks_{file_idx}.json"
                    file_chunks_path.write_text(_json3.dumps(chunks_full, indent=2), encoding="utf-8")

                # Update job file entry
                job["files"][file_idx]["status"] = "done"
                job["files"][file_idx]["page_count"] = page_count
                job["files"][file_idx]["result_path"] = str(result_path)
                job["files"][file_idx]["chunks"] = file_chunks
                job["files"][file_idx]["chunks_path"] = str(file_chunks_path) if file_chunks_path else None

                send_file_status({
                    "file_index": file_idx,
                    "file_name": fname,
                    "status": "done",
                    "page_count": page_count,
                })
                success_count += 1

            else:
                # FAILURE or PARTIAL_SUCCESS
                error_msg = str(conv_result.status.value) if hasattr(conv_result.status, 'value') else str(conv_result.status)
                job["files"][file_idx]["status"] = "error"
                job["files"][file_idx]["error"] = error_msg

                send_file_status({
                    "file_index": file_idx,
                    "file_name": fname,
                    "status": "error",
                    "error": error_msg,
                })
                fail_count += 1

        # Generate ZIP of all successful results
        zip_path = out_dir / "results.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for i, finfo in enumerate(job["files"]):
                if finfo["status"] == "done" and finfo.get("result_path"):
                    rp = Path(finfo["result_path"])
                    if rp.exists():
                        # Use original filename with new extension
                        stem = Path(finfo["name"]).stem
                        zf.write(rp, f"{stem}{ext}")
                    # Also add chunks if present
                    if finfo.get("chunks_path"):
                        cp = Path(finfo["chunks_path"])
                        if cp.exists():
                            stem = Path(finfo["name"]).stem
                            zf.write(cp, f"{stem}_chunks.json")

        job["zip_path"] = str(zip_path)

        # Determine final job status
        send_timing({"stage": "total", "duration": round(time.perf_counter() - t_total_start, 2)})

        if fail_count == 0:
            job["status"] = "done"
        elif success_count == 0:
            job["status"] = "error"
            job["error"] = f"All {fail_count} files failed conversion."
        else:
            job["status"] = "partial"

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
    finally:
        if free_vram and torch_cuda_available():
            import gc, torch
            try:
                converter.initialized_pipelines.clear()
            except Exception:
                pass
            try:
                del converter
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
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
            texts.append(_export_result(result, fmt, reorder=reorder))

    src_pdf.close()
    job["page_count"] = total_page_count

    if need_merged_doc and batch_docs:
        # Merge all batch documents into one with correct page numbers and refs
        t0 = time.perf_counter()
        merged_doc = DoclingDocument.concatenate(batch_docs)
        send_timing({"stage": "merge_done", "duration": round(time.perf_counter() - t0, 2)})
        if reorder:
            t0 = time.perf_counter()
            _reorder_body_children(merged_doc)
            send_timing({"stage": "reorder_done", "duration": round(time.perf_counter() - t0, 2)})
        t0 = time.perf_counter()
        text = _export_result_from_doc(merged_doc, fmt, reorder=reorder)
        send_timing({"stage": "export_done", "duration": round(time.perf_counter() - t0, 2)})
        all_docs = [merged_doc]
    else:
        separator = "\n\n" if fmt in ("md", "doctags") else "\n"
        text = separator.join(texts)
        all_docs = []

    # Build report from merged doc or first batch
    if send_report:
        try:
            t0 = time.perf_counter()
            report_doc = all_docs[0] if all_docs else (batch_docs[0] if batch_docs else None)
            if report_doc:
                send_report(_build_report(report_doc))
                send_timing({"stage": "report_done", "duration": round(time.perf_counter() - t0, 2)})
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
        "prompts_path": None,
        "page_count": 0,
        "error": None,
        "created_at": datetime.utcnow(),
        "multi": False,
        "files": [],
        "zip_path": None,
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
    files: list[UploadFile] = [],
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
    layout_batch_size: int = Form(default=0),
    table_batch_size: int = Form(default=0),
    ocr_batch_size: int = Form(default=0),
    table_mode: str = Form(default="accurate"),
    accelerator: str = Form(default="auto"),
    free_vram: bool = Form(default=False),
    gemini_enrich: bool = Form(default=False),
    save_images: bool = Form(default=False),
    doc_concurrency: int = Form(default=4),
    doc_batch_size_setting: int = Form(default=4),
):
    # ---- Multi-file path ----
    if len(files) > 1:
        if len(files) > MAX_MULTI_FILES:
            raise HTTPException(400, f"Maximum {MAX_MULTI_FILES} files allowed.")
        if pipeline == "vlm" and not torch_cuda_available():
            raise HTTPException(400, "VLM pipeline requires a CUDA GPU. Switch to standard pipeline.")

        job_id, job = _make_job(format)
        job["multi"] = True
        jobs[job_id] = job

        upload_dir = UPLOADS / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        total_size = 0
        saved_sources = []
        saved_names = []
        file_entries = []

        for i, f in enumerate(files):
            content = await f.read()
            total_size += len(content)
            if total_size > MAX_BYTES:
                shutil.rmtree(upload_dir, ignore_errors=True)
                del jobs[job_id]
                raise HTTPException(413, f"Total file size exceeds {MAX_UPLOAD_MB}MB limit.")

            fname = f.filename or f"file_{i}.bin"
            fpath = upload_dir / f"{i}_{fname}"
            fpath.write_bytes(content)
            saved_sources.append(str(fpath))
            saved_names.append(fname)
            file_entries.append({
                "name": fname,
                "status": "pending",
                "page_count": 0,
                "result_path": None,
                "chunks": None,
                "chunks_path": None,
                "error": None,
            })

        job["files"] = file_entries

        executor.submit(
            _run_multi_conversion, job_id, saved_sources, saved_names,
            pipeline, ocr, format, do_picture_description, pic_desc_model,
            vlm_model, do_chunk, chunk_max_tokens, pdf_backend, queue_max_size,
            reorder, layout_batch_size, table_batch_size, ocr_batch_size,
            table_mode, accelerator, doc_concurrency, doc_batch_size_setting,
            free_vram, gemini_enrich, save_images
        )
        return {"job_id": job_id}

    # ---- Single-file / URL path (unchanged logic) ----
    # If a single file came via the `files` list, treat it as the `file` param
    if len(files) == 1 and file is None:
        file = files[0]

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

    executor.submit(_run_conversion, job_id, source, pipeline, ocr, format, do_picture_description, pic_desc_model, vlm_model, do_chunk, chunk_max_tokens, page_from, page_to, pdf_backend, queue_max_size, batch_size, reorder, layout_batch_size, table_batch_size, ocr_batch_size, table_mode, accelerator, free_vram, gemini_enrich, save_images)
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
                if job["status"] in ("done", "error", "partial"):
                    # queue fully drained, job already finished before client connected
                    if job["status"] == "error":
                        yield f"event: error\ndata: {job['error']}\n\n"
                    else:
                        done_payload = {"format": job["format"]}
                        if job["multi"]:
                            done_payload["multi"] = True
                            done_payload["file_count"] = len(job["files"])
                        yield f"event: done\ndata: {_json.dumps(done_payload)}\n\n"
                    return
                continue
            else:
                if msg is None:  # sentinel
                    if job["status"] == "error":
                        yield f"event: error\ndata: {job['error']}\n\n"
                    else:
                        done_payload = {"format": job["format"]}
                        if job["multi"]:
                            done_payload["multi"] = True
                            done_payload["file_count"] = len(job["files"])
                        yield f"event: done\ndata: {_json.dumps(done_payload)}\n\n"
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
                elif safe_msg.startswith("__FILE_STATUS__:"):
                    yield f"event: file_status\ndata: {safe_msg[16:]}\n\n"
                elif safe_msg.startswith("__GEMINI__:"):
                    yield f"event: gemini_log\ndata: {safe_msg[11:]}\n\n"
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
    if not job or job["status"] not in ("done", "partial"):
        raise HTTPException(404, "Result not available.")
    if not job["chunks"]:
        raise HTTPException(404, "Chunking was not enabled for this job.")
    return {"chunks": job["chunks"], "count": len(job["chunks"])}


@app.get("/chunks/{job_id}/download")
def download_chunks(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] not in ("done", "partial") or not job["chunks_path"]:
        raise HTTPException(404, "Chunks not available.")
    return FileResponse(
        job["chunks_path"],
        media_type="application/json",
        filename="chunks.json",
        headers={"Content-Disposition": "attachment; filename=chunks.json"},
    )


@app.get("/prompts/{job_id}/download")
def download_prompts(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] not in ("done", "partial") or not job.get("prompts_path"):
        raise HTTPException(404, "Prompt log not available.")
    return FileResponse(
        job["prompts_path"],
        media_type="application/json",
        filename="prompts.json",
        headers={"Content-Disposition": "attachment; filename=prompts.json"},
    )


@app.get("/chunks/{job_id}/{index}")
def get_chunks_by_index(job_id: str, index: int):
    """Return chunks for a specific file in a multi-file job."""
    job = jobs.get(job_id)
    if not job or job["status"] not in ("done", "partial"):
        raise HTTPException(404, "Result not available.")
    if not job["multi"]:
        raise HTTPException(400, "Not a multi-file job. Use /chunks/{job_id} instead.")
    if index < 0 or index >= len(job["files"]):
        raise HTTPException(404, f"File index {index} out of range.")
    finfo = job["files"][index]
    if finfo["status"] != "done":
        raise HTTPException(404, f"File {index} did not complete successfully.")
    if not finfo.get("chunks"):
        raise HTTPException(404, "Chunking was not enabled or no chunks for this file.")
    return {"chunks": finfo["chunks"], "count": len(finfo["chunks"]), "name": finfo["name"]}


@app.get("/chunks/{job_id}/{index}/download")
def download_chunks_by_index(job_id: str, index: int):
    """Download chunks JSON for a specific file in a multi-file job."""
    job = jobs.get(job_id)
    if not job or job["status"] not in ("done", "partial"):
        raise HTTPException(404, "Result not available.")
    if not job["multi"]:
        raise HTTPException(400, "Not a multi-file job. Use /chunks/{job_id}/download instead.")
    if index < 0 or index >= len(job["files"]):
        raise HTTPException(404, f"File index {index} out of range.")
    finfo = job["files"][index]
    if finfo["status"] != "done" or not finfo.get("chunks_path"):
        raise HTTPException(404, "Chunks not available for this file.")
    cp = Path(finfo["chunks_path"])
    if not cp.exists():
        raise HTTPException(404, "Chunks file not found on disk.")
    stem = Path(finfo["name"]).stem
    fname = f"{stem}_chunks.json"
    return FileResponse(
        cp,
        media_type="application/json",
        filename=fname,
        headers={"Content-Disposition": f"attachment; filename={fname}"},
    )


@app.get("/result/{job_id}")
def result(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] not in ("done", "partial"):
        raise HTTPException(404, "Result not available.")

    if job["multi"]:
        file_summaries = []
        for finfo in job["files"]:
            file_summaries.append({
                "name": finfo["name"],
                "status": finfo["status"],
                "format": job["format"],
                "page_count": finfo.get("page_count", 0),
            })
        return {"multi": True, "files": file_summaries, "format": job["format"]}

    page_count = job.get("page_count", 0)
    if page_count > 20:
        return {"content": "", "format": job["format"], "page_count": page_count}
    content = job["result_path"].read_text(encoding="utf-8")
    return {"content": content, "format": job["format"], "page_count": page_count}


@app.get("/result/{job_id}/{index}")
def result_by_index(job_id: str, index: int):
    """Return individual file content for a multi-file job."""
    job = jobs.get(job_id)
    if not job or job["status"] not in ("done", "partial"):
        raise HTTPException(404, "Result not available.")
    if not job["multi"]:
        raise HTTPException(400, "Not a multi-file job. Use /result/{job_id} instead.")
    if index < 0 or index >= len(job["files"]):
        raise HTTPException(404, f"File index {index} out of range.")
    finfo = job["files"][index]
    if finfo["status"] != "done":
        raise HTTPException(404, f"File {index} did not complete successfully.")
    rp = Path(finfo["result_path"])
    if not rp.exists():
        raise HTTPException(404, "Result file not found on disk.")
    page_count = finfo.get("page_count", 0)
    content = rp.read_text(encoding="utf-8")
    return {
        "content": content,
        "format": job["format"],
        "page_count": page_count,
        "name": finfo["name"],
    }


@app.get("/download/{job_id}")
def download(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] not in ("done", "partial"):
        raise HTTPException(404, "Result not available.")

    if job["multi"]:
        zip_path = job.get("zip_path")
        if not zip_path or not Path(zip_path).exists():
            raise HTTPException(404, "ZIP file not available.")
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename="results.zip",
            headers={"Content-Disposition": "attachment; filename=results.zip"},
        )

    path = job["result_path"]
    mime = MIME_MAP.get(path.suffix, "text/plain")
    return FileResponse(
        path,
        media_type=mime,
        filename=f"result{path.suffix}",
        headers={"Content-Disposition": f"attachment; filename=result{path.suffix}"},
    )


@app.get("/download/{job_id}/{index}")
def download_by_index(job_id: str, index: int):
    """Download individual file result by index in a multi-file job."""
    job = jobs.get(job_id)
    if not job or job["status"] not in ("done", "partial"):
        raise HTTPException(404, "Result not available.")
    if not job["multi"]:
        raise HTTPException(400, "Not a multi-file job. Use /download/{job_id} instead.")
    if index < 0 or index >= len(job["files"]):
        raise HTTPException(404, f"File index {index} out of range.")
    finfo = job["files"][index]
    if finfo["status"] != "done":
        raise HTTPException(404, f"File {index} did not complete successfully.")
    rp = Path(finfo["result_path"])
    if not rp.exists():
        raise HTTPException(404, "Result file not found on disk.")
    ext = EXT_MAP[job["format"]]
    mime = MIME_MAP.get(ext, "text/plain")
    stem = Path(finfo["name"]).stem
    fname = f"{stem}{ext}"
    return FileResponse(
        rp,
        media_type=mime,
        filename=fname,
        headers={"Content-Disposition": f"attachment; filename={fname}"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
