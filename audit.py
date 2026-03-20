"""
audit.py — SQLite audit trail for Docling pipeline.

Creates and populates 12 tables that track every element, page, bundle,
section, table cell, visual asset, furniture item, timing metric, and error
produced during document conversion.

Two integration patterns:
  1. INLINE — called during pipeline stages (bundle status, timing, errors)
  2. POST-HOC — called after result.json is written, walks the JSON and
     bulk-inserts all content tables in one pass.
"""

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


# ── Schema ──────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
-- 1. Documents: parent record for entire PDF
CREATE TABLE IF NOT EXISTS documents (
    doc_id              TEXT PRIMARY KEY,
    filename            TEXT,
    cloud_storage_link  TEXT,
    total_pages         INTEGER,
    schema_version      TEXT,
    pipeline            TEXT,
    ocr_enabled         INTEGER,
    ocr_engine          TEXT,
    accelerator         TEXT,
    reorder_applied     INTEGER,
    master_summary      TEXT,
    total_elements      INTEGER,
    total_bundles       INTEGER,
    processing_duration REAL,
    status              TEXT DEFAULT 'RUNNING',
    created_at          TEXT
);

-- 2. Page Registry: physical page dimensions and metadata
CREATE TABLE IF NOT EXISTS page_registry (
    uuid                TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL REFERENCES documents(doc_id),
    physical_pg_num     INTEGER NOT NULL,
    logical_pg_num      TEXT,
    page_width          REAL,
    page_height         REAL,
    bundle_process_id   TEXT,
    element_count       INTEGER DEFAULT 0,
    furniture_count     INTEGER DEFAULT 0,
    has_tables          INTEGER DEFAULT 0,
    has_pictures        INTEGER DEFAULT 0,
    thumbnail_url       TEXT,
    full_page_url       TEXT,
    UNIQUE(doc_id, physical_pg_num)
);

-- 3. Knowledge Skeleton: section headers / document structure
CREATE TABLE IF NOT EXISTS knowledge_skeleton (
    section_id          TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL REFERENCES documents(doc_id),
    self_ref            TEXT NOT NULL,
    hierarchy_path      TEXT,
    hierarchy_title     TEXT,
    level               INTEGER,
    text                TEXT,
    start_physical_pg   INTEGER,
    end_physical_pg     INTEGER,
    bundle_process_id   TEXT,
    parent_section_id   TEXT REFERENCES knowledge_skeleton(section_id),
    child_count         INTEGER DEFAULT 0,
    section_summary     TEXT,
    UNIQUE(doc_id, self_ref)
);

-- 4. Extraction Element: every content element (texts, tables, pictures, groups)
CREATE TABLE IF NOT EXISTS extraction_element (
    element_id          TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL REFERENCES documents(doc_id),
    self_ref            TEXT NOT NULL,
    parent_ref          TEXT,
    element_type        TEXT,
    label               TEXT,
    content_layer       TEXT,
    content_markdown    TEXT,
    content_json        TEXT,
    orig                TEXT,
    bbox_l              REAL,
    bbox_t              REAL,
    bbox_r              REAL,
    bbox_b              REAL,
    bbox_coord_origin   TEXT,
    physical_pg_num     INTEGER,
    charspan_start      INTEGER,
    charspan_end        INTEGER,
    section_id          TEXT REFERENCES knowledge_skeleton(section_id),
    bundle_process_id   TEXT,
    body_order_index    INTEGER,
    hyperlink           TEXT,
    formatting          TEXT,
    is_furniture        INTEGER DEFAULT 0,
    is_fragment         INTEGER DEFAULT 0,
    continuation_ref    TEXT,
    has_children        INTEGER DEFAULT 0,
    child_refs          TEXT,
    description         TEXT,
    UNIQUE(doc_id, self_ref)
);

-- 5. Element Provenance: multi-prov entries (elements with >1 bbox)
CREATE TABLE IF NOT EXISTS element_provenance (
    prov_id             TEXT PRIMARY KEY,
    element_id          TEXT NOT NULL REFERENCES extraction_element(element_id),
    doc_id              TEXT NOT NULL REFERENCES documents(doc_id),
    prov_index          INTEGER,
    page_no             INTEGER,
    bbox_l              REAL,
    bbox_t              REAL,
    bbox_r              REAL,
    bbox_b              REAL,
    bbox_coord_origin   TEXT,
    charspan_start      INTEGER,
    charspan_end        INTEGER
);

-- 6. Table Cell: cell-level data for tables
CREATE TABLE IF NOT EXISTS table_cell (
    cell_id             TEXT PRIMARY KEY,
    element_id          TEXT NOT NULL REFERENCES extraction_element(element_id),
    doc_id              TEXT NOT NULL REFERENCES documents(doc_id),
    row_index           INTEGER,
    col_index           INTEGER,
    row_span            INTEGER,
    col_span            INTEGER,
    end_row_index       INTEGER,
    end_col_index       INTEGER,
    text                TEXT,
    is_column_header    INTEGER DEFAULT 0,
    is_row_header       INTEGER DEFAULT 0,
    is_row_section      INTEGER DEFAULT 0,
    is_fillable         INTEGER DEFAULT 0,
    bbox_l              REAL,
    bbox_t              REAL,
    bbox_r              REAL,
    bbox_b              REAL
);

-- 7. Visual Asset Registry: pictures with dedup and descriptions
CREATE TABLE IF NOT EXISTS visual_asset_registry (
    asset_id            TEXT PRIMARY KEY,
    element_id          TEXT NOT NULL REFERENCES extraction_element(element_id),
    doc_id              TEXT NOT NULL REFERENCES documents(doc_id),
    physical_pg_num     INTEGER,
    image_hash          TEXT,
    image_path          TEXT,
    vlm_description     TEXT,
    is_global           INTEGER DEFAULT 0,
    caption_refs        TEXT,
    child_text_refs     TEXT,
    bbox_l              REAL,
    bbox_t              REAL,
    bbox_r              REAL,
    bbox_b              REAL,
    occurrence_count    INTEGER DEFAULT 1
);

-- 8. Furniture Registry: page headers/footers with variant detection
CREATE TABLE IF NOT EXISTS furniture_registry (
    furniture_id        TEXT PRIMARY KEY,
    element_id          TEXT NOT NULL REFERENCES extraction_element(element_id),
    doc_id              TEXT NOT NULL REFERENCES documents(doc_id),
    furniture_type      TEXT,
    text                TEXT,
    text_hash           TEXT,
    physical_pg_num     INTEGER,
    bundle_process_id   TEXT,
    is_variant          INTEGER DEFAULT 0,
    variant_group       INTEGER
);

-- 8b. Merged Tables: groups of split tables that share column headers
CREATE TABLE IF NOT EXISTS merged_tables (
    merge_id            TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL REFERENCES documents(doc_id),
    column_headers      TEXT,
    num_cols            INTEGER,
    total_rows          INTEGER,
    page_start          INTEGER,
    page_end            INTEGER,
    member_count        INTEGER,
    member_refs         TEXT,
    table_summary       TEXT
);

-- 8c. Link individual tables to their merged group
CREATE TABLE IF NOT EXISTS merged_table_members (
    id                  TEXT PRIMARY KEY,
    merge_id            TEXT NOT NULL REFERENCES merged_tables(merge_id),
    element_id          TEXT NOT NULL REFERENCES extraction_element(element_id),
    doc_id              TEXT NOT NULL REFERENCES documents(doc_id),
    self_ref            TEXT,
    member_index        INTEGER,
    page_no             INTEGER,
    num_rows            INTEGER
);

-- 9. Bundle Process: per-bundle conversion tracking
CREATE TABLE IF NOT EXISTS bundle_process (
    process_id          TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL REFERENCES documents(doc_id),
    bundle_index        INTEGER,
    bundle_plan_id      TEXT,
    bundle_name         TEXT,
    toc_level           INTEGER,
    parent_section      TEXT,
    page_start          INTEGER,
    page_end            INTEGER,
    is_continuation     INTEGER DEFAULT 0,
    continuation_of     TEXT,
    status              TEXT DEFAULT 'PENDING',
    duration            REAL,
    page_count          INTEGER,
    model_load_time     REAL,
    peak_memory_mb      REAL,
    error               TEXT,
    json_path           TEXT,
    child_pid           INTEGER,
    started_at          TEXT,
    completed_at        TEXT,
    UNIQUE(doc_id, bundle_index)
);

-- 10. Process Audit: bundle lifecycle transitions
CREATE TABLE IF NOT EXISTS process_audit (
    audit_id            TEXT PRIMARY KEY,
    process_id          TEXT NOT NULL REFERENCES bundle_process(process_id),
    doc_id              TEXT NOT NULL REFERENCES documents(doc_id),
    status              TEXT,
    started_at          TEXT,
    duration            REAL,
    retry_count         INTEGER DEFAULT 0,
    worker_host         TEXT,
    gpu_utilization     REAL,
    cpu_utilization     REAL
);

-- 11. Error Log: pipeline errors at any level
CREATE TABLE IF NOT EXISTS error_log (
    error_id            TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL REFERENCES documents(doc_id),
    error_area          TEXT,
    related_process_id  TEXT,
    related_element_ref TEXT,
    related_page_num    INTEGER,
    error_description   TEXT,
    error_type          TEXT,
    stack_trace         TEXT,
    is_recoverable      INTEGER DEFAULT 1,
    occurred_at         TEXT
);

-- 12. Timing Metrics: granular stage timings
CREATE TABLE IF NOT EXISTS timing_metrics (
    metric_id           TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL REFERENCES documents(doc_id),
    process_id          TEXT,
    stage               TEXT,
    duration            REAL,
    page_count          INTEGER,
    bundle_index        INTEGER,
    metadata            TEXT,
    recorded_at         TEXT
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_element_doc_page ON extraction_element(doc_id, physical_pg_num);
CREATE INDEX IF NOT EXISTS idx_element_doc_label ON extraction_element(doc_id, label);
CREATE INDEX IF NOT EXISTS idx_element_doc_layer ON extraction_element(doc_id, content_layer);
CREATE INDEX IF NOT EXISTS idx_element_doc_order ON extraction_element(doc_id, body_order_index);
CREATE INDEX IF NOT EXISTS idx_element_section ON extraction_element(section_id);
CREATE INDEX IF NOT EXISTS idx_prov_element ON element_provenance(element_id);
CREATE INDEX IF NOT EXISTS idx_cell_element ON table_cell(element_id);
CREATE INDEX IF NOT EXISTS idx_visual_element ON visual_asset_registry(element_id);
CREATE INDEX IF NOT EXISTS idx_visual_hash ON visual_asset_registry(doc_id, image_hash);
CREATE INDEX IF NOT EXISTS idx_furniture_doc ON furniture_registry(doc_id, furniture_type);
CREATE INDEX IF NOT EXISTS idx_bundle_doc ON bundle_process(doc_id);
CREATE INDEX IF NOT EXISTS idx_timing_doc ON timing_metrics(doc_id, stage);
CREATE INDEX IF NOT EXISTS idx_error_doc ON error_log(doc_id);
CREATE INDEX IF NOT EXISTS idx_page_doc ON page_registry(doc_id, physical_pg_num);
CREATE INDEX IF NOT EXISTS idx_merged_doc ON merged_tables(doc_id);
CREATE INDEX IF NOT EXISTS idx_merged_member ON merged_table_members(merge_id);
"""


def _uid():
    return str(uuid.uuid4())


def _now():
    return datetime.now(timezone.utc).isoformat()


def _table_cells_to_markdown(data: dict) -> str | None:
    """Convert table_cells from JSON into a markdown table string.

    Builds a pipe-delimited markdown table from the cell grid so that
    extraction_element.content has a readable/queryable representation.
    """
    if not data:
        return None
    cells = data.get("table_cells", [])
    num_rows = data.get("num_rows", 0)
    num_cols = data.get("num_cols", 0)
    if not cells or num_rows == 0 or num_cols == 0:
        return None

    # Build grid
    grid = [["" for _ in range(num_cols)] for _ in range(num_rows)]
    header_rows = set()
    for cell in cells:
        r = cell.get("start_row_offset_idx", 0)
        c = cell.get("start_col_offset_idx", 0)
        if 0 <= r < num_rows and 0 <= c < num_cols:
            grid[r][c] = (cell.get("text") or "").replace("|", "\\|")
        if cell.get("column_header"):
            header_rows.add(r)

    lines = []
    for r in range(num_rows):
        line = "| " + " | ".join(grid[r]) + " |"
        lines.append(line)
        # Add separator after header row(s)
        if r in header_rows and (r + 1 >= num_rows or r + 1 not in header_rows):
            sep = "| " + " | ".join("---" for _ in range(num_cols)) + " |"
            lines.append(sep)

    return "\n".join(lines) if lines else None


# ── Database Init ───────────────────────────────────────────────────────────

def init_db(db_path: str) -> sqlite3.Connection:
    """Create audit database and all tables. Returns open connection."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


# ── INLINE: called during pipeline ──────────────────────────────────────────

def insert_document(conn: sqlite3.Connection, doc_id: str, filename: str,
                    pipeline: str = None, ocr_enabled: bool = False,
                    ocr_engine: str = None, accelerator: str = None,
                    reorder_applied: bool = False):
    """Stage 1: Insert initial document record when job starts."""
    conn.execute(
        """INSERT OR IGNORE INTO documents
           (doc_id, filename, pipeline, ocr_enabled, ocr_engine, accelerator,
            reorder_applied, status, created_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (doc_id, filename, pipeline, int(ocr_enabled), ocr_engine,
         accelerator, int(reorder_applied), "RUNNING", _now())
    )
    conn.commit()


def insert_bundles(conn: sqlite3.Connection, doc_id: str, bundles: list):
    """Stage 2: Insert all bundle rows with status=PENDING after plan_bundles().

    Args:
        bundles: list of Bundle objects (from bundler.py) with attrs:
                 id, name, page_start, page_end, toc_level, parent_section,
                 is_continuation, continuation_of
    """
    rows = []
    for i, b in enumerate(bundles):
        process_id = f"{doc_id}_bundle_{i:03d}"
        rows.append((
            process_id, doc_id, i, b.id, b.name, b.toc_level,
            b.parent_section, b.page_start, b.page_end,
            int(b.is_continuation), b.continuation_of,
            "PENDING", None, None, None, None, None, None, None, _now(), None
        ))
    conn.executemany(
        """INSERT OR IGNORE INTO bundle_process
           (process_id, doc_id, bundle_index, bundle_plan_id, bundle_name,
            toc_level, parent_section, page_start, page_end, is_continuation,
            continuation_of, status, duration, page_count, model_load_time,
            peak_memory_mb, error, json_path, child_pid, started_at, completed_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows
    )
    conn.commit()

    # Also insert initial PENDING audit entries
    audit_rows = []
    for i, b in enumerate(bundles):
        process_id = f"{doc_id}_bundle_{i:03d}"
        audit_rows.append((_uid(), process_id, doc_id, "PENDING", _now(), None, 0, None, None, None))
    conn.executemany(
        """INSERT INTO process_audit
           (audit_id, process_id, doc_id, status, started_at, duration,
            retry_count, worker_host, gpu_utilization, cpu_utilization)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        audit_rows
    )
    conn.commit()


def update_bundle_status(conn: sqlite3.Connection, doc_id: str,
                         bundle_index: int, status: str,
                         duration: float = None, page_count: int = None,
                         model_load_time: float = None, error: str = None,
                         child_pid: int = None, json_path: str = None):
    """Stage 3: Update bundle row when conversion starts/completes/fails."""
    process_id = f"{doc_id}_bundle_{bundle_index:03d}"
    now = _now()

    updates = ["status = ?"]
    params = [status]

    if duration is not None:
        updates.append("duration = ?")
        params.append(duration)
    if page_count is not None:
        updates.append("page_count = ?")
        params.append(page_count)
    if model_load_time is not None:
        updates.append("model_load_time = ?")
        params.append(model_load_time)
    if error is not None:
        updates.append("error = ?")
        params.append(error)
    if child_pid is not None:
        updates.append("child_pid = ?")
        params.append(child_pid)
    if json_path is not None:
        updates.append("json_path = ?")
        params.append(json_path)

    if status in ("DONE", "ERROR"):
        updates.append("completed_at = ?")
        params.append(now)

    params.append(process_id)
    conn.execute(
        f"UPDATE bundle_process SET {', '.join(updates)} WHERE process_id = ?",
        params
    )

    # Audit trail entry
    conn.execute(
        """INSERT INTO process_audit
           (audit_id, process_id, doc_id, status, started_at, duration,
            retry_count, worker_host, gpu_utilization, cpu_utilization)
           VALUES (?,?,?,?,?,?,0,NULL,NULL,NULL)""",
        (_uid(), process_id, doc_id, status, now, duration)
    )
    conn.commit()


def record_timing(conn: sqlite3.Connection, doc_id: str, timing_data: dict):
    """Inline: persist a timing event. Called from send_timing() wrapper."""
    stage = timing_data.get("stage", "unknown")
    bundle_index = timing_data.get("bundle_index")
    process_id = f"{doc_id}_bundle_{bundle_index:03d}" if bundle_index is not None else None

    conn.execute(
        """INSERT INTO timing_metrics
           (metric_id, doc_id, process_id, stage, duration, page_count,
            bundle_index, metadata, recorded_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (_uid(), doc_id, process_id, stage,
         timing_data.get("duration"), timing_data.get("page_count"),
         bundle_index, json.dumps(timing_data), _now())
    )
    conn.commit()


def record_error(conn: sqlite3.Connection, doc_id: str, error_area: str,
                 error: Exception, bundle_index: int = None,
                 element_ref: str = None, page_num: int = None,
                 is_recoverable: bool = True):
    """Inline: record a pipeline error."""
    import traceback
    process_id = f"{doc_id}_bundle_{bundle_index:03d}" if bundle_index is not None else None

    conn.execute(
        """INSERT INTO error_log
           (error_id, doc_id, error_area, related_process_id,
            related_element_ref, related_page_num, error_description,
            error_type, stack_trace, is_recoverable, occurred_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (_uid(), doc_id, error_area, process_id, element_ref, page_num,
         str(error), type(error).__name__, traceback.format_exc(),
         int(is_recoverable), _now())
    )
    conn.commit()


# ── POST-HOC: walk result.json and populate content tables ──────────────────

def populate_from_json(conn: sqlite3.Connection, doc_id: str,
                       json_path: str, job_meta: dict = None):
    """Walk the final result.json and bulk-insert all content tables.

    This is called ONCE after the JSON file is written to disk (Stage 9).
    It populates: documents (update), page_registry, knowledge_skeleton,
    extraction_element, element_provenance, table_cell, visual_asset_registry,
    furniture_registry.

    Args:
        conn: open SQLite connection
        doc_id: job UUID
        json_path: path to result.json
        job_meta: dict with keys like pipeline, ocr, accelerator, etc.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    job_meta = job_meta or {}

    # ── Update Documents row with final counts ──────────────────────────
    pages = data.get("pages", {})
    texts = data.get("texts", [])
    tables = data.get("tables", [])
    pictures = data.get("pictures", [])
    groups = data.get("groups", [])
    bundle_plan = data.get("bundle_plan", [])
    total_elements = len(texts) + len(tables) + len(pictures) + len(groups)

    conn.execute(
        """UPDATE documents SET
           total_pages = ?, schema_version = ?, total_elements = ?,
           total_bundles = ?, status = ?
           WHERE doc_id = ?""",
        (len(pages), data.get("version"), total_elements,
         len(bundle_plan), "COMPLETED", doc_id)
    )

    # ── Build lookup maps ───────────────────────────────────────────────

    # body.children order → body_order_index
    body_order = {}
    for idx, child in enumerate(data.get("body", {}).get("children", [])):
        cref = child.get("cref", "")
        if cref:
            body_order[cref] = idx

    # page_no → bundle_plan_id (from bundle_plan page ranges)
    page_to_bundle = {}
    for bp in bundle_plan:
        for pg in range(bp["page_start"], bp["page_end"] + 1):
            page_to_bundle[pg] = bp["id"]

    # bundle_plan_id → process_id (for FK lookups)
    bundle_to_process = {}
    for i, bp in enumerate(bundle_plan):
        bundle_to_process[bp["id"]] = f"{doc_id}_bundle_{i:03d}"

    # ── Page Registry ───────────────────────────────────────────────────
    # _logical_page_map: physical_pg_num (str) → logical string like "1-22"
    logical_page_map = data.get("_logical_page_map", {})

    page_rows = []
    for pg_key, pg_data in pages.items():
        pg_no = pg_data.get("page_no", int(pg_key))
        size = pg_data.get("size", {})
        bid = page_to_bundle.get(pg_no)
        pid = bundle_to_process.get(bid) if bid else None
        logical_pg = logical_page_map.get(str(pg_no))
        page_rows.append((
            _uid(), doc_id, pg_no, logical_pg,
            size.get("width"), size.get("height"),
            pid, 0, 0, 0, 0, None, None
        ))
    conn.executemany(
        """INSERT OR IGNORE INTO page_registry
           (uuid, doc_id, physical_pg_num, logical_pg_num, page_width, page_height,
            bundle_process_id, element_count, furniture_count,
            has_tables, has_pictures, thumbnail_url, full_page_url)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        page_rows
    )

    # ── Walk all elements ───────────────────────────────────────────────
    # We'll collect rows for bulk insert
    element_rows = []
    prov_rows = []
    cell_rows = []
    visual_rows = []
    furniture_rows = []
    section_rows = []

    # Track current section for FK assignment
    # (walk body.children in order to assign section_id)
    current_section_id = None

    # Build ordered list of body children refs for section tracking
    body_children_refs = [
        child.get("cref", "")
        for child in data.get("body", {}).get("children", [])
    ]

    # Build ref → item lookup for all arrays
    all_items = {}
    for arr_name in ("texts", "tables", "pictures", "groups"):
        for item in data.get(arr_name, []):
            sr = item.get("self_ref", "")
            if sr:
                all_items[sr] = (arr_name, item)

    # First pass: identify section headers and build section map
    section_map = {}  # self_ref → section_id
    sections_ordered = []  # for end_physical_pg computation
    for cref in body_children_refs:
        if cref not in all_items:
            continue
        arr_name, item = all_items[cref]
        if item.get("label") == "section_header":
            sid = _uid()
            section_map[cref] = sid
            prov = item.get("prov", [])
            pg = prov[0]["page_no"] if prov else None
            bundle_info = item.get("bundle", {})
            bid = bundle_info.get("id")
            pid = bundle_to_process.get(bid) if bid else None

            sections_ordered.append({
                "section_id": sid,
                "self_ref": cref,
                "text": item.get("text", ""),
                "hierarchy_path": item.get("hierarchy_path", ""),
                "hierarchy_title": item.get("hierarchy_title", ""),
                "level": item.get("level", 1),
                "start_pg": pg,
                "bundle_process_id": pid,
            })

    # Compute end_physical_pg for each section (= start of next section - 1)
    for i, sec in enumerate(sections_ordered):
        if i + 1 < len(sections_ordered):
            next_pg = sections_ordered[i + 1]["start_pg"]
            sec["end_pg"] = next_pg if next_pg else sec["start_pg"]
        else:
            sec["end_pg"] = len(pages)  # last section goes to end

    # Insert knowledge_skeleton rows
    for sec in sections_ordered:
        section_rows.append((
            sec["section_id"], doc_id, sec["self_ref"],
            sec["hierarchy_path"], sec["hierarchy_title"],
            sec["level"], sec["text"],
            sec["start_pg"], sec["end_pg"],
            sec["bundle_process_id"], None, 0, None
        ))

    conn.executemany(
        """INSERT OR IGNORE INTO knowledge_skeleton
           (section_id, doc_id, self_ref, hierarchy_path, hierarchy_title,
            level, text, start_physical_pg, end_physical_pg,
            bundle_process_id, parent_section_id, child_count, section_summary)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        section_rows
    )

    # Second pass: walk body.children in order, track current section,
    # insert all elements
    current_section_id = None

    # Page counters for page_registry updates
    page_element_count = {}
    page_furniture_count = {}
    page_has_tables = set()
    page_has_pictures = set()

    def _process_element(arr_name, item, order_idx=None):
        nonlocal current_section_id

        sr = item.get("self_ref", "")
        if not sr:
            return

        etype_map = {"texts": "text", "tables": "table",
                     "pictures": "picture", "groups": "group"}
        element_type = etype_map.get(arr_name, arr_name)

        label = item.get("label", "")
        content_layer = item.get("content_layer", "body")
        prov = item.get("prov", [])
        bbox = prov[0].get("bbox", {}) if prov else {}
        page_no = prov[0].get("page_no") if prov else None
        charspan = prov[0].get("charspan", [None, None]) if prov else [None, None]

        # Track current section
        if label == "section_header" and sr in section_map:
            current_section_id = section_map[sr]

        bundle_info = item.get("bundle", {})
        bid = bundle_info.get("id")
        pid = bundle_to_process.get(bid) if bid else None

        parent = item.get("parent", {})
        parent_ref = parent.get("cref", "") if parent else ""

        children = item.get("children", [])
        child_ref_list = [c.get("cref", "") for c in children if c.get("cref")]
        has_children = 1 if child_ref_list else 0

        is_furniture = 1 if content_layer == "furniture" else 0
        is_multi_prov = 1 if len(prov) > 1 else 0

        # content_markdown: readable text/markdown representation
        # content_json: raw JSON data (tables only — the full data dict)
        content_markdown = item.get("text")
        content_json = None
        orig = item.get("orig")
        if element_type == "table":
            table_data = item.get("data", {})
            if not content_markdown:
                content_markdown = _table_cells_to_markdown(table_data)
            if table_data:
                content_json = json.dumps(table_data)

        # Extract Gemini/VLM description from meta.description.text
        # (written by enricher.py for pictures and tables)
        item_meta = item.get("meta") or {}
        desc_field = item_meta.get("description") or {}
        description = desc_field.get("text") or None

        eid = _uid()

        element_rows.append((
            eid, doc_id, sr, parent_ref, element_type, label,
            content_layer, content_markdown, content_json, orig,
            bbox.get("l"), bbox.get("t"), bbox.get("r"), bbox.get("b"),
            bbox.get("coord_origin"),
            page_no, charspan[0] if charspan else None,
            charspan[1] if len(charspan) > 1 else None,
            current_section_id, pid, order_idx,
            item.get("hyperlink"), item.get("formatting"),
            is_furniture, is_multi_prov, None, has_children,
            json.dumps(child_ref_list) if child_ref_list else None,
            description
        ))

        # Page counters
        if page_no is not None:
            if is_furniture:
                page_furniture_count[page_no] = page_furniture_count.get(page_no, 0) + 1
            else:
                page_element_count[page_no] = page_element_count.get(page_no, 0) + 1
            if element_type == "table":
                page_has_tables.add(page_no)
            if element_type == "picture":
                page_has_pictures.add(page_no)

        # Element Provenance (for multi-prov elements)
        if len(prov) > 1:
            for pidx, p in enumerate(prov):
                pb = p.get("bbox", {})
                pcs = p.get("charspan", [None, None])
                prov_rows.append((
                    _uid(), eid, doc_id, pidx, p.get("page_no"),
                    pb.get("l"), pb.get("t"), pb.get("r"), pb.get("b"),
                    pb.get("coord_origin"),
                    pcs[0] if pcs else None,
                    pcs[1] if len(pcs) > 1 else None
                ))

        # Table Cells
        if element_type == "table" and item.get("data"):
            for cell in item["data"].get("table_cells", []):
                cb = cell.get("bbox", {})
                cell_rows.append((
                    _uid(), eid, doc_id,
                    cell.get("start_row_offset_idx"),
                    cell.get("start_col_offset_idx"),
                    cell.get("row_span"), cell.get("col_span"),
                    cell.get("end_row_offset_idx"),
                    cell.get("end_col_offset_idx"),
                    cell.get("text"),
                    int(cell.get("column_header", False)),
                    int(cell.get("row_header", False)),
                    int(cell.get("row_section", False)),
                    int(cell.get("fillable", False)),
                    cb.get("l"), cb.get("t"), cb.get("r"), cb.get("b")
                ))

        # Visual Asset (pictures)
        if element_type == "picture":
            captions = [c.get("cref", "") for c in item.get("captions", []) if c.get("cref")]
            img_path = item.get("image_path")

            # Extract description from meta.description.text (written by enricher.py)
            # This is the Gemini/VLM description — stored as vlm_description
            item_meta_va = item.get("meta") or {}
            desc_field_va = item_meta_va.get("description") or {}
            vlm_desc = desc_field_va.get("text") or None

            # Fallback: check annotations (older format)
            if not vlm_desc:
                for ann in item.get("annotations", []):
                    if isinstance(ann, dict) and ann.get("text"):
                        vlm_desc = ann["text"]
                        break

            visual_rows.append((
                _uid(), eid, doc_id, page_no,
                img_path,  # image_path
                vlm_desc,
                0,  # is_global — computed later
                json.dumps(captions) if captions else None,
                json.dumps(child_ref_list) if child_ref_list else None,
                bbox.get("l"), bbox.get("t"), bbox.get("r"), bbox.get("b"),
                1  # occurrence_count default
            ))

        # Furniture Registry (page headers/footers)
        if label in ("page_header", "page_footer"):
            txt = content_markdown or ""
            text_hash = hashlib.md5(txt.encode()).hexdigest()
            furniture_rows.append((
                _uid(), eid, doc_id, label, txt, text_hash,
                page_no, pid, 0, None  # is_variant and variant_group computed later
            ))

    # Walk body.children in reading order
    for order_idx, cref in enumerate(body_children_refs):
        if cref in all_items:
            arr_name, item = all_items[cref]
            _process_element(arr_name, item, order_idx)

            # Also process child elements (group children, picture children)
            for child in item.get("children", []):
                child_ref = child.get("cref", "")
                if child_ref in all_items:
                    c_arr, c_item = all_items[child_ref]
                    _process_element(c_arr, c_item, order_idx=None)

    # Catch any elements NOT in body.children (orphans, nested group children)
    processed_refs = set()
    for row in element_rows:
        processed_refs.add(row[2])  # self_ref is index 2

    for sr, (arr_name, item) in all_items.items():
        if sr not in processed_refs:
            _process_element(arr_name, item, order_idx=None)

    # ── Bulk inserts ────────────────────────────────────────────────────
    conn.executemany(
        """INSERT OR IGNORE INTO extraction_element
           (element_id, doc_id, self_ref, parent_ref, element_type, label,
            content_layer, content_markdown, content_json, orig,
            bbox_l, bbox_t, bbox_r, bbox_b,
            bbox_coord_origin, physical_pg_num, charspan_start, charspan_end,
            section_id, bundle_process_id, body_order_index,
            hyperlink, formatting, is_furniture, is_fragment,
            continuation_ref, has_children, child_refs, description)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        element_rows
    )

    conn.executemany(
        """INSERT OR IGNORE INTO element_provenance
           (prov_id, element_id, doc_id, prov_index, page_no,
            bbox_l, bbox_t, bbox_r, bbox_b, bbox_coord_origin,
            charspan_start, charspan_end)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        prov_rows
    )

    conn.executemany(
        """INSERT OR IGNORE INTO table_cell
           (cell_id, element_id, doc_id, row_index, col_index,
            row_span, col_span, end_row_index, end_col_index,
            text, is_column_header, is_row_header, is_row_section,
            is_fillable, bbox_l, bbox_t, bbox_r, bbox_b)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        cell_rows
    )

    conn.executemany(
        """INSERT OR IGNORE INTO visual_asset_registry
           (asset_id, element_id, doc_id, physical_pg_num,
            image_path, vlm_description,
            is_global, caption_refs, child_text_refs,
            bbox_l, bbox_t, bbox_r, bbox_b, occurrence_count)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        visual_rows
    )

    conn.executemany(
        """INSERT OR IGNORE INTO furniture_registry
           (furniture_id, element_id, doc_id, furniture_type, text,
            text_hash, physical_pg_num, bundle_process_id,
            is_variant, variant_group)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        furniture_rows
    )

    # ── Update page_registry counts ─────────────────────────────────────
    for pg_no in page_element_count:
        conn.execute(
            """UPDATE page_registry SET element_count = ?, has_tables = ?, has_pictures = ?
               WHERE doc_id = ? AND physical_pg_num = ?""",
            (page_element_count.get(pg_no, 0),
             int(pg_no in page_has_tables), int(pg_no in page_has_pictures),
             doc_id, pg_no)
        )
    for pg_no in page_furniture_count:
        conn.execute(
            "UPDATE page_registry SET furniture_count = ? WHERE doc_id = ? AND physical_pg_num = ?",
            (page_furniture_count.get(pg_no, 0), doc_id, pg_no)
        )

    # ── Furniture variant detection ─────────────────────────────────────
    # Group identical text_hash values, mark less common ones as variants
    _compute_furniture_variants(conn, doc_id)

    # ── Merge split tables (from pre-merged JSON) ─────────────────────
    # If tables were merged at the document level before export,
    # the merged_tables info is already in the JSON as table.meta.merged_from.
    # Populate merged_tables/merged_table_members from that metadata.
    _populate_merged_table_metadata(conn, doc_id, data)

    # ── Update knowledge_skeleton child_count ───────────────────────────
    conn.execute(
        """UPDATE knowledge_skeleton SET child_count = (
               SELECT COUNT(*) FROM extraction_element
               WHERE extraction_element.section_id = knowledge_skeleton.section_id
           ) WHERE doc_id = ?""",
        (doc_id,)
    )

    conn.commit()


def _compute_furniture_variants(conn: sqlite3.Connection, doc_id: str):
    """Detect variant furniture (non-standard headers/footers)."""
    for ftype in ("page_header", "page_footer"):
        # Find most common text_hash for this type
        row = conn.execute(
            """SELECT text_hash, COUNT(*) as cnt FROM furniture_registry
               WHERE doc_id = ? AND furniture_type = ?
               GROUP BY text_hash ORDER BY cnt DESC LIMIT 1""",
            (doc_id, ftype)
        ).fetchone()

        if not row:
            continue
        dominant_hash = row[0]

        # Assign variant_group (auto-increment per unique hash)
        hashes = conn.execute(
            """SELECT DISTINCT text_hash FROM furniture_registry
               WHERE doc_id = ? AND furniture_type = ?""",
            (doc_id, ftype)
        ).fetchall()

        for group_idx, (h,) in enumerate(hashes):
            is_variant = 0 if h == dominant_hash else 1
            conn.execute(
                """UPDATE furniture_registry SET variant_group = ?, is_variant = ?
                   WHERE doc_id = ? AND furniture_type = ? AND text_hash = ?""",
                (group_idx, is_variant, doc_id, ftype, h)
            )


def _populate_merged_table_metadata(conn: sqlite3.Connection, doc_id: str, json_data: dict):
    """Read merged_from metadata from tables and populate merged_tables/members.

    This reads the metadata injected by _merge_split_tables() at the document
    level. Each merged table has a 'merged_from' field listing the original
    table refs, pages, and row counts.
    """
    tables = json_data.get("tables", [])
    ref_to_eid = {}
    rows = conn.execute(
        "SELECT element_id, self_ref FROM extraction_element WHERE doc_id = ? AND element_type = 'table'",
        (doc_id,)
    ).fetchall()
    for eid, sr in rows:
        ref_to_eid[sr] = eid

    for tbl in tables:
        merged_from = tbl.get("merged_from")
        if not merged_from or len(merged_from) < 2:
            continue

        merge_id = _uid()
        headers_list = []
        cells = tbl.get("data", {}).get("table_cells", [])
        headers_list = sorted(
            [c for c in cells if c.get("column_header")],
            key=lambda c: c.get("start_col_offset_idx", 0)
        )
        header_texts = [c.get("text", "") for c in headers_list]

        total_rows = tbl.get("data", {}).get("num_rows", 0)
        num_cols = tbl.get("data", {}).get("num_cols", 0)
        pages = [m["page_no"] for m in merged_from if m.get("page_no")]
        member_refs = [m["original_ref"] for m in merged_from]

        # Get table description from meta.description.text (Gemini enrichment)
        tbl_meta = tbl.get("meta") or {}
        tbl_desc = (tbl_meta.get("description") or {}).get("text") or None

        conn.execute(
            """INSERT OR IGNORE INTO merged_tables
               (merge_id, doc_id, column_headers, num_cols, total_rows,
                page_start, page_end, member_count, member_refs, table_summary)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (merge_id, doc_id, json.dumps(header_texts), num_cols, total_rows,
             min(pages) if pages else None, max(pages) if pages else None,
             len(merged_from), json.dumps(member_refs), tbl_desc)
        )

        eid = ref_to_eid.get(tbl.get("self_ref"))
        for midx, m in enumerate(merged_from):
            conn.execute(
                """INSERT OR IGNORE INTO merged_table_members
                   (id, merge_id, element_id, doc_id, self_ref, member_index, page_no, num_rows)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (_uid(), merge_id, eid, doc_id, m["original_ref"], midx,
                 m.get("page_no"), m.get("num_rows"))
            )

    conn.commit()


def merge_split_tables(conn: sqlite3.Connection, doc_id: str, json_data: dict):
    """Detect and merge tables that span multiple pages.

    Algorithm:
    1. Walk tables in body.children order
    2. For each consecutive pair on different pages, compare column headers
    3. If headers match → they're parts of the same logical table
    4. Group chains (A→B→C where all share headers) into one merged_table

    The merged_tables row gets total_rows, page range, and member refs.
    Individual tables keep their extraction_element rows unchanged — the
    merge is an overlay, not a destructive rewrite.
    """
    tables = json_data.get("tables", [])
    if len(tables) < 2:
        return

    # Build ordered list of tables by body_order_index
    body_order = {}
    for idx, child in enumerate(json_data.get("body", {}).get("children", [])):
        cref = child.get("cref", "")
        if cref:
            body_order[cref] = idx

    ordered_tables = sorted(
        [(body_order.get(t["self_ref"], 999999), t) for t in tables],
        key=lambda x: x[0]
    )

    def _get_headers(tbl):
        """Extract column header texts from a table."""
        cells = tbl.get("data", {}).get("table_cells", [])
        headers = sorted(
            [c for c in cells if c.get("column_header")],
            key=lambda c: c.get("start_col_offset_idx", 0)
        )
        return [c.get("text", "").strip() for c in headers]

    def _get_page(tbl):
        prov = tbl.get("prov", [])
        return prov[0]["page_no"] if prov else None

    # Find chains of consecutive tables with matching headers
    chains = []
    current_chain = [ordered_tables[0]]

    for i in range(1, len(ordered_tables)):
        _, prev_tbl = current_chain[-1]
        _, curr_tbl = ordered_tables[i]

        prev_headers = _get_headers(prev_tbl)
        curr_headers = _get_headers(curr_tbl)
        prev_page = _get_page(prev_tbl)
        curr_page = _get_page(curr_tbl)

        if (prev_headers and curr_headers
                and prev_headers == curr_headers
                and prev_page is not None and curr_page is not None
                and prev_page != curr_page):
            current_chain.append(ordered_tables[i])
        else:
            if len(current_chain) > 1:
                chains.append(current_chain)
            current_chain = [ordered_tables[i]]

    if len(current_chain) > 1:
        chains.append(current_chain)

    if not chains:
        return

    # Look up element_ids for table self_refs
    ref_to_eid = {}
    rows = conn.execute(
        "SELECT element_id, self_ref FROM extraction_element WHERE doc_id = ? AND element_type = 'table'",
        (doc_id,)
    ).fetchall()
    for eid, sr in rows:
        ref_to_eid[sr] = eid

    # Insert merged table groups
    for chain in chains:
        merge_id = _uid()
        member_refs = [t["self_ref"] for _, t in chain]
        pages = [_get_page(t) for _, t in chain]
        headers = _get_headers(chain[0][1])
        total_rows = sum(t.get("data", {}).get("num_rows", 0) for _, t in chain)
        num_cols = chain[0][1].get("data", {}).get("num_cols", 0)

        conn.execute(
            """INSERT INTO merged_tables
               (merge_id, doc_id, column_headers, num_cols, total_rows,
                page_start, page_end, member_count, member_refs, table_summary)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (merge_id, doc_id, json.dumps(headers), num_cols, total_rows,
             min(p for p in pages if p), max(p for p in pages if p),
             len(chain), json.dumps(member_refs), None)
        )

        for midx, (_, tbl) in enumerate(chain):
            sr = tbl["self_ref"]
            eid = ref_to_eid.get(sr)
            if not eid:
                continue
            pg = _get_page(tbl)
            nrows = tbl.get("data", {}).get("num_rows", 0)
            conn.execute(
                """INSERT INTO merged_table_members
                   (id, merge_id, element_id, doc_id, self_ref, member_index, page_no, num_rows)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (_uid(), merge_id, eid, doc_id, sr, midx, pg, nrows)
            )

    conn.commit()


def finalize_document(conn: sqlite3.Connection, doc_id: str,
                      status: str, processing_duration: float = None):
    """Stage 11: Mark document as completed/failed with total duration."""
    conn.execute(
        "UPDATE documents SET status = ?, processing_duration = ? WHERE doc_id = ?",
        (status, processing_duration, doc_id)
    )
    conn.commit()


def close_db(conn: sqlite3.Connection):
    """Safely close the database connection."""
    if conn:
        try:
            conn.close()
        except Exception:
            pass
