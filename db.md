# Audit Database — Detailed Pipeline Integration

## How it starts, logs, and ends

The audit database (`audit.db`) is created in the output directory alongside `result.json`. It tracks every step from the moment a PDF is uploaded to the final completion signal. There are **two conversion paths** — both follow the same audit pattern.

---

## Lifecycle Overview

```
User uploads PDF
       │
       ▼
┌─────────────────────────────────────────────────────┐
│ Stage 1: init_db() + insert_document()              │ ← DB created, documents row inserted
│   Tables written: documents                         │    status = "RUNNING"
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│ Stage 2: insert_bundles()                           │ ← All bundles inserted as PENDING
│   Tables written: bundle_process, process_audit     │    (bundled path only)
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│ Stage 3: Bundle conversion loop                     │
│   Per bundle:                                       │
│     update_bundle_status(CONVERTING)                │ ← bundle_process, process_audit
│     ... Docling processes pages ...                 │
│     update_bundle_status(DONE/ERROR)                │ ← bundle_process, process_audit
│                                                     │
│   Every send_timing() call also fires:              │
│     record_timing()                                 │ ← timing_metrics
│                                                     │
│   Any exception also fires:                         │
│     record_error()                                  │ ← error_log
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│ Stage 4-8: Merge → Table merge → Reorder →          │
│            Report → Enrich → Export                  │
│                                                     │
│   record_timing() at each stage boundary            │ ← timing_metrics
│   record_error() on any failure                     │ ← error_log
│                                                     │
│   Enricher writes meta.description to pictures      │
│   and tables on the DoclingDocument object           │
│   (these get serialized into result.json)           │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│ Stage 9: result.json written to disk                │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│ ★ Stage 9.5: populate_from_json()                   │ ← THE BIG ONE
│                                                     │
│   Reads result.json and bulk-inserts:               │
│     documents        (UPDATE with final counts)     │
│     page_registry    (1 row per page)               │
│     knowledge_skeleton (1 row per section_header)   │
│     extraction_element (1 row per text/table/pic/   │
│                         group)                      │
│     element_provenance (for multi-prov elements)    │
│     table_cell       (1 row per cell in every table)│
│     visual_asset_registry (1 row per picture)       │
│     furniture_registry (1 row per header/footer)    │
│     merged_tables    (1 row per merged table group) │
│     merged_table_members (links individual tables)  │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│ Stage 11: finalize_document(COMPLETED/FAILED)       │ ← documents status + duration
│           close_db()                                │ ← connection closed
└─────────────────────────────────────────────────────┘
```

---

## Inline Tables (written during pipeline execution)

These tables are populated in real-time as the pipeline runs, before result.json exists.

### documents

**When created:** Immediately after the output directory is created, before any conversion starts.

**Where in app.py:**
- `_run_bundled_conversion`: `_audit.insert_document()` right after `out_dir.mkdir()`
- `_run_conversion`: same pattern

**Initial insert:**
```
doc_id           = job_id (UUID from /convert endpoint)
filename         = Path(source).name
pipeline         = form param ("standard", "vlm", "simple")
ocr_enabled      = form param
accelerator      = form param ("cuda", "cpu", "mps")
reorder_applied  = form param
status           = "RUNNING"
created_at       = now()
```

**Updated by populate_from_json (Stage 9.5):**
```
total_pages          = len(json["pages"])
schema_version       = json["version"]
total_elements       = len(texts) + len(tables) + len(pictures) + len(groups)
total_bundles        = len(json["bundle_plan"])
status               = "COMPLETED"
```

**Updated by finalize_document (Stage 11):**
```
status               = "COMPLETED" or "FAILED"
processing_duration  = total wall-clock seconds
```

---

### bundle_process

**When created:** After `plan_bundles()` returns the TOC-based bundle list. All bundles inserted at once with status=PENDING. Only exists in the bundled conversion path.

**Where in app.py:** `_audit.insert_bundles()` after line `job["bundles"] = bundle_meta`

**Initial insert (per bundle):**
```
process_id       = "{doc_id}_bundle_{i:03d}"
doc_id           = job_id
bundle_index     = 0, 1, 2, ... (sequence number)
bundle_plan_id   = bundle.id (e.g. "b002_1_introduction")
bundle_name      = bundle.name (e.g. "1 Introduction")
toc_level        = bundle.toc_level
parent_section   = bundle.parent_section
page_start       = bundle.page_start
page_end         = bundle.page_end
is_continuation  = bundle.is_continuation
continuation_of  = bundle.continuation_of
status           = "PENDING"
started_at       = now()
```

**Updated per bundle during conversion (3 modes):**

| Mode | CONVERTING update | DONE update | ERROR update |
|------|-------------------|-------------|--------------|
| Parallel | after thread submit | after future.result() returns | on exception |
| Sequential no-reload | before converter.convert() | after serialize to disk | on exception |
| Subprocess | after proc.start() (includes `child_pid=proc.pid`) | after meta.json read (includes `model_load_time`, `child_pid`) | after meta read or proc failure |

**Fields updated on DONE:**
```
status           = "DONE"
duration         = meta["duration"]
page_count       = meta["page_count"]
model_load_time  = meta.get("model_load_time")  (subprocess only)
child_pid        = proc.pid  (subprocess only)
json_path        = str(bundle_json_path)
completed_at     = now()
```

---

### process_audit

**When created:** A PENDING audit entry is created for each bundle at the same time as bundle_process. Then a new audit row is added on every status transition (CONVERTING, DONE, ERROR).

**Where in app.py:** Inside `_audit.insert_bundles()` and `_audit.update_bundle_status()`

**Each row captures a status transition:**
```
audit_id     = uuid4()
process_id   = "{doc_id}_bundle_{i:03d}"
doc_id       = job_id
status       = "PENDING" / "CONVERTING" / "DONE" / "ERROR"
started_at   = now()
duration     = seconds in this status (null for PENDING/CONVERTING)
retry_count  = 0 (we don't retry currently)
```

**Typical sequence for one bundle:** 3 rows: PENDING → CONVERTING → DONE

---

### timing_metrics

**When created:** Every time `send_timing()` is called anywhere in the pipeline. The `send_timing()` function is wrapped to also call `_audit.record_timing()`.

**Where in app.py:** The `send_timing()` closure defined in both `_run_conversion` and `_run_bundled_conversion` includes:
```python
def send_timing(data: dict):
    job["queue"].put(f"__TIMING__:{_json2.dumps(data)}")
    try:
        _audit.record_timing(_audit_db, job_id, data)
    except Exception:
        pass
```

**Stages that produce timing events:**

| Stage | Key | Extra fields |
|-------|-----|-------------|
| Pipeline init | `pipeline_init` | duration |
| Bundle planning | `bundle_plan` | duration, bundles, total_pages |
| Per-bundle done | `bundle_done` | bundle_index, bundle_id, name, duration, page_count |
| Per-bundle error | `bundle_error` | bundle_index, bundle_id, error, duration |
| Model loading | `models_loading` | duration, bundles |
| Merge | `merge_done` | duration, bundles_merged |
| Table merge | `table_merge` | duration, tables_merged |
| Reorder | `reorder_done` | duration |
| Report | `report_done` | duration |
| Gemini enrichment | `gemini_enrich_done` | duration, pictures, input_tokens, output_tokens, cost_usd |
| Image save | `images_saved` | duration, pictures, tables |
| Export | `export_done` | duration |
| File write | `file_write_done` | duration, size_kb |
| Audit populate | `audit_populate` | duration |
| Chunking | `chunking_done` | duration, chunk_count |
| Total | `total` | duration |

**Each row:**
```
metric_id    = uuid4()
doc_id       = job_id
process_id   = "{doc_id}_bundle_{i:03d}" if bundle_index present, else NULL
stage        = timing_data["stage"]
duration     = timing_data["duration"]
page_count   = timing_data.get("page_count")
bundle_index = timing_data.get("bundle_index")
metadata     = json.dumps(timing_data)  ← full dict preserved
recorded_at  = now()
```

---

### error_log

**When created:** In the `except` blocks of both conversion functions, and potentially during any stage that has error handling.

**Where in app.py:**
- Main pipeline exception handler: `_audit.record_error(_audit_db, job_id, "pipeline", e, is_recoverable=False)`
- Bundle errors go through `update_bundle_status` (no separate error_log entry — the error is on bundle_process.error)

**Each row:**
```
error_id             = uuid4()
doc_id               = job_id
error_area           = "pipeline" / "bundle" / "merge" / "reorder" / "export" / "enrichment"
related_process_id   = "{doc_id}_bundle_{i:03d}" if bundle context, else NULL
related_element_ref  = self_ref if element context (currently unused)
related_page_num     = page number if available (currently unused)
error_description    = str(exception)
error_type           = type(exception).__name__
stack_trace          = traceback.format_exc()
is_recoverable       = 1 if pipeline continued, 0 if fatal
occurred_at          = now()
```

---

## Post-Hoc Tables (populated from result.json after file write)

All of these are populated by a single call: `audit.populate_from_json(conn, doc_id, json_path)`. This function reads the final `result.json` and bulk-inserts everything in one pass.

### page_registry

**Source:** `json["pages"]` dict + `json["_logical_page_map"]` + `json["bundle_plan"]`

**Per page:**
```
uuid              = uuid4()
doc_id            = job_id
physical_pg_num   = pages[key]["page_no"]
logical_pg_num    = _logical_page_map[str(pg_no)]       ← e.g. "7-18" (chapter 7, page 18)
page_width        = pages[key]["size"]["width"]
page_height       = pages[key]["size"]["height"]
bundle_process_id = looked up via: page_no → bundle_plan → "{doc_id}_bundle_{i:03d}"
```

**Computed after all elements are inserted (UPDATE):**
```
element_count     = COUNT of body elements on this page
furniture_count   = COUNT of furniture elements on this page
has_tables        = 1 if any table on this page
has_pictures      = 1 if any picture on this page
```

---

### knowledge_skeleton

**Source:** `json["texts"]` where `label == "section_header"` + hierarchy metadata from `_post_process_json()`

**Per section header:**
```
section_id          = uuid4()
doc_id              = job_id
self_ref            = item["self_ref"]               ← e.g. "#/texts/42"
hierarchy_path      = item["hierarchy_path"]          ← e.g. "7/4/2" (injected by _post_process_json)
hierarchy_title     = item["hierarchy_title"]          ← e.g. "Troubleshooting>Display>Board Assembly"
level               = item["level"]                   ← heading level (1 in current data)
text                = item["text"]                    ← the header text
start_physical_pg   = item["prov"][0]["page_no"]
end_physical_pg     = next section's start_pg (or total pages for last section)
bundle_process_id   = from item["bundle"]["id"] → bundle_to_process lookup
parent_section_id   = NULL (not computed currently)
```

**Computed after all elements inserted (UPDATE):**
```
child_count         = COUNT of extraction_elements with this section_id
```

---

### extraction_element

**Source:** `json["texts"]` + `json["tables"]` + `json["pictures"]` + `json["groups"]`

This is the main content table. Every element from the document gets a row.

**Walk order:** body.children is walked in reading order. For each element:
1. If it's a section_header, the `current_section_id` is updated
2. The element gets the current section_id assigned
3. Child elements (group children, picture children) are also processed

**Per element:**
```
element_id        = uuid4()
doc_id            = job_id
self_ref          = item["self_ref"]                  ← "#/texts/0", "#/tables/5", "#/pictures/2", "#/groups/3"
parent_ref        = item["parent"]["cref"]            ← "#/body", "#/groups/N"
element_type      = derived from array: texts→"text", tables→"table", pictures→"picture", groups→"group"
label             = item["label"]                     ← "section_header", "text", "list_item", "table", "picture", etc.
content_layer     = item["content_layer"]             ← "body" or "furniture"
content           = item["text"] for texts
                    _table_cells_to_markdown() for tables  ← FULL MARKDOWN of the table
                    NULL for pictures and groups
orig              = item["orig"]                      ← original unprocessed text
bbox_l/t/r/b      = item["prov"][0]["bbox"]           ← NULL for groups (no prov)
bbox_coord_origin = item["prov"][0]["bbox"]["coord_origin"]
physical_pg_num   = item["prov"][0]["page_no"]         ← NULL for groups
charspan_start    = item["prov"][0]["charspan"][0]
charspan_end      = item["prov"][0]["charspan"][1]
section_id        = current_section_id                ← FK to knowledge_skeleton, tracked during walk
bundle_process_id = from item["bundle"]["id"] → bundle_to_process lookup
body_order_index  = position in body.children          ← NULL for nested children
hyperlink         = item["hyperlink"]
formatting        = item["formatting"]
is_furniture      = 1 if content_layer == "furniture"
is_fragment       = 1 if len(prov) > 1                ← multi-page element
continuation_ref  = NULL
has_children      = 1 if children exist
child_refs        = JSON array of children crefs
description       = item["meta"]["description"]["text"] ← Gemini enrichment (pictures + tables)
```

---

### element_provenance

**Source:** Elements with `len(prov) > 1` (~30 out of 10,000+ elements)

**When an element has multiple provenance entries** (e.g. a list_item spanning two columns on the same page), each prov entry gets its own row:

```
prov_id           = uuid4()
element_id        = parent element's UUID
doc_id            = job_id
prov_index        = 0, 1, 2, ... (index in the prov array)
page_no           = prov[i]["page_no"]
bbox_l/t/r/b      = prov[i]["bbox"]
bbox_coord_origin = prov[i]["bbox"]["coord_origin"]
charspan_start    = prov[i]["charspan"][0]
charspan_end      = prov[i]["charspan"][1]
```

---

### table_cell

**Source:** `json["tables"][N]["data"]["table_cells"]` for every table

**Per cell in every table:**
```
cell_id           = uuid4()
element_id        = parent table's UUID (from extraction_element)
doc_id            = job_id
row_index         = cell["start_row_offset_idx"]
col_index         = cell["start_col_offset_idx"]
row_span          = cell["row_span"]
col_span          = cell["col_span"]
end_row_index     = cell["end_row_offset_idx"]
end_col_index     = cell["end_col_offset_idx"]
text              = cell["text"]
is_column_header  = int(cell["column_header"])
is_row_header     = int(cell["row_header"])
is_row_section    = int(cell["row_section"])
is_fillable       = int(cell["fillable"])
bbox_l/t/r/b      = cell["bbox"]
```

**For merged tables:** The cells include re-indexed rows from continuation tables. Each cell may have `origin_page` and `origin_ref` (injected during JSON export) to trace back to the source table/page.

---

### visual_asset_registry

**Source:** `json["pictures"]` — one row per picture element

```
asset_id          = uuid4()
element_id        = parent picture's UUID (from extraction_element)
doc_id            = job_id
physical_pg_num   = item["prov"][0]["page_no"]
image_hash        = NULL (pHash computation deferred)
image_path        = item["image_path"]                ← path to saved image file
vlm_description   = item["meta"]["description"]["text"] ← Gemini-generated description
                    fallback: item["annotations"][*]["text"]
is_global         = 0 (not computed currently)
caption_refs      = JSON array of item["captions"][*]["cref"]
child_text_refs   = JSON array of item["children"][*]["cref"]
bbox_l/t/r/b      = from first prov bbox
occurrence_count  = 1 (default)
```

**The `vlm_description` field** contains the structured Gemini response:
```
PURPOSE: One sentence — what the image shows.
COMPONENTS: Comma-separated labeled parts, callout numbers.
VALUES: Measurements, settings visible. "None" if absent.
DESCRIPTION: 2-3 sentences for a reader.
```

---

### furniture_registry

**Source:** Elements in `json["texts"]` where `label` is `page_header` or `page_footer`

```
furniture_id       = uuid4()
element_id         = parent element's UUID (from extraction_element)
doc_id             = job_id
furniture_type     = "page_header" or "page_footer"
text               = item["text"]
text_hash          = md5(text)                         ← for grouping identical furniture
physical_pg_num    = item["prov"][0]["page_no"]
bundle_process_id  = from bundle lookup
is_variant         = computed post-insert              ← 1 if differs from most common text
variant_group      = computed post-insert              ← integer grouping identical text_hash values
```

**Variant detection** runs after all furniture rows are inserted. For each type (header/footer):
1. Find the most common `text_hash` (the "dominant" version)
2. Mark all rows with a different hash as `is_variant = 1`
3. Assign `variant_group` numbers (0 = dominant, 1, 2, ... = variants)

---

### merged_tables

**Source:** `json["tables"]` where `merged_from` field exists (injected by `_merge_split_tables()` during pipeline)

A merged table is one that was originally split across multiple pages but had matching column headers. The document-level merge combined them before export.

```
merge_id          = uuid4()
doc_id            = job_id
column_headers    = JSON array of header cell texts
num_cols          = table["data"]["num_cols"]
total_rows        = table["data"]["num_rows"]          ← combined rows from all parts
page_start        = min(merged_from[*]["page_no"])
page_end          = max(merged_from[*]["page_no"])
member_count      = len(merged_from)
member_refs       = JSON array of original table refs   ← e.g. ["#/tables/20", "#/tables/21", "#/tables/22"]
table_summary     = table["meta"]["description"]["text"] ← Gemini summary of the full merged table
```

### merged_table_members

**Source:** Each entry in `merged_from` array on the merged table

```
id                = uuid4()
merge_id          = parent merged_tables UUID
element_id        = looked up from extraction_element by the merged table's self_ref
doc_id            = job_id
self_ref          = merged_from[i]["original_ref"]     ← e.g. "#/tables/21" (now deleted from output)
member_index      = 0, 1, 2, ...
page_no           = merged_from[i]["page_no"]          ← which page this part came from
num_rows          = merged_from[i]["num_rows"]          ← how many rows this part contributed
```

---

## Data Flow: Enrichment → JSON → DB

The enricher runs on the `DoclingDocument` object before JSON export. Here's how the enrichment data flows:

```
enricher.py                    result.json                    audit.db
───────────                    ───────────                    ────────

Pictures:
  pic.meta.description    →    pictures[].meta.description →  extraction_element.description
    .text = "PURPOSE:..."        .text = "PURPOSE:..."         = "PURPOSE:..."
    .created_by = "gemini"       .created_by = "gemini"
                                                            →  visual_asset_registry.vlm_description
                                                               = "PURPOSE:..."

Tables:
  tbl.meta.description    →    tables[].meta.description  →  extraction_element.description
    .text = "PURPOSE:..."        .text = "PURPOSE:..."         = "PURPOSE:..."
    .created_by = "gemini"       .created_by = "gemini"

Merged Tables:
  (same as tables above)  →    tables[].meta.description  →  merged_tables.table_summary
                                                               = "PURPOSE:..."
```

---

## Row Count Expectations (484-page doc)

| Table | Rows | When populated |
|-------|------|----------------|
| documents | 1 | Stage 1 (insert) + Stage 9.5 (update) + Stage 11 (finalize) |
| page_registry | 484 | Stage 9.5 |
| knowledge_skeleton | ~607 | Stage 9.5 |
| extraction_element | ~11,269 | Stage 9.5 |
| element_provenance | ~60 | Stage 9.5 |
| table_cell | ~6,000 | Stage 9.5 |
| visual_asset_registry | ~380 | Stage 9.5 |
| furniture_registry | ~1,401 | Stage 9.5 |
| merged_tables | ~27 | Stage 9.5 |
| merged_table_members | ~82 | Stage 9.5 |
| bundle_process | 18 | Stage 2 (insert) + Stage 3 (updates) |
| process_audit | ~54 | Stage 2 + Stage 3 (3 transitions per bundle) |
| timing_metrics | ~30 | Throughout pipeline |
| error_log | 0-5 | On failures |

**Total: ~20,000+ rows. SQLite handles in <1 second.**

---

## File Layout

```
outputs/{job_id}/
  result.json          ← document output
  audit.db             ← SQLite audit database (14 tables)
  prompts.json         ← Gemini prompt/response log (if enrichment ran)
  chunks.json          ← chunked output (if chunking enabled)
  pictures/            ← saved picture images
  tables/              ← saved table images
```
