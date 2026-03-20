# Docling Audit Table Design

## Overview

12 tables providing full traceability for every element produced by the Docling pipeline. Each element gets a UUID primary key, and `(parent_doc_id, self_ref)` is the composite unique constraint that handles multiple PDFs sharing the same `#/texts/0` etc.

### Implementation: `audit.py`

Population happens via two mechanisms:

1. **Inline hooks** — lightweight inserts during pipeline stages (bundle status, timing, errors)
2. **Post-hoc JSON walk** — after `result.json` is written, `populate_from_json()` walks the JSON and bulk-inserts all content tables

The SQLite file lives at `outputs/{job_id}/audit.db` alongside `result.json`.

---

## Pipeline Integration Map

```
Stage 1: Upload (/convert, line 2272)
  → audit.init_db(outputs/{job_id}/audit.db)
  → audit.insert_document(uuid, filename, pipeline, ocr, ...)

Stage 2: Bundle Plan (line 1730)
  → audit.insert_bundles(parent_doc_id, bundles)           # all PENDING rows

Stage 3: Conversion — 3 modes all do the same:
  Before each bundle:
    → audit.update_bundle_status(CONVERTING, child_pid=proc.pid)
  After each bundle:
    → audit.update_bundle_status(DONE/ERROR, duration, page_count, ...)
  Each send_timing() call:
    → audit.record_timing(parent_doc_id, timing_data)
  Each except block:
    → audit.record_error(parent_doc_id, stage, exception, bundle_index)

Stage 4-8: Merge/Reorder/Report/Enrich/Export
  → audit.record_timing() per stage
  → audit.record_error() on failure

Stage 9: File Write (line 2175)
  → result.json now on disk

★ Stage 9.5: AUDIT POPULATION (after line 2178)
  → audit.populate_from_json(conn, parent_doc_id, result_path)
  → Bulk-inserts into ALL content tables:
     Documents (update counts), Page_Registry, Knowledge_Skeleton,
     Extraction_Element, Element_Provenance, Table_Cell,
     Visual_Asset_Registry, Furniture_Registry

Stage 11: Completion (line 2215)
  → audit.finalize_document(parent_doc_id, "COMPLETED", total_duration)
  → audit.close_db(conn)
```

---

## Tables

### 1. Documents

**Parent record. Every FK chain starts here. UUID = job_id.**

| Column | Type | Description | Populated From |
|--------|------|-------------|----------------|
| `uuid` | TEXT PK | Document unique id (= job_id) | `/convert` endpoint |
| `filename` | TEXT | Name of the PDF file | upload |
| `cloud_storage_link` | TEXT | Link to the raw file | upload |
| `total_pages` | INT | Total raw pages | post-hoc JSON len(json["pages"]) |
| `master_summary` | TEXT | AI-generated document summary | enrichment stage |
| `schema_version` | TEXT | json["version"] | post-hoc JSON |
| `pipeline` | TEXT | `standard`/`vlm`/`simple` | form param |
| `ocr_enabled` | INT | Boolean | form param |
| `ocr_engine` | TEXT | `easyocr`/`tesseract` | form param |
| `accelerator` | TEXT | `cuda`/`cpu`/`mps` | form param |
| `reorder_applied` | INT | Boolean | form param |
| `total_elements` | INT | texts+tables+pics+groups | post-hoc JSON |
| `total_bundles` | INT | len(bundle_plan) | post-hoc JSON |
| `processing_duration` | REAL | Total seconds | `total` timing |
| `status` | TEXT | RUNNING→COMPLETED/FAILED | inline |
| `created_at` | TEXT | ISO timestamp | inline |

---

### 2. Page_Registry

**Skeleton table mapping physical page to logical page and physical page to image (needed for ColPali).**

| Column | Type | Description | Populated From |
|--------|------|-------------|----------------|
| `uuid` | TEXT PK | Unique id | post-hoc |
| `parent_doc_id` | TEXT FK | -> Documents.uuid | post-hoc |
| `physical_pg_num` | INT | Physical page number (1-based) | json["pages"] |
| `logical_pg_num` | TEXT | Logical path e.g. `7-18` (chapter 7, page 18) | hierarchy + physical page |
| `page_width` | REAL | Points | json["pages"][N]["size"]["width"] |
| `page_height` | REAL | Points | json["pages"][N]["size"]["height"] |
| `thumbnail_url` | TEXT | Path to Low Res picture (for text page UI and ColPali) | if generated |
| `full_page_url` | TEXT | Path to high-res screenshot (used in Gemini during synthesis) | if generated |
| `is_complex` | INT | Boolean — True means use high_res and False means use fast | computed |
| `triage_type` | TEXT | `triage_fast` (text only) or `triage_hires` (for complex pages) | computed |
| `status` | TEXT | DISPATCHED / COMPLETED / FAILED (for Celery processing) | inline |
| `bundle_id` | TEXT FK | -> Bundle_Process | bundle_plan page ranges |
| `element_count` | INT | Body elements on this page | computed after element insert |
| `furniture_count` | INT | Headers/footers on this page | computed after element insert |
| `has_tables` | INT | Boolean | computed |
| `has_pictures` | INT | Boolean | computed |

UNIQUE(parent_doc_id, physical_pg_num)

---

### 3. Knowledge_Skeleton

**Provides the structure of the document. Section headers = document skeleton.**

| Column | Type | Description | Populated From |
|--------|------|-------------|----------------|
| `section_id` | TEXT PK | UUID used to identify this section unique id | post-hoc |
| `parent_doc_id` | TEXT FK | -> Documents.uuid | post-hoc |
| `self_ref` | TEXT | e.g. `#/texts/42` | json texts[].self_ref |
| `hierarchy_path` | TEXT | Path for the section e.g. `7/4/2` (Chapter 7 → Section 4 → Subsection 2) | `_post_process_json()` output |
| `hierarchy_title` | TEXT | Full title path e.g. `Troubleshooting>Display>Board Assembly` | `_post_process_json()` output |
| `level` | INT | Integer value that shows the level within the TOC | texts[].level |
| `text` | TEXT | Header text | texts[].text |
| `start_physical_pg` | INT | Start page number physical one | texts[].prov[0].page_no |
| `end_physical_pg` | INT | End physical page of that section | inferred from next section |
| `section_summary` | TEXT | Summarization activity — for this specific block which should be a logical section | enrichment |
| `bundle_id` | TEXT FK | -> Bundle_Process (foreign key to connect with bundle table) | via page→bundle map |
| `parent_section_id` | TEXT FK | -> self (parent section) | hierarchy parsing |
| `child_count` | INT | Elements in section | computed after all inserts |

UNIQUE(parent_doc_id, self_ref)

---

### 4. Extraction_Element

**The Mega Markdown content of the entire document. This is where the actual content lives. Instead of one giant file, we are storing it at the element level to allow for "Precision Retrieval."**

| Column | Type | Description | Populated From |
|--------|------|-------------|----------------|
| `element_id` | TEXT PK | UUID used to identify this element unique id | post-hoc |
| `parent_doc_id` | TEXT FK | -> Documents.uuid | post-hoc |
| `self_ref` | TEXT | `#/texts/0`, `#/tables/5`, etc. | item["self_ref"] |
| `parent_ref` | TEXT | `#/body`, `#/groups/N` | item["parent"]["cref"] |
| `type` | TEXT | Text, table or image description | derived from array name |
| `label` | TEXT | Docling label | item["label"] |
| `content_layer` | TEXT | `body`/`furniture` | item["content_layer"] |
| `content` | TEXT | The actual Markdown or JSON formatted Table | item["text"] |
| `orig` | TEXT | Original unprocessed text | item["orig"] |
| `bbox` | TEXT | Bounding Box — coordinates of where the content is [x,y,w,h] | prov[0]["bbox"] as JSON |
| `physical_pg_num` | INT | Physical page for which this Markdown is relevant — 1 Physical Page can have multiple Markdown entries | prov[0]["page_no"] |
| `charspan_start` | INT | Start | prov[0]["charspan"][0] |
| `charspan_end` | INT | End | prov[0]["charspan"][1] |
| `section_id` | TEXT FK | -> Knowledge_Skeleton (foreign key to the knowledge skeleton section) | tracked during body walk |
| `bundle_id` | TEXT FK | -> Bundle_Process | via bundle["id"] on item |
| `body_order_index` | INT | Reading order position | enumerate(body.children) |
| `hyperlink` | TEXT | URL | item["hyperlink"] |
| `formatting` | TEXT | Bold/italic/etc. | item["formatting"] |
| `is_furniture` | INT | 1 if furniture layer | content_layer check |
| `is_fragment` | INT | Boolean — set to TRUE if the same row is split between pages. At retrieval time, stitch by removing trailing pipe of first page and leading pipe of second page | len(prov) > 1 |
| `continuation_id` | TEXT | UUID of another Markdown element (continuation becomes important when table entries spill to next page) | cross-page detection |
| `has_children` | INT | 1 if children[] | len(children) > 0 |
| `child_refs` | TEXT | JSON array of crefs | children[].cref |

UNIQUE(parent_doc_id, self_ref)

**Groups note:** Groups have NO prov and NO bundle info. Their bbox/page/bundle fields will be NULL.

---

### 5. Element_Provenance

**Multi-prov entries. ~30/10,109 elements need this but it's cheap.**

| Column | Type | Description | Populated From |
|--------|------|-------------|----------------|
| `prov_id` | TEXT PK | Generated UUID | post-hoc |
| `element_id` | TEXT FK | -> Extraction_Element | parent element UUID |
| `parent_doc_id` | TEXT FK | -> Documents.uuid | post-hoc |
| `prov_index` | INT | 0-based index | enumerate(prov) |
| `page_no` | INT | Page number | prov[i]["page_no"] |
| `bbox_l` | REAL | Left | prov[i]["bbox"]["l"] |
| `bbox_t` | REAL | Top | prov[i]["bbox"]["t"] |
| `bbox_r` | REAL | Right | prov[i]["bbox"]["r"] |
| `bbox_b` | REAL | Bottom | prov[i]["bbox"]["b"] |
| `bbox_coord_origin` | TEXT | Origin | prov[i]["bbox"]["coord_origin"] |
| `charspan_start` | INT | Start | prov[i]["charspan"][0] |
| `charspan_end` | INT | End | prov[i]["charspan"][1] |

---

### 6. Table_Cell

**Cell-level data for structured tables.**

| Column | Type | Description | Populated From |
|--------|------|-------------|----------------|
| `cell_id` | TEXT PK | Generated UUID | post-hoc |
| `element_id` | TEXT FK | -> Extraction_Element | parent table UUID |
| `parent_doc_id` | TEXT FK | -> Documents.uuid | post-hoc |
| `row_index` | INT | Row start | cell["start_row_offset_idx"] |
| `col_index` | INT | Col start | cell["start_col_offset_idx"] |
| `row_span` | INT | Rows spanned | cell["row_span"] |
| `col_span` | INT | Cols spanned | cell["col_span"] |
| `end_row_index` | INT | Row end | cell["end_row_offset_idx"] |
| `end_col_index` | INT | Col end | cell["end_col_offset_idx"] |
| `text` | TEXT | Cell content | cell["text"] |
| `is_column_header` | INT | Boolean | cell["column_header"] |
| `is_row_header` | INT | Boolean | cell["row_header"] |
| `is_row_section` | INT | Boolean | cell["row_section"] |
| `is_fillable` | INT | Boolean | cell["fillable"] |
| `bbox_l` | REAL | Left | cell["bbox"]["l"] |
| `bbox_t` | REAL | Top | cell["bbox"]["t"] |
| `bbox_r` | REAL | Right | cell["bbox"]["r"] |
| `bbox_b` | REAL | Bottom | cell["bbox"]["b"] |

---

### 7. Visual_Asset_Registry

**Stores the repeat content. The repeat content can be a logo across all pages vs fixed footer info across chapters, etc. In any case these are something that appears a lot of time.**

| Column | Type | Description | Populated From |
|--------|------|-------------|----------------|
| `visual_asset_id` | TEXT PK | UUID used to identify this element unique id | post-hoc |
| `element_id` | TEXT FK | -> Extraction_Element | parent picture UUID |
| `parent_doc_id` | TEXT FK | -> Documents.uuid | post-hoc |
| `physical_pg_registry_num` | INT FK | -> Page_Registry (foreign key to the Page Registry table) | prov[0]["page_no"] |
| `image_hash` | TEXT | pHash of the image | computed from saved image |
| `web_url` | TEXT | Path to the cropped asset | item["image_path"] |
| `vlm_description` | TEXT | Gemini generated description | enrichment |
| `is_global` | INT | Boolean TRUE means across all pages. False means only some sections | computed post-insert |
| `bbox` | TEXT | [x,y,w,h] coordinate of where it is on the page | prov[0]["bbox"] as JSON |
| `caption_refs` | TEXT | JSON array of caption refs | item["captions"] |
| `child_text_refs` | TEXT | JSON array of child refs | item["children"] |
| `occurrence_count` | INT | Hash match count | computed |

**If is_global is false then many of the other elements become valid and populated like Bbox, pagenumber, etc.**

---

### 8. Furniture_Registry

**Page headers/footers with variant detection.**

| Column | Type | Description | Populated From |
|--------|------|-------------|----------------|
| `furniture_id` | TEXT PK | Generated UUID | post-hoc |
| `element_id` | TEXT FK | -> Extraction_Element | parent element UUID |
| `parent_doc_id` | TEXT FK | -> Documents.uuid | post-hoc |
| `furniture_type` | TEXT | `page_header`/`page_footer` | item["label"] |
| `text` | TEXT | Content | item["text"] |
| `text_hash` | TEXT | MD5 for grouping | computed |
| `physical_pg_num` | INT | Page | prov[0]["page_no"] |
| `bundle_id` | TEXT FK | -> Bundle_Process | via bundle info |
| `is_variant` | INT | Differs from most common | computed post-insert |
| `variant_group` | INT | Groups identical text | computed post-insert |

---

### 9. Bundle_Process

**Per-bundle conversion tracking. UUID to identify the bundle Primary Key.**

| Column | Type | Description | Populated From |
|--------|------|-------------|----------------|
| `bundle_id` | TEXT PK | UUID to identify the bundle Primary Key | inline |
| `doc_id` | TEXT FK | Associated document id -> Documents.uuid | inline |
| `index` | INT | The sequence number (0, 1, 2...) — we need bundle sequences for stitching | enumerate(bundles) |
| `extraction_status` | TEXT | PENDING, COMPLETED, FAILED | inline updates |
| `has_open_elements` | INT | Boolean — flag for the Stitcher; essentially the bundle worker will need to detect if this should be TRUE (open table, open list item, para with hyphen, etc.) | computed |
| `gcs_path` | TEXT | Path to raw Markdown | `_bundle_{i:03d}.json` path |
| `bundle_plan_id` | TEXT | e.g. `b002_1_introduction` | bundle.id |
| `bundle_name` | TEXT | e.g. `1 Introduction` | bundle.name |
| `toc_level` | INT | TOC depth | bundle.toc_level |
| `parent_section` | TEXT | Parent section name | bundle.parent_section |
| `page_start` | INT | First page | bundle.page_start |
| `page_end` | INT | Last page | bundle.page_end |
| `is_continuation` | INT | Boolean | bundle.is_continuation |
| `continuation_of` | TEXT | Parent bundle_id | bundle.continuation_of |
| `duration` | REAL | Seconds | meta["duration"] |
| `page_count` | INT | Pages converted | meta["page_count"] |
| `model_load_time` | REAL | Model load seconds | meta["model_load_time"] |
| `error` | TEXT | Error message | meta["error"] |
| `child_pid` | INT | Process PID (subprocess only) | proc.pid |
| `started_at` | TEXT | ISO timestamp | inline |
| `completed_at` | TEXT | ISO timestamp | inline |

UNIQUE(doc_id, index)

---

### 10. Task_Audit

**For the process bundle audit — refer to FDR logic for more context. Bundle lifecycle status transitions.**

| Column | Type | Description | Populated From |
|--------|------|-------------|----------------|
| `audit_id` | TEXT PK | UUID used to identify this element unique id | inline |
| `bundle_id` | TEXT FK | -> Bundle_Process | inline |
| `parent_doc_id` | TEXT FK | -> Documents.uuid | inline |
| `status` | TEXT | Pending, In Progress, Completed, Failed | inline |
| `worker_id` | TEXT | Celery worker identifier | inline |
| `retry_count` | INT | Helps with three strike rule | inline |
| `started_at` | TEXT | When status changed | inline |
| `duration` | REAL | Seconds in this status | inline |
| `worker_host` | TEXT | Machine hostname | inline |
| `gpu_utilization` | REAL | GPU % | OS metrics |
| `cpu_utilization` | REAL | CPU % | OS metrics |

---

### 11. Error_Log

**This is to capture Error details. {Requires work}. Pipeline errors at any level.**

| Column | Type | Description | Populated From |
|--------|------|-------------|----------------|
| `error_id` | TEXT PK | UUID used to identify this element unique id | inline |
| `document_id` | TEXT FK | -> Documents.uuid (foreign key of the document) | inline |
| `error_area` | TEXT | Knowledge, Page, Element, Celery Worker | except block context |
| `related_object_id` | TEXT FK | Foreign key that represents the table (-> Bundle_Process, nullable) | bundle_index if applicable |
| `related_element_ref` | TEXT | self_ref of failed element | if available |
| `related_page_num` | INT | Page number | if available |
| `error_description` | TEXT | Helps with three strike rule | inline |
| `error_type` | TEXT | Exception class name | type(e).__name__ |
| `stack_trace` | TEXT | Full traceback | traceback.format_exc() |
| `is_recoverable` | INT | Processing continued? | context |
| `date_time_stamp` | TEXT | When occurred (ISO timestamp) | inline |

---

### 12. Timing_Metrics

**Granular stage timings. Persists what send_timing() already produces.**

| Column | Type | Description | Populated From |
|--------|------|-------------|----------------|
| `metric_id` | TEXT PK | Generated UUID | inline |
| `parent_doc_id` | TEXT FK | -> Documents.uuid | inline |
| `bundle_id` | TEXT FK | -> Bundle_Process (nullable) | bundle_index if applicable |
| `stage` | TEXT | `bundle_plan`/`bundle_done`/`merge`/etc. | timing_data["stage"] |
| `duration` | REAL | Seconds | timing_data["duration"] |
| `page_count` | INT | Pages if applicable | timing_data["page_count"] |
| `bundle_index` | INT | Bundle if applicable | timing_data["bundle_index"] |
| `metadata` | TEXT | Full timing dict as JSON | json.dumps(timing_data) |
| `recorded_at` | TEXT | ISO timestamp | inline |

---

## Row Count Estimates (484-page doc)

| Table | Rows | Timing |
|-------|------|--------|
| Documents | 1 | inline + post-hoc update |
| Page_Registry | 484 | post-hoc |
| Knowledge_Skeleton | 617 | post-hoc |
| Extraction_Element | 11,269 | post-hoc |
| Element_Provenance | ~60 | post-hoc |
| Table_Cell | ~5,000-15,000 | post-hoc |
| Visual_Asset_Registry | 380 | post-hoc |
| Furniture_Registry | 1,401 | post-hoc |
| Bundle_Process | 18 | inline |
| Task_Audit | ~54 (18x3 transitions) | inline |
| Error_Log | 0-5 | inline |
| Timing_Metrics | ~30 | inline |

**Total: ~19,000-29,000 rows. SQLite handles in <1 second.**

---

## Relationship Diagram

```
Documents (uuid)
  |
  |-- (N) Extraction_Element              [every element, FK: parent_doc_id]
  |       |-- (N) Element_Provenance      [multi-bbox]
  |       |-- (N) Table_Cell              [cell data]
  |       |-- (1) Visual_Asset_Registry   [picture details]
  |       |-- (1) Furniture_Registry      [header/footer tracking]
  |       |-- FK -> Knowledge_Skeleton.section_id
  |       |-- FK -> Bundle_Process.bundle_id
  |
  |-- (N) Page_Registry                   [page context, FK: parent_doc_id]
  |       |-- FK -> Bundle_Process.bundle_id
  |
  |-- (N) Knowledge_Skeleton              [section structure, FK: parent_doc_id]
  |       |-- FK -> Bundle_Process.bundle_id
  |       |-- FK -> self (parent_section_id)
  |
  |-- (N) Bundle_Process                  [conversion tracking, FK: doc_id]
  |       |-- (N) Task_Audit              [lifecycle, FK: bundle_id]
  |
  |-- (N) Error_Log                       [failures, FK: document_id]
  |-- (N) Timing_Metrics                  [performance, FK: parent_doc_id]
```
