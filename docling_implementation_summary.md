# Docling UI — Implementation Summary

## 1. What This System Does

We built a web-based wrapper around IBM's **Docling** library that converts PDF documents into structured JSON, Markdown, HTML, or DocTags. The wrapper adds capabilities Docling does not provide natively: **TOC-based bundle splitting** for memory-bounded conversion of large PDFs, **reading-order correction**, **Gemini Vision enrichment** of pictures, **image extraction**, **hierarchy metadata injection**, and **real-time progress streaming** via SSE.

The system is a **FastAPI** backend (`app.py`, ~2680 lines) with a vanilla JS/HTML/CSS frontend, plus three support modules: `bundler.py` (TOC extraction and bundle planning), `enricher.py` (Gemini Vision API integration), and `reorder.py` (standalone reading-order correction library).

---

## 2. Architecture Overview

```
User Browser
    │
    ├─ POST /convert (file upload + options)
    │     │
    │     ▼
    │  FastAPI Endpoint
    │     │
    │     ├─── Bundle mode? ──► _run_bundled_conversion()
    │     │                         │
    │     │                         ├─ Plan bundles from TOC (PyMuPDF)
    │     │                         ├─ Convert each bundle (subprocess / shared / parallel)
    │     │                         ├─ Merge via DoclingDocument.concatenate()
    │     │                         ├─ Reorder body.children
    │     │                         ├─ Gemini enrich + image save (parallel)
    │     │                         ├─ Post-process JSON (hierarchy injection)
    │     │                         ├─ Inject bundle metadata
    │     │                         └─ Export + write file
    │     │
    │     └─── Standard mode? ──► _run_conversion()
    │                                 │
    │                                 ├─ Build converter
    │                                 ├─ converter.convert(source, page_range)
    │                                 ├─ Reorder body.children
    │                                 ├─ Gemini enrich + image save (parallel)
    │                                 ├─ Post-process JSON (hierarchy injection)
    │                                 └─ Export + write file
    │
    ├─ GET /stream/{job_id} (SSE)
    │     └─ Real-time: logs, timing, report, bundle_status, gemini_log
    │
    └─ GET /result, /download, /chunks
```

---

## 3. The Memory Problem and How We Solved It

### 3.1 The Problem

Docling has a well-documented memory leak — the **#1 community issue** (GitHub issues #2209, #2829, #2786, #2779, #1654). Three root causes:

1. **C++ PDF backend accumulation**: `DoclingParseDocumentBackend` (v5) uses a C++ parser (`docling-parse`) that allocates memory for page structures. Even with `backend.unload()`, residual C++ allocations accumulate across multiple `convert()` calls in the same process.

2. **PyTorch CUDA context fragmentation**: Once PyTorch allocates GPU memory for model weights and intermediate tensors, the CUDA memory allocator keeps those blocks reserved. `torch.cuda.empty_cache()` releases cached blocks back to CUDA but does not return them to the OS. The CUDA context itself grows permanently within a process.

3. **Pipeline cache retention**: Docling's `DocumentConverter` caches initialized pipelines (`converter.initialized_pipelines`). Models, tokenizers, and image processors remain in memory between calls.

### 3.2 Key Insight: GPU VRAM Does NOT Scale with Page Count

Through research across 21+ GitHub issues and Docling's source code, we established that:

- **GPU VRAM is bounded by batch_size**, not document size. The layout model (RT-DETR) and TableFormer process pages in fixed-size batches (`layout_batch_size`, `table_batch_size`, default ~4-8 pages). Each batch loads tensors, runs inference under `@torch.inference_mode()`, outputs results, and the tensors are freed. A 100-page PDF uses the same peak VRAM as a 1000-page PDF.

- **CPU RAM scales linearly** with page count. The C++ backend holds the full PDF structure in memory, `ConversionResult.pages` accumulates page objects, and the final `DoclingDocument` grows with content.

- **The real problem is cross-job accumulation**: within a single large document, memory is bounded (aside from CPU growth). The leak manifests when processing multiple documents sequentially — each conversion leaves residual C++ allocations and CUDA context growth.

### 3.3 Solution: Three-Mode Bundle Conversion

We implemented three conversion modes, each trading speed for memory isolation:

#### Mode 1: Subprocess Isolation (Model Reload ON — Default)
**`app.py` lines 1963-2026**

```
For each bundle:
    1. Spawn multiprocessing.Process targeting _subprocess_convert_bundle()
    2. Child process: fresh DocumentConverter → convert page_range → serialize JSON to disk → exit
    3. OS reclaims ALL memory (CPU + GPU) when child exits
    4. Parent reads JSON from disk, continues to next bundle
```

**Why it works**: Each child process gets its own CUDA context. When the process exits, the OS reclaims everything — no leaks possible. Memory usage is constant regardless of document size: it's bounded by `max_pages_per_bundle` (default 50).

**Cost**: Model reload per bundle (~3-5 seconds overhead each). For a 484-page PDF with 10 bundles, that's ~30-50 seconds of overhead.

**Windows requirement**: `multiprocessing.set_start_method("spawn")` at line 18. Windows cannot fork CUDA processes — spawn creates a fresh interpreter per child.

The subprocess target function (`_subprocess_convert_bundle`, lines 73-216) is defined at module level (not nested) because `multiprocessing.Process` on Windows needs to pickle the target, and nested functions can't be pickled.

#### Mode 2: Shared Converter, No Reload (Model Reload OFF)
**`app.py` lines 1892-1961**

```
1. Build ONE converter (models loaded once)
2. For each bundle:
    a. converter.convert(source, page_range=(start, end))
    b. Serialize to disk
    c. result.input._backend.unload()   ← free C++ memory
    d. del result; gc.collect()
3. _full_cleanup(converter) after all bundles
```

**Why it exists**: No model reload overhead. Faster than subprocess mode.

**Trade-off**: C++ backend memory accumulates slightly between bundles. GPU stays flat (same models, same batch sizes). CPU grows slightly per bundle due to residual C++ allocations that `unload()` doesn't fully reclaim.

#### Mode 3: Parallel Threads (Parallel ON)
**`app.py` lines 1770-1891**

```
1. Build ONE converter
2. Warm up: convert page 1 to force lazy pipeline initialization (avoids race condition)
3. asyncio + ThreadPoolExecutor(max_workers=2)
4. Each thread calls converter.convert(source, page_range) on its bundle
5. PyTorch forward passes release the GIL → true parallelism for C++/CUDA code
```

**Why 2 workers, not 4**: We initially tried 4 workers. The C++ backend (`docling-parse`) crashed with `std::bad_alloc` — too many concurrent PDF accesses overwhelmed C++ memory. 2 workers is the safe maximum.

**Warm-up at line 1800**: Docling lazily initializes pipelines on the first `convert()` call. Without warm-up, two threads could race to initialize simultaneously, causing crashes. We convert page 1 in the main thread first.

### 3.4 The _full_cleanup() Function
**`app.py` lines 240-314**

This function runs after **every** conversion (not just when "Free VRAM" is checked). It addresses all three leak sources:

```python
# Step 1: Unload page-level backends (lines 248-263)
# Each page has a _backend (pypdfium2 page handle) and _image_cache.
# Unloading frees per-page C++ memory.
for page in result.pages:
    page._backend.unload()
    page._image_cache.clear()

# Step 2: Unload document-level C++ backend (lines 266-271)
# This calls docling-parse's C++ pdf_parser.unload_document(key)
result.input._backend.unload()

# Step 3: Move models GPU → CPU, then drop references (lines 273-299)
# This is the key VRAM recovery step.
for pipeline in converter.initialized_pipelines.values():
    for attr in ('layout_model', 'table_model', 'ocr_model', ...):
        model.cpu()           # Moves tensor data from VRAM to CPU RAM
        pipeline.attr = None  # Drop reference so gc can collect CPU tensors
converter.initialized_pipelines.clear()

# Step 4: Force garbage collection, all 3 generations (lines 301-304)
gc.collect(generation=0)
gc.collect(generation=1)
gc.collect(generation=2)

# Step 5: Release CUDA cached memory (lines 306-313)
torch.cuda.synchronize()  # Finish pending ops
torch.cuda.empty_cache()  # Release cached blocks to CUDA runtime
```

**Why `model.cpu()` matters**: Simply deleting a model doesn't free VRAM immediately — PyTorch's CUDA allocator holds blocks. Moving to CPU first (`model.cpu()`) forces an explicit VRAM→RAM transfer. Then `empty_cache()` releases the now-unused CUDA blocks.

**Observed behavior**: GPU stays at constant VRAM during bundled processing. Between jobs, there's a slight baseline increase (~100-200MB) because CUDA context metadata is permanent within a process. This is why subprocess isolation remains the definitive solution for long-running services.

---

## 4. TOC-Based Bundle Planning

### 4.1 How It Works
**`bundler.py`, 328 lines**

The bundle planner opens the PDF with **PyMuPDF** (`fitz`), which is lightweight — it only reads the xref/outline structure, no page content. This uses ~few MB regardless of PDF size.

```
extract_toc(pdf_path)
    → fitz.open(pdf_path).get_toc()
    → Returns [[level, title, page], ...]
    → e.g., [[1, "Introduction", 1], [2, "Overview", 3], [3, "Details", 5], ...]

_build_section_tree(toc, total_pages)
    → Converts flat TOC into nested _SectionNode tree
    → Computes page_end for each node: scans forward for next same-or-higher-level entry

_split_section_into_bundles(node, max_pages, ...)
    → Recursive. If section fits in max_pages → single bundle
    → If too large → split at subsection boundaries
    → Preamble pages (before first child) absorbed into first group
    → No-children fallback: fixed-page split with continuation tracking
```

### 4.2 Preamble Absorption
**`bundler.py` lines 153-159, 213**

Problem: A chapter like "9. Troubleshooting" (pages 257-414) might have 3 pages of introductory text before the first subsection "9.1 Common Issues" starts on page 260. Without handling this, those 3 pages become a tiny standalone bundle.

Solution: The `override_start` parameter. When the first child's `page_start` > parent's `page_start`, we track those preamble pages and absorb them into the first child group by overriding the bundle's `page_start`.

### 4.3 Bundle Metadata
Each bundle carries metadata that survives into the final JSON:

```python
@dataclass
class Bundle:
    id: str                    # "b003_troubleshooting"
    name: str                  # "Troubleshooting"
    page_start: int            # 257 (1-based inclusive)
    page_end: int              # 414 (1-based inclusive)
    toc_level: int             # 1
    parent_section: str        # "Troubleshooting"
    is_continuation: bool      # False
    continuation_of: str|None  # None (or bundle_id of first part)
```

During JSON export (lines 2146-2163), every element gets a `bundle` field injected based on its page number:

```json
{
  "self_ref": "#/texts/42",
  "text": "Check the display cable...",
  "bundle": {
    "id": "b003_troubleshooting",
    "name": "Troubleshooting",
    "is_continuation": false,
    "continuation_of": null
  }
}
```

The full `bundle_plan` array is also added at the top level of the JSON.

### 4.4 Fallback When No TOC
**`app.py` lines 1734-1748**

If PyMuPDF finds no TOC/bookmarks, the bundled conversion falls back to `_run_conversion()` (standard single-pass mode). The user sees "No TOC found — falling back to standard conversion" in the UI.

---

## 5. Reading-Order Correction

### 5.1 The Problem
Docling's `body.children` array doesn't always reflect human reading order. Elements can be ordered by their internal index rather than spatial position. For example, a header at the top of page 5 might appear after a paragraph from page 3 in the output.

Additionally, Docling has a parsing quirk where description text next to a numbered list item gets parsed as a separate body element instead of part of the list item.

### 5.2 Two-Phase Fix

**Phase 1: Merge Orphaned List Descriptions**
**`app.py` lines 553-629 (also `reorder.py` lines 37-114)**

```
1. Build ref_map: self_ref → item for all elements
2. Build list_item_by_pos: (page_no, rounded_top_y) → (list_item, bbox)
   - Only items inside groups with label="list"
3. For each body-level text element:
   - If its top-Y matches a list item's top-Y (±3px tolerance)
   - AND its left-X is to the right of the list item's right edge
   → It's an orphaned description. Merge text+provenance into the list item, delete orphan.
```

**Why ±3px tolerance**: Docling's bbox coordinates have sub-pixel variations. Two elements on the same visual line might differ by 1-2 points in their `t` coordinate. We round to integers and check `±y_tolerance` (default 3).

**Phase 2: Sort body.children by spatial reading order**
**`app.py` lines 632-675 (also `reorder.py` lines 117-160)**

```python
sort_key = (page ASC, -bbox.t ASC, bbox.l ASC)
```

- **page ASC**: Elements on earlier pages come first.
- **-bbox.t ASC**: Docling uses BOTTOMLEFT coordinate origin — higher `t` = higher on page = comes first. Negating `t` makes ascending sort go top-to-bottom.
- **bbox.l ASC**: Left-to-right within the same vertical position.

**Groups are atomic units**: A `list` group's position is determined by its first leaf element's bbox (recursing through nested groups). The entire group sorts as one unit.

### 5.3 JSON Reindexing
**`app.py` lines 678-752 (also `reorder.py` lines 163-250)**

After `body.children` is sorted, the arrays (`texts[]`, `tables[]`, `pictures[]`, `groups[]`) still have their original indices. Element `#/texts/0` might now be the 50th item in reading order, but its `self_ref` still says `#/texts/0`.

`_reindex_json_reading_order()` fixes this:

1. Depth-first walk of `body.children` → collect canonical reading order per array
2. Build `old_ref → new_ref` mapping (e.g., `#/texts/47 → #/texts/0`)
3. Reorder the actual arrays so index matches reading order
4. Recursively rewrite every JSON-pointer string in the entire document

This ensures that `#/texts/0` is always the first text element in reading order.

---

## 6. Post-Processing: Hierarchy Injection

### 6.1 How It Works
**`app.py` lines 752-821**

After reindexing, `_post_process_json()` walks `body.children` in order and maintains a hierarchy stack:

```
1. Regex: ^(\d+(?:\.\d+)*)\s+(.+)$
   Matches: "7.4.2 Board Assembly" → num_parts=[7,4,2], title="Board Assembly"

2. Stack management:
   - When encountering "7.4.2", pop stack until depth < 3
   - Push ([7,4,2], "Board Assembly")
   - Stack is now: [([7], "Troubleshooting"), ([7,4], "Display"), ([7,4,2], "Board Assembly")]

3. Every element gets:
   - hierarchy_path: "7/4/2"
   - hierarchy_title: "Troubleshooting>Display>Board Assembly"
```

Elements before the first numbered header (cover page, TOC, etc.) get empty strings.

Group children (e.g., list items inside a `list` group) are also injected — the function recurses into group children to ensure every leaf element gets hierarchy metadata.

### 6.2 Where It Runs
- Single/batch export: `_export_result_from_doc()` line 943, after `_reindex_json_reading_order`
- Bundled export: `_run_bundled_conversion()` line 2133, after reindex, before bundle metadata injection

---

## 7. Gemini Vision Enrichment

### 7.1 Purpose
**`enricher.py`, 421 lines**

For each picture in the document, we send the cropped image + surrounding document context to **Gemini 2.5 Flash** via Vertex AI. Gemini returns a structured description:

```
PURPOSE: One sentence — what the image is showing.
COMPONENTS: Comma-separated labeled parts, callouts, part numbers.
VALUES: Measurements, thresholds, settings visible. "None" if absent.
DESCRIPTION: 2-3 sentences for a reader, grounded in document context.
```

### 7.2 Context Building
The enricher doesn't just send the image — it provides Gemini with:

1. **Document filename**: e.g., "Carestation_620_Manual.pdf"
2. **Page number**: From the picture's provenance
3. **Classification**: From Docling's picture classifier (figure, chart, photograph, logo, etc.)
4. **Nearest heading**: Walks backward in the flat element list to find the closest `section_header`
5. **Surrounding text**: All text/table content from the nearest heading above to the next heading below, labeled as `[BEFORE IMAGE]` and `[AFTER IMAGE]`
6. **Caption**: From `pic_item.caption_text(doc)`

This context grounding is critical — it prevents Gemini from hallucinating part numbers or descriptions not present in the document.

### 7.3 Concurrency and Rate Limiting

```python
sem = asyncio.Semaphore(concurrency)  # Default: 10 concurrent calls
```

Each picture is processed as an async task. The semaphore caps concurrent Gemini API calls to avoid 429 rate limits. The `@retry` decorator with `tenacity` handles rate limit errors (429, "resource exhausted", "quota") with exponential backoff, up to 6 attempts.

### 7.4 Token Tracking and Cost
Every Gemini response includes `usage_metadata.prompt_token_count` and `candidates_token_count`. We sum these across all pictures and compute cost:

```python
cost = (input_tokens / 1M) * $0.30 + (output_tokens / 1M) * $3.00
```

The prompt log (every prompt + response) is saved to `prompts.json` and downloadable from the UI.

### 7.5 Parallel with Image Saving
**`app.py` lines 1217-1257 (standard), 2079-2118 (bundled)**

Gemini enrichment and image saving run in parallel using a 2-thread pool:

```python
with ThreadPoolExecutor(max_workers=2) as pool:
    f_enrich = pool.submit(_gemini_enrich, doc, filename)
    f_images = pool.submit(_save_item_images, doc, out_dir, source)
    image_path_map = f_images.result()
    n_enriched, prompt_log, token_stats = f_enrich.result()
```

This saves significant time since Gemini calls are I/O-bound (network) while image cropping is CPU-bound.

---

## 8. Image Extraction

### 8.1 Pictures
**`app.py` lines 878-888**

Pictures use Docling's built-in `pic.get_image(doc)` which returns the PIL image stored during conversion (when `generate_picture_images=True`). Saved as PNG to `pictures/picture_{idx}.png`.

### 8.2 Tables
**`app.py` lines 889-924**

Tables don't have stored images in the DoclingDocument. We re-render the PDF page using **pypdfium2** at 2x scale, then crop the table region using its bounding box.

**Coordinate conversion** (`_crop_from_page`, lines 839-860): Docling uses BOTTOMLEFT coordinate origin (`higher t = higher on page`), but PIL uses TOPLEFT. The conversion:

```python
if "BOTTOMLEFT" in origin:
    t, b = page_h - bbox.t, page_h - bbox.b
```

Then scale by the ratio of rendered image size to PDF page size:

```python
sx = img_w / page_w  # scale factor
left = min(l, r) * sx
top = min(t, b) * sy
```

### 8.3 Image Paths in JSON
After saving, `image_path_map` maps `self_ref → relative_path` (e.g., `"#/pictures/5" → "pictures/picture_5.png"`). These are injected into the JSON output as `image_path` fields on each picture and table element.

---

## 9. The Bundled Conversion Pipeline (End-to-End)

Here is the complete flow of `_run_bundled_conversion()` (lines 1665-2228):

### Step 1: Plan Bundles (line 1729-1763)
- `plan_bundles(source, max_pages=50)` — PyMuPDF TOC extraction
- If no TOC: falls back to `_run_conversion()` (standard mode)
- Sends `bundle_plan` timing event + initial "pending" status for each bundle

### Step 2: Convert Bundles (lines 1766-2026)
Three modes as described in Section 3.3. Each bundle produces a JSON file on disk (`_bundle_000.json`, `_bundle_001.json`, ...).

Key: reorder is NOT run per-bundle. This was a deliberate decision — `_merge_orphaned_list_descriptions()` needs the full document to correctly merge orphans that might reference list items in other bundles. Comment at line 184:

```python
# Note: reorder (including orphan merge) runs AFTER concatenate()
# in the main process, not here.
```

### Step 3: Merge (lines 2033-2051)
```python
bundle_docs = [DoclingDocument.load_from_json(filename=str(bp)) for bp in bundle_doc_paths]
merged_doc = DoclingDocument.concatenate(bundle_docs)
del bundle_docs; gc.collect()
```

`DoclingDocument.concatenate()` (from `docling-core`) handles:
- Re-indexing `self_ref` / `parent` / `children` references across documents
- Shifting page numbers so they're globally unique
- Preserving all element content and structure

### Step 3b: Reorder Merged Doc (lines 2053-2057)
Now that we have the full document, run `_reorder_body_children(merged_doc)` which:
1. Merges orphaned list descriptions (needs full doc context)
2. Sorts body.children by spatial reading order

### Step 4: Build page→bundle Lookup (lines 2060-2066)
```python
page_to_bundle = {}
for b in bundles:
    for p in range(b.page_start, b.page_end + 1):
        page_to_bundle[p] = b
```

### Step 5: Report (lines 2068-2075)
`_build_report(merged_doc)` extracts all countable metadata: element counts by type, table cell stats, picture classification distribution, page coverage, heading levels, text formatting, etc.

### Step 6: Gemini Enrich + Image Save (lines 2077-2125)
Run in parallel (2 threads). Gemini enrichment operates on the merged doc so it has full document context for surrounding text extraction.

### Step 7: Export (lines 2127-2168)
For JSON format:
1. `merged_doc.model_dump(mode='json')` — serialize to dict
2. `_reindex_json_reading_order(d)` — renumber arrays to match reading order
3. `_post_process_json(d)` — inject `hierarchy_path` and `hierarchy_title`
4. Inject `image_path` on pictures and tables
5. Inject `bundle` metadata on every element (from `page_to_bundle` lookup)
6. Add `bundle_plan` array at top level

### Step 8: Write File (lines 2171-2178)
Write to `outputs/{job_id}/result.json`. Track size in KB.

### Step 9: Chunking (lines 2180-2206)
Optional. Uses Docling's `HybridChunker` on the merged doc.

### Cleanup (lines 2208-2228)
Delete temp bundle JSON files. Send total timing. Set job status.

---

## 10. Real-Time Streaming (SSE)

### 10.1 Queue-Based Architecture
Each job has a `queue.Queue()`. The conversion thread pushes messages with prefixed tags:

| Prefix | SSE Event | Purpose |
|--------|-----------|---------|
| `__TIMING__:` | `timing` | Stage durations, page counts, token costs |
| `__INFO__:` | `info` | Pipeline config, device, model names |
| `__REPORT__:` | `report` | Document analysis report |
| `__BUNDLE_STATUS__:` | `bundle_status` | Per-bundle pending/converting/done/error |
| `__FILE_STATUS__:` | `file_status` | Per-file status (multi-file mode) |
| `__GEMINI__:` | `gemini_log` | Gemini enrichment progress |
| `__INFO_BATCH__:` | `log` | Informational messages |
| (anything else) | `log` | Docling pipeline logs |
| `None` | `done`/`error` | Sentinel — job complete |

### 10.2 Log Capture
**`app.py` lines 957-1008**

`_JobFilter` ensures only logs from the correct job's thread (or Docling's internal `Stage-*` threads) are captured. `_QueueHandler` pushes formatted log records into the job's queue without blocking.

### 10.3 Frontend Handling
**`static/app.js`**

The `EventSource` API subscribes to `/stream/{job_id}`. Each event type has a dedicated handler:

- `handleTiming(data)`: Builds the timing panel with stage-by-stage breakdown
- `handleInfo(data)`: Shows device, model, pipeline config
- `handleReport(data)`: Renders the document report table
- `handleBundleStatus(data)`: Creates/updates the bundle status table with animated badges
- `updateFileStatus(data)`: Multi-file per-row status updates

---

## 11. Document Report

### 11.1 What It Contains
**`app.py` lines 316-550, `_build_report()`**

The report extracts every countable metric from the DoclingDocument:

- **Overview**: filename, MIME type, page count, page dimensions, pages with rendered images
- **Elements by label**: text (641), list_item (189), page_footer (118), section_header (57), page_header (51), caption (20), picture (70), table (25), etc.
- **Heading levels**: L1 count, L2 count, etc.
- **Table detail**: total cells, column/row headers, merged cells, fillable cells, average table size, tables with captions
- **Picture detail**: with image data, with caption, with description (from Gemini), classification labels
- **Text formatting**: bold, italic, underline, strikethrough, hyperlinks
- **List items**: enumerated vs bulleted count
- **Structure**: elements with bbox, with parent ref, unique parents, content layers, groups by label
- **Pages coverage**: pages with content, average elements per page
- **Key-value / Form detail**: regions and cell counts

---

## 12. Converter Construction

### 12.1 _build_converter()
**`app.py` lines 1010-1109**

Shared by all conversion paths. Constructs a `DocumentConverter` with:

- **Backend**: `DoclingParseDocumentBackend` (default) or `PyPdfiumDocumentBackend`
- **Pipeline**: Standard (`PdfPipelineOptions`) or VLM (`VlmPipelineOptions`)
- **Batch sizes**: `layout_batch_size`, `table_batch_size`, `ocr_batch_size` — controls how many pages go through each model in one forward pass. Directly controls GPU VRAM peak.
- **Queue size**: `queue_max_size` — max pages in-flight between pipeline stages. Lower = less memory, higher = better throughput.
- **Accelerator**: Auto-detect CUDA or force CPU/CUDA
- **Table mode**: Accurate (TableFormer) or Fast
- **Picture options**: `generate_picture_images=True` when Gemini enrich or save_images is enabled

### 12.2 Model Presets
**Lines 218-235**

Two preset registries:

- **PIC_DESC_PRESETS**: SmolVLM-256M, Granite-Vision-3.3-2B, Pixtral-12B, Qwen2.5-VL-3B
- **VLM_PRESETS**: Granite-Docling-258M, SmolDocling-256M, Granite-Vision-3.2-2B, Phi-4-multimodal, Dolphin, GOT-OCR-2.0

The `/model-status` endpoint checks the HuggingFace cache to report which models are already downloaded.

---

## 13. What Runs on CPU vs GPU

When GPU device is selected:

| Component | Device | Why |
|-----------|--------|-----|
| Layout model (RT-DETR / DocLayNet) | **GPU** | PyTorch inference, benefits from CUDA |
| TableFormer (table structure) | **GPU** | PyTorch inference |
| OCR (RapidOCR) | **CPU** | RapidOCR always runs on CPU, even with GPU mode |
| C++ PDF parser (docling-parse) | **CPU** | C++ native code, no CUDA |
| Preprocessing | **CPU** | Image preparation, page rendering |
| Assembly / Reading order | **CPU** | Tree building, sorting — pure Python |
| Gemini enrichment | **Network** | API calls to Vertex AI |
| Image extraction (pypdfium2) | **CPU** | PDF page rendering |

Roughly 40-50% of work still runs on CPU even with GPU mode.

---

## 14. The JSON Output Structure

After all post-processing, the JSON output for a bundled conversion contains:

```json
{
  "schema_name": "DoclingDocument",
  "version": "1.10.0",
  "origin": { "filename": "...", "mimetype": "application/pdf", "binary_hash": ... },
  "body": {
    "children": [
      { "cref": "#/texts/0" },
      { "cref": "#/texts/1" },
      { "cref": "#/pictures/0" },
      ...
    ]
  },
  "texts": [
    {
      "self_ref": "#/texts/0",
      "label": "section_header",
      "text": "1 Introduction",
      "prov": [{ "page_no": 3, "bbox": { "l": ..., "t": ..., "r": ..., "b": ..., "coord_origin": "BOTTOMLEFT" } }],
      "hierarchy_path": "1",
      "hierarchy_title": "Introduction",
      "bundle": { "id": "b001_introduction", "name": "Introduction", ... }
    },
    ...
  ],
  "tables": [
    {
      "self_ref": "#/tables/0",
      "label": "table",
      "data": { "table_cells": [...], "num_rows": 29, "num_cols": 2 },
      "image_path": "tables/table_0.png",
      "hierarchy_path": "1",
      "hierarchy_title": "Introduction",
      "bundle": { ... }
    },
    ...
  ],
  "pictures": [
    {
      "self_ref": "#/pictures/0",
      "label": "picture",
      "image_path": "pictures/picture_0.png",
      "hierarchy_path": "1/6",
      "hierarchy_title": "Introduction>Display control",
      "bundle": { ... }
    },
    ...
  ],
  "pages": { "1": { "page_no": 1, "size": { "width": 612.0, "height": 792.0 } }, ... },
  "bundle_plan": [
    { "id": "b000_front_matter", "name": "Front Matter", "page_start": 1, "page_end": 2, ... },
    { "id": "b001_introduction", "name": "Introduction", "page_start": 3, "page_end": 50, ... },
    ...
  ]
}
```

Every element has:
- **`hierarchy_path`**: "7/4/2" — numeric path for programmatic navigation
- **`hierarchy_title`**: "Troubleshooting>Display>Board Assembly" — human-readable breadcrumb
- **`bundle`**: Which bundle this element came from (when bundle mode is used)
- **`image_path`**: Relative path to cropped image (for pictures and tables, when save_images is enabled)
- **`prov`**: Page number and bounding box coordinates

---

## 15. Frontend

### 15.1 UI Controls
**`static/index.html`**

The UI exposes every backend parameter:
- Pipeline (Standard / VLM), Device (Auto / CPU / GPU), PDF Backend (DoclingParse / PyPdfium)
- OCR toggle, Table mode (Accurate / Fast), Picture description toggle
- Page range, Queue size, Layout/Table/OCR batch sizes
- Bundle toggle → reveals: Max pages/bundle, Model reload toggle, Parallel toggle
- Save images, Free VRAM, Gemini Enrich, Chunk toggles
- Export format (MD / HTML / JSON / DocTags)

### 15.2 Real-Time Updates
**`static/app.js`**

- **Timing panel**: Stage-by-stage breakdown with durations
- **Bundle status table**: Scrollable table with animated badges (pending → converting → done/error)
- **File status table**: For multi-file mode with click-to-view individual results
- **Log console**: Docling pipeline logs in real-time
- **Gemini console**: Separate terminal for enrichment progress
- **Document report table**: Rendered from report data with copy-to-clipboard

---

## 16. Validated Results

We tested with a 484-page medical equipment manual (Carestation 620/650/650c Technical Reference Manual):

- **10,127 text elements, 211 tables, 380 pictures, 338,191 total characters** — matched exactly between bundled and non-bundled output
- **Zero content loss** from bundle splitting + merging
- **Memory**: Subprocess mode maintained constant memory per bundle. GPU VRAM stayed flat. CPU RAM bounded by max_pages_per_bundle.
- **Hierarchy injection**: Correct for all 43 numbered sections across 3 levels of nesting

---

## 17. File Index

| File | Lines | Purpose |
|------|-------|---------|
| `app.py` | ~2680 | FastAPI backend — all conversion logic, endpoints, cleanup, post-processing |
| `bundler.py` | 328 | TOC extraction (PyMuPDF), section tree building, bundle planning |
| `enricher.py` | 421 | Gemini Vision enrichment — prompts, context building, async processing |
| `reorder.py` | 251 | Standalone reading-order correction (no app.py dependencies) |
| `static/index.html` | 273 | UI markup — dropzone, options panel, status tables, report section |
| `static/app.js` | ~750 | Client-side logic — SSE handling, timing panel, file/bundle status |
| `static/style.css` | ~400 | Dark theme design system — badges, tables, timing panel, console |
