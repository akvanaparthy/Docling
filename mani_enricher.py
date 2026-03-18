"""
pdf_router/enricher.py
======================
Post-processing step: takes *combined_output.json* produced by the router,
fires async Gemini vision / text calls via Vertex AI, and writes
*enriched_output.json* with three new metadata fields:

  image_description  – natural-language description of each Image element
  table_summary      – 1-2 sentence plain-English summary of each Table
  table_markdown     – HTML table converted to GitHub-flavoured Markdown

All Gemini calls are launched simultaneously (asyncio.gather) and throttled
by asyncio.Semaphore(concurrency) — fires up to N calls at once, much faster
than a QPM/time-window limiter. Exponential-backoff retry on 429 / ResourceExhausted
handles any real quota errors. Typical wall time: 20-40s for ~636 calls.

  Old approach (commented out): AsyncLimiter QPM-based throttle (was slower).

Usage (CLI)::

    python -m pdf_router.enricher \\
        --input  test-files/outputs/combined_output.json \\
        --output test-files/outputs/enriched_output.json

Usage (Python)::

    from pdf_router.enricher import enrich
    import asyncio
    elements = asyncio.run(enrich())
"""

import argparse
import asyncio
import base64
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# from aiolimiter import AsyncLimiter  # replaced by Semaphore (QPM limiter was slower)
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from google import genai
from google.genai import types

from .config import (
    ENRICHER_CONCURRENCY,
    # ENRICHER_QPM,  # replaced by Semaphore
    GEMINI_MODEL,
    IMAGE_MODEL,
    OUTPUT_DIR,
    TABLE_MODEL,
    VERTEX_LOCATION,
    VERTEX_PROJECT,
)

log = logging.getLogger(__name__)

# ── Rate-limited Gemini call with retry ───────────────────────────────────────

def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True for 429 / ResourceExhausted errors that should be retried."""
    msg = str(exc).lower()
    return "429" in msg or "resource exhausted" in msg or "quota" in msg


@retry(
    retry=retry_if_exception(_is_rate_limit_error),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(6),
    reraise=True,
)
async def _call_gemini(
    client: genai.Client,
    model: str,
    contents: Any,
) -> Any:
    """Single Gemini generate_content call with exponential-backoff retry on 429."""
    return await client.aio.models.generate_content(
        model=model,
        contents=contents,
    )


# ── Section context helpers ────────────────────────────────────────────────────

_TITLE_TYPES = {"Title", "Header"}
_TEXT_TYPES  = {"NarrativeText", "ListItem", "FigureCaption", "Text",
                "UncategorizedText", "Title", "Header", "Table"}


def _build_id_maps(
    elements: List[Dict[str, Any]],
) -> tuple:
    """
    Build two lookup dicts over the full element list:
      id_to_idx : element_id  → list index
      id_to_el  : element_id  → element dict
    Built once in ``enrich()`` and shared across all worker calls.
    """
    id_to_idx: Dict[str, int] = {}
    id_to_el:  Dict[str, Dict[str, Any]] = {}
    for i, el in enumerate(elements):
        eid = el.get("element_id", "")
        if eid:
            id_to_idx[eid] = i
            id_to_el[eid]  = el
    return id_to_idx, id_to_el


def _find_section_root_id(
    element: Dict[str, Any],
    id_to_el: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """
    Walk up the parent_id chain from *element*, skipping Header nodes,
    and return the element_id of the first ancestor whose type is "Title".
    Returns None if no Title ancestor is found.

    This correctly handles sub-sections: an image whose immediate parent is
    a Header (e.g. "10.2 Sub-section") keeps walking until it reaches the
    top-level Title (e.g. "10 Chapter").
    """
    current = element
    visited: set = set()
    while True:
        pid = current.get("metadata", {}).get("parent_id")
        if not pid or pid in visited:
            return None
        visited.add(pid)
        parent = id_to_el.get(pid)
        if parent is None:
            return None
        ptype = parent.get("type", "")
        if ptype == "Title":
            return pid          # found the section root
        if ptype == "Header":
            current = parent    # keep climbing
        else:
            return pid          # unexpected type — treat as root


def _section_title(
    element: Dict[str, Any],
    id_to_el: Dict[str, Dict[str, Any]],
) -> str:
    """
    Return the text of the Title ancestor for *element* via parent_id chain.
    Falls back to 'Unknown section' if no Title ancestor exists.
    """
    root_id = _find_section_root_id(element, id_to_el)
    if root_id:
        return (id_to_el[root_id].get("text") or "Unknown section").strip()
    return "Unknown section"


def _section_context(
    elements: List[Dict[str, Any]],
    element_pos: int,
    id_to_idx: Dict[str, int],
    id_to_el: Dict[str, Dict[str, Any]],
) -> str:
    """
    Collect all text context for the element at *element_pos*.

    BEFORE the element
    ------------------
    Use the parent_id chain to find the section root Title index, then
    walk forward to *element_pos* collecting every text/table element.
    This gives the full section narrative, sub-headers, steps and tables
    that precede the image/table regardless of page boundaries.

    AFTER the element
    -----------------
    Walk forward from *element_pos + 1* collecting text/table elements
    and stop only when a top-level "Title" is encountered (NOT "Header").
    This means sub-headings after the image are traversed through,
    so a procedure step or caption that follows the image is always included.

    Returns a plain-text block (before + after joined), or empty string.
    """
    element = elements[element_pos]

    # ── Section start via parent_id chain ────────────────────────────────────
    root_id = _find_section_root_id(element, id_to_el)
    if root_id and root_id in id_to_idx:
        section_start = id_to_idx[root_id]
    else:
        # Fallback: backwards linear walk for elements with no parent_id
        section_start = 0
        for i in range(element_pos - 1, -1, -1):
            if elements[i].get("type") == "Title":
                section_start = i
                break

    # ── Text BEFORE the element (section_start → element_pos, exclusive) ─────
    before: List[str] = []
    for i in range(section_start, element_pos):
        el = elements[i]
        if el.get("type") in _TEXT_TYPES:
            t = (el.get("text") or "").strip()
            if t:
                before.append(t)

    # ── Text AFTER the element (element_pos+1 → next Title, exclusive) ───────
    # Stop only at "Title" — "Header" nodes are traversed through so that
    # sub-section text and tables after the image are included.
    after: List[str] = []
    for i in range(element_pos + 1, len(elements)):
        el = elements[i]
        if el.get("type") == "Title":   # new top-level section — stop
            break
        if el.get("type") in _TEXT_TYPES:
            t = (el.get("text") or "").strip()
            if t:
                after.append(t)

    return "\n".join(before + after)


# ── RAG-optimised prompt builders ─────────────────────────────────────────────

def _image_prompt(element: Dict[str, Any], section: str, section_body: str) -> str:
    """Build a structured RAG-optimised vision prompt for an Image element."""
    meta  = element.get("metadata", {})
    page  = meta.get("page_number", "?")
    fname = Path(meta.get("filename", "document.pdf")).name

    context_section = (
        f"--- BEGIN DOCUMENT CONTEXT ---\n{section_body.strip()}\n--- END DOCUMENT CONTEXT ---"
        if section_body.strip()
        else "(No surrounding text available for this image.)"
    )

    return (
        f"You are indexing a page from a medical equipment service manual for a RAG system.\n"
        f"Technicians will search this index with questions like:\n"
        f'  "how do I replace X", "what is part number Y", "where is component Z"\n\n'
        f"Image location: page {page} of \'{fname}\'\n"
        f"Parent section: {section}\n\n"
        f"The following text appears in the same section of the document as this image.\n"
        f"Use it to ground your description — prefer terminology and values from this context:\n\n"
        f"{context_section}\n\n"
        "IMPORTANT RULES:\n"
        "- Only describe what is VISIBLE in the image or CONFIRMED by the context above.\n"
        "- Do NOT invent part numbers, measurements, or procedures not present in the image or context.\n"
        "- If a callout or label is visible, include it exactly as shown.\n"
        "- If something is unclear in the image, say so rather than guessing.\n\n"
        "Respond in this exact structure (no extra text before or after):\n\n"
        "PURPOSE: One sentence — what this diagram or photo is showing or instructing.\n"
        "COMPONENTS: Comma-separated list of every labeled part, callout number, "
        "reference code (e.g. AC.24.263), and part number visible in the image.\n"
        "VALUES: Any measurements, settings, pressures, voltages, flow rates, or thresholds "
        "visible. Write \"None\" if absent.\n"
        "DESCRIPTION: 2-3 sentences describing the technical content for a service technician, "
        "grounded in the document context above."
    )


def _table_prompt(element: Dict[str, Any], section: str, section_body: str) -> str:
    """Build a structured RAG-optimised prompt for a Table element."""
    meta  = element.get("metadata", {})
    page  = meta.get("page_number", "?")
    fname = Path(meta.get("filename", "document.pdf")).name
    html  = meta.get("text_as_html", "")

    context_section = (
        f"--- BEGIN DOCUMENT CONTEXT ---\n{section_body.strip()}\n--- END DOCUMENT CONTEXT ---"
        if section_body.strip()
        else "(No surrounding text available for this table.)"
    )

    return (
        f"You are indexing a table from a medical equipment service manual for a RAG system.\n"
        f"Technicians will search this index with questions like:\n"
        f'  "what is the stock number for X", "which parts are used in Y procedure"\n\n'
        f"Table location: page {page} of \'{fname}\'\n"
        f"Parent section: {section}\n\n"
        f"The following text appears in the same section of the document as this table.\n"
        f"Use it to understand the purpose and context of the table:\n\n"
        f"{context_section}\n\n"
        f"TABLE HTML:\n{html}\n\n"
        "Respond in this exact structure (no extra text before or after):\n\n"
        "PURPOSE: One sentence — what this table is for.\n"
        "KEY_ITEMS: Comma-separated list of all stock numbers and item names in the table.\n"
        "SUMMARY: 1-2 sentences a technician could match against a search query.\n"
        "MARKDOWN: The table in GitHub-flavoured Markdown."
    )


# ── Response parsers ──────────────────────────────────────────────────────────

_SECTION_KEYS = {"PURPOSE", "COMPONENTS", "VALUES", "DESCRIPTION",
                 "KEY_ITEMS", "SUMMARY", "MARKDOWN"}


def _parse_structured(text: str) -> Dict[str, str]:
    """
    Generic parser for structured Gemini responses.

    Handles multi-line sections like::

        PURPOSE: One sentence.
        COMPONENTS: part1, part2, AC.24.263
        VALUES: None
        DESCRIPTION: Two or three
          sentences here.

    Returns a dict keyed by section label (uppercase), values stripped.
    Falls back to ``{"DESCRIPTION": full_text}`` if no known keys found.
    """
    result: Dict[str, str] = {}
    current: Optional[str] = None
    buf: List[str] = []

    for line in text.splitlines():
        # detect a new labelled section ("KEY: ...")  
        matched_key = None
        for key in _SECTION_KEYS:
            if line.startswith(f"{key}:"):
                matched_key = key
                break

        if matched_key:
            if current is not None:
                result[current] = "\n".join(buf).strip()
            current = matched_key
            buf = [line[len(matched_key) + 1:].strip()]
        else:
            buf.append(line)

    if current is not None:
        result[current] = "\n".join(buf).strip()

    # fallback
    if not result:
        result["DESCRIPTION"] = text.strip()

    return result


# ── Per-element async workers ──────────────────────────────────────────────────

async def _enrich_image(
    client: genai.Client,
    element: Dict[str, Any],
    all_elements: List[Dict[str, Any]],
    element_pos: int,
    sem: asyncio.Semaphore,
    # limiter: AsyncLimiter,  # QPM limiter was slower than Semaphore
    model: str,
    id_to_idx: Dict[str, int],
    id_to_el: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Describe a single Image element using Gemini vision (RAG-optimised)."""
    meta    = element.get("metadata", {})
    b64     = meta.get("image_base64", "")
    mime    = meta.get("image_mime_type", "image/png")
    section = _section_title(element, id_to_el)
    body    = _section_context(all_elements, element_pos, id_to_idx, id_to_el)
    prompt  = _image_prompt(element, section, body)

    async with sem:
        # async with limiter:  # QPM limiter was slower than Semaphore
        try:
            response = await _call_gemini(
                client, model,
                [
                    types.Part.from_bytes(
                        data=base64.b64decode(b64),
                        mime_type=mime,
                    ),
                    types.Part.from_text(text=prompt),
                ],
            )
            parsed = _parse_structured(response.text)
            meta["image_section"]    = section
            meta["image_purpose"]    = parsed.get("PURPOSE", "")
            meta["image_components"] = parsed.get("COMPONENTS", "")
            meta["image_values"]     = parsed.get("VALUES", "")
            meta["image_description"] = parsed.get("DESCRIPTION", "")
        except Exception as exc:                                    # noqa: BLE001
            log.warning(
                "Image enrich failed (page %s, file %s): %s",
                meta.get("page_number"),
                meta.get("filename"),
                exc,
            )
            for key in ("image_section", "image_purpose", "image_components",
                        "image_values", "image_description"):
                meta.setdefault(key, "")

    return element


async def _enrich_table(
    client: genai.Client,
    element: Dict[str, Any],
    all_elements: List[Dict[str, Any]],
    element_pos: int,
    sem: asyncio.Semaphore,
    # limiter: AsyncLimiter,  # QPM limiter was slower than Semaphore
    model: str,
    id_to_idx: Dict[str, int],
    id_to_el: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Summarise and convert a Table element using Gemini (RAG-optimised)."""
    meta    = element.get("metadata", {})
    section = _section_title(element, id_to_el)
    body    = _section_context(all_elements, element_pos, id_to_idx, id_to_el)
    prompt  = _table_prompt(element, section, body)

    async with sem:
        # async with limiter:  # QPM limiter was slower than Semaphore
        try:
            response = await _call_gemini(client, model, prompt)
            parsed = _parse_structured(response.text)
            meta["table_section"]  = section
            meta["table_purpose"]  = parsed.get("PURPOSE", "")
            meta["table_key_items"] = parsed.get("KEY_ITEMS", "")
            meta["table_summary"]  = parsed.get("SUMMARY", "")
            meta["table_markdown"] = parsed.get("MARKDOWN", "")
        except Exception as exc:                                    # noqa: BLE001
            log.warning(
                "Table enrich failed (page %s, file %s): %s",
                meta.get("page_number"),
                meta.get("filename"),
                exc,
            )
            for key in ("table_section", "table_purpose", "table_key_items",
                        "table_summary", "table_markdown"):
                meta.setdefault(key, "")

    return element


# ── Main coroutine ─────────────────────────────────────────────────────────────

async def enrich(
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    elements: Optional[List[Dict[str, Any]]] = None,
    project: str = VERTEX_PROJECT,
    location: str = VERTEX_LOCATION,
    image_model: str = IMAGE_MODEL,
    table_model: str = TABLE_MODEL,
    concurrency: int = ENRICHER_CONCURRENCY,
    # qpm: int = ENRICHER_QPM,  # replaced by Semaphore
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Enrich Image and Table elements via Gemini and write *output_path*.

    Parameters
    ----------
    input_path:
        Path to combined_output.json.  Ignored when *elements* is supplied.
        Defaults to OUTPUT_DIR/combined_output.json.
    output_path:
        Path for enriched_output.json (defaults to OUTPUT_DIR/enriched_output.json).
    elements:
        Pre-loaded element list.  When provided the file-load step is skipped
        (used by router.py to avoid a redundant round-trip through disk).
    project:
        GCP project ID for Vertex AI (default: VERTEX_PROJECT from config).
    location:
        Vertex AI region (default: VERTEX_LOCATION from config).
    image_model:
        Gemini model for image description (default: IMAGE_MODEL from config).
    table_model:
        Gemini model for table summarisation (default: TABLE_MODEL from config).
    concurrency:
        Max simultaneous Gemini calls (Semaphore). Higher = faster.
        Default 300 fires 300 calls at once; retry handles any real 429s.

    Returns
    -------
    (elements, stats)
        *elements* — enriched list (mutated in-place).
        *stats*    — dict with keys: images_ok, images_total, tables_ok,
                     tables_total, wall_time_sec.

    New metadata fields written per element
    ---------------------------------------
    Images: image_section, image_purpose, image_components, image_values, image_description
    Tables: table_section, table_purpose, table_key_items, table_summary, table_markdown
    """
    input_path  = input_path  or str(Path(OUTPUT_DIR) / "combined_output.json")
    output_path = output_path or str(Path(OUTPUT_DIR) / "enriched_output.json")

    if elements is None:
        log.info("Loading %s …", input_path)
        with open(input_path, encoding="utf-8") as fh:
            elements = json.load(fh)
    else:
        log.info("Using %d pre-loaded elements (skipping file read)", len(elements))

    # ── Bucket elements (keep position in full list for section lookup) ───────
    image_idx: List[int] = []
    table_idx: List[int] = []

    for idx, el in enumerate(elements):
        el_type = el.get("type", "")
        meta    = el.get("metadata", {})
        if el_type == "Table" and meta.get("text_as_html"):
            table_idx.append(idx)
        elif meta.get("image_base64"):
            image_idx.append(idx)

    log.info(
        "Elements: %d total | %d images (%s) | %d tables (%s) | %d pass-through",
        len(elements),
        len(image_idx), image_model,
        len(table_idx), table_model,
        len(elements) - len(image_idx) - len(table_idx),
    )

    _empty_stats: Dict[str, Any] = {
        "images_ok": 0, "images_total": 0,
        "tables_ok": 0, "tables_total": 0,
        "wall_time_sec": 0.0,
    }
    if not image_idx and not table_idx:
        log.info("Nothing to enrich — writing output unchanged.")
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(elements, fh, ensure_ascii=False, indent=2)
        return elements, _empty_stats

    # ── Build element ID lookup maps (used by section context helpers) ────────
    id_to_idx, id_to_el = _build_id_maps(elements)

    # ── Build Vertex AI / Gemini client ──────────────────────────────────────
    log.info(
        "Connecting to Vertex AI: project=%s  location=%s",
        project, location,
    )
    client  = genai.Client(vertexai=True, project=project, location=location)
    sem = asyncio.Semaphore(concurrency)
    # limiter = AsyncLimiter(max_rate=qpm, time_period=60.0)  # QPM limiter was slower

    # ── Launch all calls simultaneously ──────────────────────────────────────
    image_tasks = [
        _enrich_image(client, elements[i], elements, i, sem, image_model,
                      id_to_idx, id_to_el)
        for i in image_idx
    ]
    table_tasks = [
        _enrich_table(client, elements[i], elements, i, sem, table_model,
                      id_to_idx, id_to_el)
        for i in table_idx
    ]

    log.info(
        "Firing %d Gemini calls (concurrency=%d, with retry): %d image→%s  %d table→%s",
        len(image_tasks) + len(table_tasks), concurrency,
        len(image_tasks), image_model,
        len(table_tasks), table_model,
    )
    t0      = time.monotonic()
    results = await asyncio.gather(*image_tasks, *table_tasks, return_exceptions=True)
    wall    = time.monotonic() - t0

    # ── Count successes ───────────────────────────────────────────────────────
    n_img = len(image_idx)
    img_ok = sum(
        1
        for r in results[:n_img]
        if not isinstance(r, Exception)
        and r.get("metadata", {}).get("image_description")
    )
    tbl_ok = sum(
        1
        for r in results[n_img:]
        if not isinstance(r, Exception)
        and r.get("metadata", {}).get("table_summary")
    )

    log.info(
        "Done in %.1fs | images %d/%d | tables %d/%d",
        wall, img_ok, len(image_idx), tbl_ok, len(table_idx),
    )

    # ── Write output ─────────────────────────────────────────────────────────
    log.info("Writing %s …", output_path)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(elements, fh, ensure_ascii=False, indent=2)

    stats: Dict[str, Any] = {
        "images_ok":      img_ok,
        "images_total":   len(image_idx),
        "tables_ok":      tbl_ok,
        "tables_total":   len(table_idx),
        "wall_time_sec":  round(wall, 2),
    }

    # Only print when running standalone (not when called from router)
    log.info(
        "Enrichment: %d/%d images described  |  %d/%d tables summarised  |  %.1fs",
        img_ok, len(image_idx), tbl_ok, len(table_idx), wall,
    )

    return elements, stats


# ── CLI entry point ────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    """Command-line interface for the enricher."""
    parser = argparse.ArgumentParser(
        prog="pdf_router.enricher",
        description=(
            "Enrich combined_output.json with Gemini image descriptions "
            "and table summaries, writing enriched_output.json."
        ),
    )
    parser.add_argument(
        "--input",
        default=None,
        metavar="PATH",
        help="Path to combined_output.json (default: OUTPUT_DIR/combined_output.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Path to write enriched_output.json (default: OUTPUT_DIR/enriched_output.json)",
    )
    parser.add_argument(
        "--project",
        default=VERTEX_PROJECT,
        help="GCP project ID (default: %(default)s)",
    )
    parser.add_argument(
        "--location",
        default=VERTEX_LOCATION,
        help="Vertex AI region (default: %(default)s)",
    )
    parser.add_argument(
        "--image-model",
        default=IMAGE_MODEL,
        help="Gemini model for image description (default: %(default)s)",
    )
    parser.add_argument(
        "--table-model",
        default=TABLE_MODEL,
        help="Gemini model for table summarisation (default: %(default)s)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=ENRICHER_CONCURRENCY,
        metavar="N",
        help="Max simultaneous Gemini calls (default: %(default)s)",
    )
    # parser.add_argument(  # QPM limiter was slower than Semaphore
    #     "--qpm",
    #     type=int,
    #     default=ENRICHER_QPM,
    #     metavar="N",
    #     help="Gemini API queries-per-minute rate limit (default: %(default)s)",
    # )
    args = parser.parse_args(argv)

    _, stats = asyncio.run(
        enrich(
            input_path=args.input,
            output_path=args.output,
            project=args.project,
            location=args.location,
            image_model=args.image_model,
            table_model=args.table_model,
            concurrency=args.concurrency,
            # qpm=args.qpm,  # QPM limiter replaced by Semaphore
        )
    )
    print(
        f"\n── Enrichment Summary ──────────────────────────────\n"
        f"  Output:   {args.output or 'OUTPUT_DIR/enriched_output.json'}\n"
        f"  Images:   {stats['images_ok']}/{stats['images_total']} described  ({args.image_model})\n"
        f"  Tables:   {stats['tables_ok']}/{stats['tables_total']} summarised  ({args.table_model})\n"
        f"  Wall  :   {stats['wall_time_sec']:.1f}s\n"
        f"────────────────────────────────────────────────────\n"
    )


if __name__ == "__main__":
    main()
