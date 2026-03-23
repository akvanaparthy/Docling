"""
enricher.py
===========
Post-processing step for Docling documents.
Sends each picture to Gemini Vision via Vertex AI (ADC) and writes
a structured description back to picture_item.meta.description.

Output format per picture (written as plain text to meta.description.text):
    PURPOSE: One sentence — what the image is showing.
    COMPONENTS: Comma-separated labeled parts, callouts, part numbers.
    VALUES: Measurements, thresholds, settings visible. "None" if absent.
    DESCRIPTION: 2-3 sentences for a reader, grounded in document context.

Config (from .env):
    VERTEX_PROJECT      GCP project ID
    VERTEX_LOCATION     Vertex AI region (default: us-central1)
    GEMINI_MODEL        Model to use (default: gemini-2.0-flash-001)
    ENRICHER_CONCURRENCY  Max concurrent Gemini calls (default: 10)
"""

import asyncio
import base64
import logging
import os
from io import BytesIO
from typing import Optional

from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

log = logging.getLogger(__name__)

VERTEX_PROJECT       = os.environ.get("VERTEX_PROJECT", "")
VERTEX_LOCATION      = os.environ.get("VERTEX_LOCATION", "us-central1")
GEMINI_MODEL         = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-001")
IMAGE_MODEL          = os.environ.get("IMAGE_MODEL", GEMINI_MODEL)
TABLE_MODEL          = os.environ.get("TABLE_MODEL", GEMINI_MODEL)
ENRICHER_CONCURRENCY = int(os.environ.get("ENRICHER_CONCURRENCY", "10"))

# Gemini 2.5 Flash pricing ($/1M tokens) — override via .env if rates change
# Source: GCP official pricing for gemini-2.5-flash
# Input  (text, image, video): $0.30/1M — same for <=200K and >200K context
# Output (text + reasoning):   $3.00/1M — same for <=200K and >200K context
GEMINI_INPUT_PRICE_PER_M  = float(os.environ.get("GEMINI_INPUT_PRICE_PER_M",  "0.30"))
GEMINI_OUTPUT_PRICE_PER_M = float(os.environ.get("GEMINI_OUTPUT_PRICE_PER_M", "3.00"))

_TYPE_HINTS = {
    "figure":      "diagram or technical figure",
    "chart":       "chart or graph",
    "photograph":  "photograph",
    "logo":        "logo or brand mark",
    "barcode":     "barcode or QR code",
    "signature":   "signature",
    "molecule":    "molecular structure diagram",
    "map":         "map or spatial diagram",
}

_HEADING_LABELS = {"section_header", "title"}
_TEXT_LABELS    = {"paragraph", "text", "list_item", "caption",
                   "section_header", "title", "footnote"}
_TABLE_LABEL    = "table"
_SECTION_KEYS   = {"PURPOSE", "COMPONENTS", "VALUES", "DESCRIPTION",
                    "KEY_ITEMS", "SUMMARY", "MARKDOWN"}


# ── Gemini call with retry ─────────────────────────────────────────────────────

def _is_rate_limit(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "resource exhausted" in msg or "quota" in msg


@retry(
    retry=retry_if_exception(_is_rate_limit),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(6),
    reraise=True,
)
async def _call_gemini(client, model: str, contents):
    resp = await client.aio.models.generate_content(model=model, contents=contents)
    return resp


# ── Document tree helpers ──────────────────────────────────────────────────────

def _build_ref_map(doc) -> dict:
    ref_map = {}
    for lst in [doc.texts, doc.tables, doc.pictures,
                getattr(doc, "key_value_items", []),
                getattr(doc, "form_items", [])]:
        for item in lst:
            if hasattr(item, "self_ref"):
                ref_map[item.self_ref] = item
    for g in doc.groups:
        ref_map[g.self_ref] = g
    return ref_map


def _flat_order(doc, ref_map) -> list:
    """Depth-first walk of body.children → flat [(ref, item)] list."""
    result = []
    seen = set()

    def walk(cref: str):
        if not cref or cref in seen:
            return
        seen.add(cref)
        item = ref_map.get(cref)
        if item is None:
            return
        # Pictures and tables are leaf nodes for enrichment purposes —
        # their children are captions/footnotes, not sub-elements.
        # Add them directly without recursing into children.
        if cref.startswith("#/pictures/") or cref.startswith("#/tables/"):
            result.append((cref, item))
            return
        if hasattr(item, "children") and item.children:
            for child in item.children:
                walk(child.cref)
        else:
            result.append((cref, item))

    for child in doc.body.children:
        walk(child.cref)
    return result


def _label_value(item) -> str:
    label = getattr(item, "label", None)
    if label is None:
        return ""
    return label.value if hasattr(label, "value") else str(label)


def _nearest_heading(flat: list, pic_pos: int) -> str:
    for i in range(pic_pos - 1, -1, -1):
        _, item = flat[i]
        if _label_value(item) in _HEADING_LABELS:
            return (getattr(item, "text", "") or "").strip()
    return "Unknown section"


def _item_text(item, doc) -> Optional[str]:
    """Extract text from a text or table item."""
    lbl = _label_value(item)
    if lbl == _TABLE_LABEL:
        try:
            md = item.export_to_markdown(doc)
            return f"[TABLE]\n{md.strip()}" if md.strip() else None
        except Exception:
            return None
    if lbl in _TEXT_LABELS:
        return (getattr(item, "text", "") or "").strip() or None
    return None


def _surrounding_text(flat: list, pic_pos: int, doc) -> str:
    """Collect labelled text/tables from nearest heading above to next heading below."""
    # Find heading start
    heading_pos = 0
    for i in range(pic_pos - 1, -1, -1):
        if _label_value(flat[i][1]) in _HEADING_LABELS:
            heading_pos = i
            break

    before, after = [], []

    for i in range(heading_pos, pic_pos):
        _, item = flat[i]
        t = _item_text(item, doc)
        if t:
            before.append(t)

    for i in range(pic_pos + 1, len(flat)):
        _, item = flat[i]
        if _label_value(item) in _HEADING_LABELS:
            break
        t = _item_text(item, doc)
        if t:
            after.append(t)

    parts = []
    if before:
        parts.append("[BEFORE IMAGE]\n" + "\n".join(before))
    if after:
        parts.append("[AFTER IMAGE]\n" + "\n".join(after))
    return "\n\n".join(parts)


def _get_classification(pic_item) -> str:
    try:
        if pic_item.meta and pic_item.meta.classification:
            preds = pic_item.meta.classification.predictions
            if preds:
                return preds[0].class_name.lower()
    except Exception:
        pass
    return "figure"


def _pic_to_b64(pic_item, doc) -> tuple[Optional[str], str]:
    """Extract base64 image data and mime type from a picture item.

    Tries the pre-encoded data URI first (pic_item.image.uri), falling back
    to PIL re-encode only if the URI is missing. Returns (b64_string, mime_type).
    """
    # Try pre-encoded data URI first (avoids PIL re-encode)
    try:
        uri = pic_item.image.uri if pic_item.image else None
        if uri and uri.startswith("data:"):
            # Format: "data:image/png;base64,iVBOR..."
            header, b64_data = uri.split(",", 1)
            mime = header.split(":")[1].split(";")[0]  # "image/png"
            return b64_data, mime
    except Exception:
        pass

    # Fallback: re-encode from PIL
    try:
        pil_img = pic_item.get_image(doc)
        if pil_img is None:
            return None, "image/png"
        buf = BytesIO()
        pil_img.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode(), "image/png"
    except Exception:
        return None, "image/png"


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _build_prompt(classification: str, section: str, context: str,
                  caption: str, page_no: int, filename: str) -> str:
    type_hint = _TYPE_HINTS.get(classification, "image")
    context_block = (
        f"--- BEGIN DOCUMENT CONTEXT ---\n{context.strip()}\n--- END DOCUMENT CONTEXT ---"
        if context.strip()
        else "(No surrounding text available for this image.)"
    )
    caption_line = f"Image Caption: {caption}\n" if caption else ""
    return (
        f"You are indexing a {type_hint} from a document for a RAG system.\n"
        f"Image location: page {page_no} of '{filename}'\n"
        f"Parent section: {section}\n"
        f"{caption_line}"
        f"\nThe following text appears in the same section as this image.\n"
        f"Use it to ground your description:\n\n"
        f"{context_block}\n\n"
        "IMPORTANT RULES:\n"
        "- Only describe what is VISIBLE in the image or CONFIRMED by the context above.\n"
        "- Do NOT invent part numbers, measurements, or procedures not present in the image or context.\n"
        "- If a callout or label is visible, include it exactly as shown.\n"
        "- If something is unclear, say so rather than guessing.\n\n"
        "Respond in this exact structure (no extra text before or after):\n\n"
        "PURPOSE: One sentence — what this image is showing or instructing.\n"
        "COMPONENTS: Comma-separated list of every labeled part, callout number, "
        "reference code, and part number visible in the image.\n"
        "VALUES: Any measurements, settings, pressures, voltages, flow rates, or thresholds "
        "visible. Write \"None\" if absent.\n"
        "DESCRIPTION: 2-3 sentences describing the content for a reader, "
        "grounded in the document context above."
    )


# ── Table prompt builder ──────────────────────────────────────────────────────

def _build_table_prompt(section: str, context: str, markdown: str,
                        page_no, filename: str) -> str:
    """Build a structured RAG-optimised prompt for table summarisation.
    Follows the same pattern as mani_enricher.py _table_prompt."""
    context_block = (
        f"--- BEGIN DOCUMENT CONTEXT ---\n{context.strip()}\n--- END DOCUMENT CONTEXT ---"
        if context.strip()
        else "(No surrounding text available for this table.)"
    )
    return (
        f"You are indexing a table from a document for a RAG system.\n"
        f"Users will search this index with questions about the table contents.\n\n"
        f"Table location: page {page_no} of '{filename}'\n"
        f"Parent section: {section}\n\n"
        f"The following text appears in the same section of the document as this table.\n"
        f"Use it to understand the purpose and context of the table:\n\n"
        f"{context_block}\n\n"
        f"TABLE MARKDOWN:\n{markdown}\n\n"
        "Respond in this exact structure (no extra text before or after):\n\n"
        "PURPOSE: One sentence — what this table is for.\n"
        "KEY_ITEMS: Comma-separated list of important items, part numbers, "
        "codes, or values in the table.\n"
        "SUMMARY: 1-2 sentences a reader could match against a search query."
    )


def _table_to_markdown(table_item, doc) -> Optional[str]:
    """Convert a Docling table item to markdown string."""
    try:
        md = table_item.export_to_markdown(doc)
        return md.strip() if md and md.strip() else None
    except Exception:
        return None


# ── Response parser ────────────────────────────────────────────────────────────

def _parse_response(text: str) -> str:
    """Return full structured text as-is (Option A).
    Falls back gracefully if Gemini didn't follow the format."""
    result = {}
    current = None
    buf = []
    for line in text.splitlines():
        matched = None
        for key in _SECTION_KEYS:
            if line.startswith(f"{key}:"):
                matched = key
                break
        if matched:
            if current:
                result[current] = "\n".join(buf).strip()
            current = matched
            buf = [line[len(matched) + 1:].strip()]
        else:
            buf.append(line)
    if current:
        result[current] = "\n".join(buf).strip()

    if not result:
        return text.strip()

    parts = []
    for key in ("PURPOSE", "COMPONENTS", "VALUES", "DESCRIPTION",
                "KEY_ITEMS", "SUMMARY"):
        if key in result:
            parts.append(f"{key}: {result[key]}")
    return "\n".join(parts)


def _parse_table_response(text: str) -> dict:
    """Parse a table enrichment response into structured fields."""
    result = {}
    current = None
    buf = []
    for line in text.splitlines():
        matched = None
        for key in ("PURPOSE", "KEY_ITEMS", "SUMMARY"):
            if line.startswith(f"{key}:"):
                matched = key
                break
        if matched:
            if current is not None:
                result[current] = "\n".join(buf).strip()
            current = matched
            buf = [line[len(matched) + 1:].strip()]
        else:
            buf.append(line)
    if current is not None:
        result[current] = "\n".join(buf).strip()
    if not result:
        result["SUMMARY"] = text.strip()
    return result


# ── Per-picture async worker ───────────────────────────────────────────────────

async def _enrich_picture(client, doc, pic_item, flat: list,
                          pic_pos: int, sem: asyncio.Semaphore,
                          model: str, filename: str,
                          prompt_log: list, b64_data: str = None,
                          mime_type: str = "image/png"):
    b64 = b64_data
    mime = mime_type
    if not b64:
        b64, mime = _pic_to_b64(pic_item, doc)
    if not b64:
        log.warning("Could not get image for picture %s — skipping", pic_item.self_ref)
        return

    page_no = pic_item.prov[0].page_no if pic_item.prov else "?"
    classification = _get_classification(pic_item)
    section = _nearest_heading(flat, pic_pos)
    context = _surrounding_text(flat, pic_pos, doc)
    caption = ""
    try:
        caption = pic_item.caption_text(doc).strip()
    except Exception:
        pass
    prompt = _build_prompt(classification, section, context, caption, page_no, filename)

    # Log prompt for debugging download
    entry = {
        "picture_ref": pic_item.self_ref,
        "page": page_no,
        "classification": classification,
        "section": section,
        "caption": caption,
        "prompt": prompt,
        "response": None,
        "error": None,
    }
    prompt_log.append(entry)

    async with sem:
        try:
            from google.genai import types
            resp = await _call_gemini(
                client, model,
                [
                    types.Part.from_bytes(
                        data=base64.b64decode(b64),
                        mime_type=mime,
                    ),
                    types.Part.from_text(text=prompt),
                ],
            )
            response_text = resp.text
            usage = resp.usage_metadata
            entry["input_tokens"]  = getattr(usage, "prompt_token_count", 0) or 0
            entry["output_tokens"] = getattr(usage, "candidates_token_count", 0) or 0
            description = _parse_response(response_text)
            entry["response"] = response_text

            # Write back to picture item
            from docling_core.types.doc.document import DescriptionMetaField
            from docling_core.types.doc import PictureMeta
            if pic_item.meta is None:
                pic_item.meta = PictureMeta()
            pic_item.meta.description = DescriptionMetaField(
                text=description,
                created_by=model,
            )
            log.info("Enriched picture %s (page %s, %s)", pic_item.self_ref, page_no, classification)
        except Exception as exc:
            entry["error"] = str(exc)
            log.warning("Gemini enrich failed for picture %s: %s", pic_item.self_ref, exc)


# ── Per-table async worker ────────────────────────────────────────────────────

async def _enrich_table(client, doc, table_item, flat: list,
                        tbl_pos: int, sem: asyncio.Semaphore,
                        model: str, filename: str,
                        prompt_log: list):
    """Summarise a single table using Gemini text (RAG-optimised).

    Converts the table to markdown, collects section context, sends to Gemini,
    and writes back table_summary to the table item's meta field.
    """
    markdown = _table_to_markdown(table_item, doc)
    if not markdown:
        log.warning("Could not export table %s to markdown — skipping", table_item.self_ref)
        return

    page_no = table_item.prov[0].page_no if table_item.prov else "?"
    section = _nearest_heading(flat, tbl_pos)
    context = _surrounding_text(flat, tbl_pos, doc)
    prompt = _build_table_prompt(section, context, markdown, page_no, filename)

    entry = {
        "type": "table",
        "table_ref": table_item.self_ref,
        "page": page_no,
        "section": section,
        "prompt": prompt,
        "response": None,
        "error": None,
    }
    prompt_log.append(entry)

    async with sem:
        try:
            resp = await _call_gemini(client, model, prompt)
            response_text = resp.text
            usage = resp.usage_metadata
            entry["input_tokens"] = getattr(usage, "prompt_token_count", 0) or 0
            entry["output_tokens"] = getattr(usage, "candidates_token_count", 0) or 0
            parsed = _parse_table_response(response_text)
            entry["response"] = response_text

            # Build summary string and write back to table item meta
            summary_parts = []
            for key in ("PURPOSE", "KEY_ITEMS", "SUMMARY"):
                if key in parsed:
                    summary_parts.append(f"{key}: {parsed[key]}")
            summary_text = "\n".join(summary_parts)

            from docling_core.types.doc.document import DescriptionMetaField, FloatingMeta
            if table_item.meta is None:
                table_item.meta = FloatingMeta()
            table_item.meta.description = DescriptionMetaField(
                text=summary_text,
                created_by=model,
            )
            log.info("Enriched table %s (page %s)", table_item.self_ref, page_no)
        except Exception as exc:
            entry["error"] = str(exc)
            log.warning("Gemini table enrich failed for %s: %s", table_item.self_ref, exc)


# ── Main entry point ───────────────────────────────────────────────────────────

async def _enrich_async(doc, filename: str,
                        project: str, location: str,
                        image_model: str, table_model: str,
                        concurrency: int):
    from google import genai

    if not project:
        raise ValueError("VERTEX_PROJECT not set in environment / .env")

    client = genai.Client(vertexai=True, project=project, location=location)
    sem    = asyncio.Semaphore(concurrency)

    ref_map = _build_ref_map(doc)
    flat    = _flat_order(doc, ref_map)

    # Build picture and table position lookups
    pic_positions = {}
    tbl_positions = {}
    for i, (ref, item) in enumerate(flat):
        if ref.startswith("#/pictures/"):
            pic_positions[ref] = i
        elif ref.startswith("#/tables/"):
            tbl_positions[ref] = i

    pictures = [p for p in doc.pictures if p.self_ref in pic_positions]
    tables = [t for t in doc.tables if t.self_ref in tbl_positions
              and _label_value(t) == _TABLE_LABEL]

    if not pictures and not tables:
        log.info("No pictures or tables found — skipping Gemini enrichment")
        return 0, [], {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

    log.info("Gemini enrichment: %d pictures→%s + %d tables→%s (concurrency=%d)",
             len(pictures), image_model, len(tables), table_model, concurrency)

    # Pre-encode all picture images (sync, before async loop)
    pic_b64_cache = {}
    for pic in pictures:
        b64, mime = _pic_to_b64(pic, doc)
        if b64:
            pic_b64_cache[pic.self_ref] = (b64, mime)

    prompt_log: list = []

    # Picture tasks — use IMAGE_MODEL + pre-encoded b64
    pic_tasks = [
        _enrich_picture(client, doc, pic, flat, pic_positions[pic.self_ref],
                        sem, image_model, filename, prompt_log,
                        b64_data=pic_b64_cache.get(pic.self_ref, (None, None))[0],
                        mime_type=pic_b64_cache.get(pic.self_ref, (None, "image/png"))[1])
        for pic in pictures
    ]

    # Table tasks — use TABLE_MODEL (text-only, no vision needed)
    tbl_tasks = [
        _enrich_table(client, doc, tbl, flat, tbl_positions[tbl.self_ref],
                      sem, table_model, filename, prompt_log)
        for tbl in tables
    ]

    await asyncio.gather(*pic_tasks, *tbl_tasks, return_exceptions=True)

    # Sum token usage and compute cost
    input_tokens  = sum(e.get("input_tokens",  0) for e in prompt_log)
    output_tokens = sum(e.get("output_tokens", 0) for e in prompt_log)
    cost_usd = (
        (input_tokens  / 1_000_000) * GEMINI_INPUT_PRICE_PER_M +
        (output_tokens / 1_000_000) * GEMINI_OUTPUT_PRICE_PER_M
    )
    token_stats = {
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "cost_usd":      round(cost_usd, 6),
    }
    log.info("Token usage — input: %d  output: %d  cost: $%.6f",
             input_tokens, output_tokens, cost_usd)

    n_enriched = len(pictures) + len(tables)
    return n_enriched, prompt_log, token_stats


def enrich(doc, filename: str = "document.pdf",
           project: str = VERTEX_PROJECT,
           location: str = VERTEX_LOCATION,
           image_model: str = IMAGE_MODEL,
           table_model: str = TABLE_MODEL,
           concurrency: int = ENRICHER_CONCURRENCY) -> tuple:
    """Synchronous wrapper — call from _run_conversion thread.

    Returns (n_enriched, prompt_log, token_stats) where token_stats has
    input_tokens, output_tokens, cost_usd.
    """
    return asyncio.run(_enrich_async(doc, filename, project, location,
                                     image_model, table_model, concurrency))
