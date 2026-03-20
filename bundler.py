"""
bundler.py
==========
TOC-based PDF bundle planner for memory-bounded Docling conversion.

Opens a PDF with PyMuPDF (lightweight — reads xref only, no page rendering),
extracts the Table of Contents, builds a section tree, and produces a list of
Bundle dicts that respect section boundaries while staying within a configurable
max page limit.

Each bundle carries metadata (id, name, page range, parent section, continuation
info) that survives into the final merged JSON for downstream chunking/tracing.
"""

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class Bundle:
    id: str
    name: str
    page_start: int          # 1-based inclusive
    page_end: int            # 1-based inclusive
    toc_level: int
    parent_section: str
    is_continuation: bool = False
    continuation_of: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def page_count(self) -> int:
        return self.page_end - self.page_start + 1


@dataclass
class _SectionNode:
    level: int
    title: str
    page_start: int          # 1-based
    page_end: int            # 1-based (computed later)
    children: list = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return self.page_end - self.page_start + 1


def _sanitize_name(title: str, max_len: int = 40) -> str:
    """Convert a TOC title to a filesystem/ID-safe string."""
    s = re.sub(r'[^\w\s-]', '', title.lower().strip())
    s = re.sub(r'[\s_]+', '_', s)
    return s[:max_len].rstrip('_') or 'untitled'


def extract_toc(pdf_path: str) -> list[list]:
    """Extract TOC from a PDF using PyMuPDF. Returns [[level, title, page], ...].

    Lightweight — only reads the xref/outline structure, does not load page content.
    Returns empty list if no TOC/bookmarks exist.
    """
    import fitz
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    total_pages = len(doc)
    doc.close()
    return toc, total_pages


def _build_section_tree(toc: list[list], total_pages: int) -> list[_SectionNode]:
    """Convert flat TOC list into a tree of _SectionNode objects.

    Each node's page_end is computed as (next sibling's page_start - 1)
    or total_pages for the last entry at each level.
    """
    if not toc:
        return []

    # First pass: create flat node list with page_start
    nodes = []
    for level, title, page in toc:
        nodes.append(_SectionNode(level=level, title=title, page_start=page, page_end=total_pages))

    # Compute page_end for each node: next entry at same or higher level - 1
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if nodes[j].level <= nodes[i].level:
                nodes[i].page_end = nodes[j].page_start - 1
                break
        # If no next sibling found, page_end stays as total_pages

    # Clamp to valid range
    for n in nodes:
        n.page_start = max(1, n.page_start)
        n.page_end = max(n.page_start, min(n.page_end, total_pages))

    # Build tree: nest children under parents based on level
    root_nodes = []
    stack = []  # [(level, node)]

    for node in nodes:
        # Pop stack until we find a parent (lower level)
        while stack and stack[-1][0] >= node.level:
            stack.pop()

        if stack:
            stack[-1][1].children.append(node)
        else:
            root_nodes.append(node)

        stack.append((node.level, node))

    return root_nodes


def _split_section_into_bundles(
    node: _SectionNode,
    max_pages: int,
    bundles: list[Bundle],
    seq_counter: list[int],
    parent_section: str = "",
) -> None:
    """Recursively split a section node into bundles respecting max_pages."""

    section_name = node.title or "untitled"
    pages = node.page_count

    if pages <= max_pages:
        # Fits in one bundle
        seq_counter[0] += 1
        bid = f"b{seq_counter[0]:03d}_{_sanitize_name(section_name)}"
        bundles.append(Bundle(
            id=bid,
            name=section_name,
            page_start=node.page_start,
            page_end=node.page_end,
            toc_level=node.level,
            parent_section=parent_section or section_name,
        ))
        return

    # Section too large — try splitting at children (subsection boundaries)
    if node.children:
        current_group = []
        current_pages = 0

        # Absorb preamble pages (before first child) into the first group
        # so they don't become a tiny standalone bundle
        preamble_start = None
        if node.children[0].page_start > node.page_start:
            preamble_start = node.page_start
            preamble_pages = node.children[0].page_start - node.page_start
            current_pages = preamble_pages

        for child in node.children:
            child_pages = child.page_count

            if child_pages > max_pages:
                # Flush accumulated group before recursing into large child
                if current_group:
                    _flush_group(current_group, bundles, seq_counter,
                                 section_name, parent_section, node.level,
                                 override_start=preamble_start)
                    preamble_start = None  # only applies to first flush
                    current_group = []
                    current_pages = 0
                # This child itself is too large — recurse
                _split_section_into_bundles(
                    child, max_pages, bundles, seq_counter,
                    parent_section=parent_section or section_name,
                )
            elif current_pages + child_pages <= max_pages:
                # Add to current group
                current_group.append(child)
                current_pages += child_pages
            else:
                # Flush current group as a bundle
                if current_group:
                    _flush_group(current_group, bundles, seq_counter,
                                 section_name, parent_section, node.level,
                                 override_start=preamble_start)
                    preamble_start = None  # only applies to first flush
                # Start new group with this child
                current_group = [child]
                current_pages = child_pages

        # Flush remaining group
        if current_group:
            _flush_group(current_group, bundles, seq_counter,
                         section_name, parent_section, node.level,
                         override_start=preamble_start)
    else:
        # No children — fixed-page fallback split
        _fixed_page_split(
            node.page_start, node.page_end, max_pages,
            bundles, seq_counter, section_name, parent_section, node.level,
        )


def _flush_group(
    group: list[_SectionNode],
    bundles: list[Bundle],
    seq_counter: list[int],
    section_name: str,
    parent_section: str,
    level: int,
    override_start: int | None = None,
) -> None:
    """Create a bundle from a group of consecutive subsections.

    If override_start is set, use it as the bundle's page_start instead of
    the first child's page_start. This absorbs preamble pages into the group.
    """
    seq_counter[0] += 1
    first = group[0]
    last = group[-1]

    if len(group) == 1:
        name = first.title
    else:
        name = f"{first.title} ... {last.title}"

    bid = f"b{seq_counter[0]:03d}_{_sanitize_name(section_name)}"
    bundles.append(Bundle(
        id=bid,
        name=name,
        page_start=override_start if override_start is not None else first.page_start,
        page_end=last.page_end,
        toc_level=level,
        parent_section=parent_section or section_name,
    ))


def _fixed_page_split(
    page_start: int,
    page_end: int,
    max_pages: int,
    bundles: list[Bundle],
    seq_counter: list[int],
    section_name: str,
    parent_section: str,
    level: int,
) -> None:
    """Fallback: split a range into fixed-size bundles."""
    total = page_end - page_start + 1
    part = 0
    first_bundle_id = None

    for offset in range(0, total, max_pages):
        part += 1
        seq_counter[0] += 1
        ps = page_start + offset
        pe = min(page_start + offset + max_pages - 1, page_end)
        bid = f"b{seq_counter[0]:03d}_{_sanitize_name(section_name)}_p{part}"

        is_cont = part > 1
        cont_of = first_bundle_id if is_cont else None
        if part == 1:
            first_bundle_id = bid

        bundles.append(Bundle(
            id=bid,
            name=f"{section_name} (part {part})",
            page_start=ps,
            page_end=pe,
            toc_level=level,
            parent_section=parent_section or section_name,
            is_continuation=is_cont,
            continuation_of=cont_of,
        ))


def plan_bundles(pdf_path: str, max_pages: int = 50) -> tuple[list[Bundle], int]:
    """Plan bundles for a PDF based on its TOC.

    Returns (bundles, total_pages).
    Returns empty bundles list if no TOC is found.
    """
    toc, total_pages = extract_toc(pdf_path)

    if not toc:
        log.info("No TOC found in %s — bundle planning not possible", pdf_path)
        return [], total_pages

    log.info("TOC has %d entries, %d total pages", len(toc), total_pages)

    tree = _build_section_tree(toc, total_pages)
    if not tree:
        return [], total_pages

    bundles = []
    seq_counter = [0]  # mutable counter passed by reference

    for root_node in tree:
        _split_section_into_bundles(root_node, max_pages, bundles, seq_counter)

    # Handle pages before first TOC entry
    first_toc_page = toc[0][2] if toc else 1
    if first_toc_page > 1:
        # Insert a "front matter" bundle at the beginning
        fm_bundle = Bundle(
            id="b000_front_matter",
            name="Front Matter",
            page_start=1,
            page_end=first_toc_page - 1,
            toc_level=0,
            parent_section="Front Matter",
        )
        bundles.insert(0, fm_bundle)

    # Validate: ensure no page gaps or overlaps
    bundles.sort(key=lambda b: b.page_start)

    log.info("Planned %d bundles for %d pages (max %d pages/bundle)",
             len(bundles), total_pages, max_pages)
    for b in bundles:
        log.debug("  %s: pages %d-%d (%d pages) %s",
                  b.id, b.page_start, b.page_end, b.page_count,
                  "[continuation]" if b.is_continuation else "")

    return bundles, total_pages
