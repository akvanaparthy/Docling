"""
reorder.py
==========
Reading-order correction for DoclingDocument objects.

Two main functions:

1. _merge_orphaned_list_descriptions(doc)
   Finds body-level text elements that spatially belong to list items
   (same Y coordinate, to the right) and merges them into the list item.
   Fixes a Docling parsing quirk where description text next to a numbered
   list item gets parsed as a separate body element.

2. _reorder_body_children(doc)
   Sorts body.children in spatial reading order: page ASC, top-to-bottom, left-to-right.
   Uses BOTTOMLEFT coordinate origin (higher t = higher on page = comes first).
   Groups are treated as atomic units positioned by their first leaf element's bbox.
   Calls _merge_orphaned_list_descriptions first.

3. _reindex_json_reading_order(data)
   After body.children are sorted, the arrays (texts[], tables[], etc.) still have
   their original indices. This function renumbers them to match reading order and
   rewrites all self_ref/parent/children references throughout the JSON dict.

Usage:
    from reorder import reorder_document, reindex_json

    # On a DoclingDocument object (in-place):
    reorder_document(doc)

    # On a JSON dict (returns new dict):
    d = doc.model_dump(mode='json')
    d = reindex_json(d)
"""


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


# ── Public API ────────────────────────────────────────────────────────────────

def reorder_document(doc) -> None:
    """Reorder a DoclingDocument in-place: merge orphaned list descriptions,
    then sort body.children by spatial reading order."""
    _reorder_body_children(doc)


def reindex_json(data: dict) -> dict:
    """Reindex a DoclingDocument JSON dict so array indices match reading order.
    Call AFTER reorder_document(). Returns the modified dict."""
    return _reindex_json_reading_order(data)
