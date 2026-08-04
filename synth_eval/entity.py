"""synth_eval.entity — build an entity-hub (star) schema from a shared key.

Slowly-changing-dimension tables often share a durable *business key* (e.g.
``CONT_ID``) that identifies one real entity across all of its dated version
rows — but that key is **not unique in any single table**, so it can't be used
directly as an HMA parent primary key.  This module derives a synthetic parent
"hub" table (one row per distinct entity key) and rewires the original tables
as its children (the shared key becomes a foreign key).  HMA then generates
entities first and cascades child rows per entity, so cross-table referential
integrity holds by construction.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


# Ordered list of known prefix conventions that mark an otherwise-canonical
# column as a variant (e.g. this schema's "X_" extension-field convention,
# so a source system's OCCUPATION_TP_CD showing up as X_OCCUPATION_TP_CD in
# another table is still recognized as the same key). Add a prefix here --
# nothing else needs to change -- when another naming convention shows up;
# each is applied deterministically, never guessed or fuzzy-matched.
_KEY_PREFIX_RULES: List[str] = ["X_"]


def _normalize_key_name(col: str) -> str:
    """Deterministic normalization for cross-table key matching: case-fold
    to upper, then strip the first matching prefix from
    :data:`_KEY_PREFIX_RULES`, if any. Two columns normalize to the same
    string iff they're the same key modulo case and a known prefix
    convention -- no fuzzy matching.
    """
    c = col.upper()
    for prefix in _KEY_PREFIX_RULES:
        p = prefix.upper()
        if c.startswith(p) and len(c) > len(p):
            return c[len(p):]
    return c


def _resolve_key_column(df: pd.DataFrame, entity_key: str) -> Optional[str]:
    """The actual column name in ``df`` matching ``entity_key``, either
    literally or via :func:`_normalize_key_name`. ``None`` if there's no
    match at all."""
    if entity_key in df.columns:
        return entity_key
    target = _normalize_key_name(entity_key)
    for c in df.columns:
        if _normalize_key_name(c) == target:
            return c
    return None


def entity_key_tables(tables: Dict[str, pd.DataFrame], entity_key: str) -> List[str]:
    """Names of tables that contain ``entity_key`` (and could become
    children), literally or via a deterministic naming-convention match
    (see :func:`_normalize_key_name`)."""
    return [t for t, df in tables.items() if _resolve_key_column(df, entity_key) is not None]


def _invariant_columns(tables, entity_key, child_tables, max_cols=8):
    """Columns that are constant within every entity group and can be lifted to
    the parent (e.g. BIRTH_DT).  Checked per table; a column is lifted from the
    first child table where it is invariant.  Returns {column: source_table}."""
    lifted: Dict[str, str] = {}
    for t in child_tables:
        df = tables[t]
        g = df.groupby(entity_key, dropna=True)
        for c in df.columns:
            if c == entity_key or c in lifted:
                continue
            # constant within every group and actually has some signal
            try:
                if g[c].nunique(dropna=True).max() <= 1 and df[c].notna().any():
                    lifted[c] = t
            except TypeError:  # unhashable / weird dtype
                continue
            if len(lifted) >= max_cols:
                return lifted
    return lifted


def build_entity_hub(
    tables: Dict[str, pd.DataFrame],
    entity_keys: Sequence[str],
    parent_names: Optional[Dict[str, str]] = None,
    child_primary_keys: Optional[Dict[str, str]] = None,
    lift_invariant: bool = True,
    child_tables: Optional[Dict[str, Sequence[str]]] = None,
) -> Tuple[Dict[str, pd.DataFrame], object, List[dict], Dict[str, dict]]:
    """Return ``(new_tables, metadata, relationships, info)`` for one or more
    simultaneous entity-key hubs.

    Each key in ``entity_keys`` gets its own derived parent hub table (one row
    per distinct value of that key, across whichever tables carry it). A
    table can be a child of more than one hub at once -- e.g. PERSON could
    carry both a CONT_ID-style key and an OCCUPATION_TP_CD-style key as two
    independent foreign keys -- that's just two different columns on the same
    table, no conflict between them.

    All hubs are built together into ONE ``Metadata``/relationships set in a
    single pass, rather than one ``build_entity_hub`` call per key chained
    sequentially: ``Metadata.detect_from_dataframes`` re-scans every table
    fresh on each call, so a second sequential call would silently lose the
    first hub's forced id/primary-key/relationship overrides.

    * ``new_tables`` = the original tables plus one derived parent hub per
      key; any per-entity invariant columns are moved onto their hub's parent
      when ``lift_invariant`` is set.
    * ``metadata`` = an SDV ``Metadata`` with every parent's primary key, every
      child's foreign key, and every parent→child relationship, across all
      hubs at once.
    * ``relationships`` = the same relationships as plain dicts (for the report).
    * ``info`` = ``{entity_key: {parent, children, lifted_columns, n_entities}}``,
      one entry per key, for messaging.

    ``child_tables``, if given, is ``{entity_key: [table, ...]}`` restricting
    which tables become children of THAT hub; a key omitted from the dict (or
    the whole argument omitted) uses every table that carries it. Tables that
    contain a key but aren't chosen for it are left unlinked to that hub (they
    may still be linked to another).  ``parent_names``, if given, is
    ``{entity_key: name}``; keys omitted default to ``f"{entity_key}_HUB"``
    (deduped against name collisions).

    Raises ValueError if any entity_key is in fewer than one selected table.
    """
    from sdv.metadata import Metadata

    parent_names = dict(parent_names or {})
    child_tables = dict(child_tables or {})
    child_primary_keys = child_primary_keys or {}

    # Normalize each child's own local variant of each key (e.g.
    # X_OCCUPATION_TP_CD) to that key's canonical name, on a shared WORKING
    # COPY only -- everything below assumes one literal column name per key
    # across every one of its children, and this is the one place that
    # assumption gets made true, without mutating the caller's original
    # tables (defensive: the caller, typically dashboard_core._run_job,
    # normalizes its own working copies earlier too, so this is usually a
    # no-op by the time it runs).
    tables = dict(tables)
    per_key_children: Dict[str, List[str]] = {}
    for key in entity_keys:
        have_key = entity_key_tables(tables, key)
        restrict = child_tables.get(key)
        children = [t for t in restrict if t in have_key] if restrict is not None else have_key
        if not children:
            raise ValueError(f"entity key '{key}' is not present in any selected table")
        per_key_children[key] = children
        for t in children:
            local_col = _resolve_key_column(tables[t], key)
            if local_col and local_col != key:
                tables[t] = tables[t].rename(columns={local_col: key})

    used_names = set(tables)
    resolved_parent_names: Dict[str, str] = {}
    for key in entity_keys:
        name = parent_names.get(key) or f"{key}_HUB"
        while name in used_names:           # avoid a name collision
            name += "_"
        used_names.add(name)
        resolved_parent_names[key] = name

    new_tables = {t: df.copy() for t, df in tables.items()}
    parent_dfs: Dict[str, pd.DataFrame] = {}
    lifted_by_key: Dict[str, dict] = {}
    for key in entity_keys:
        children = per_key_children[key]
        ids = set()
        for t in children:
            ids |= set(tables[t][key].dropna().unique())
        parent_df = pd.DataFrame({key: sorted(ids, key=lambda v: (str(type(v)), str(v)))})

        lifted = _invariant_columns(tables, key, children) if lift_invariant else {}
        # attach one invariant value per entity to this hub's parent, and drop
        # those columns from its children so they live in exactly one place.
        for col, src in lifted.items():
            vals = (new_tables[src][[key, col]].dropna(subset=[key])
                    .drop_duplicates(subset=[key]).set_index(key)[col])
            parent_df[col] = parent_df[key].map(vals)
            for t in children:
                if col in new_tables[t].columns:
                    new_tables[t] = new_tables[t].drop(columns=[col])
        lifted_by_key[key] = lifted
        parent_dfs[key] = parent_df

    new_tables = {**{resolved_parent_names[k]: df for k, df in parent_dfs.items()}, **new_tables}

    # metadata: detect once over every table (parents + children, every hub),
    # then force the key roles + relationships for each hub
    meta = Metadata.detect_from_dataframes(new_tables)
    md = meta.to_dict()
    rels: List[dict] = []
    for key in entity_keys:
        parent_name = resolved_parent_names[key]
        children = per_key_children[key]
        md["tables"][parent_name]["columns"][key] = {"sdtype": "id"}
        md["tables"][parent_name]["primary_key"] = key
        for t in children:
            md["tables"][t]["columns"][key] = {"sdtype": "id"}
            pk = child_primary_keys.get(t)
            if pk and pk in new_tables[t].columns and new_tables[t][pk].is_unique:
                md["tables"][t]["columns"][pk] = {"sdtype": "id"}
                md["tables"][t]["primary_key"] = pk
            rels.append({
                "parent_table_name": parent_name, "parent_primary_key": key,
                "child_table_name": t, "child_foreign_key": key,
            })
    md["relationships"] = rels

    metadata = Metadata.load_from_dict(md)
    info = {
        key: {"parent": resolved_parent_names[key], "children": per_key_children[key],
              "lifted_columns": lifted_by_key[key], "n_entities": len(parent_dfs[key])}
        for key in entity_keys
    }
    return new_tables, metadata, rels, info
