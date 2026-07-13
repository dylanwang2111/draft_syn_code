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


def entity_key_tables(tables: Dict[str, pd.DataFrame], entity_key: str) -> List[str]:
    """Names of tables that contain ``entity_key`` (and could become children)."""
    return [t for t, df in tables.items() if entity_key in df.columns]


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
    entity_key: str,
    parent_name: Optional[str] = None,
    child_primary_keys: Optional[Dict[str, str]] = None,
    lift_invariant: bool = True,
    child_tables: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, pd.DataFrame], object, List[dict], dict]:
    """Return ``(new_tables, metadata, relationships, info)``.

    * ``new_tables`` = the original tables plus a derived parent hub keyed by
      ``entity_key`` (one row per distinct value across all child tables); any
      per-entity invariant columns are moved onto the parent when
      ``lift_invariant`` is set.
    * ``metadata`` = an SDV ``Metadata`` with the parent's primary key, each
      child's foreign key, and the parent→child relationships.
    * ``relationships`` = the same relationships as plain dicts (for the report).
    * ``info`` = {parent, children, lifted_columns} for messaging.

    ``child_tables`` optionally restricts which tables become children of the
    hub (the user's "select a customized number of tables").  Any table it names
    that lacks ``entity_key`` is ignored; when omitted, every table containing
    the key is used.  Tables that contain the key but aren't chosen are left as
    unlinked standalone tables.

    Raises ValueError if ``entity_key`` is in fewer than one table.
    """
    from sdv.metadata import Metadata

    have_key = entity_key_tables(tables, entity_key)
    children = [t for t in child_tables if t in have_key] if child_tables is not None else have_key
    if not children:
        raise ValueError(f"entity key '{entity_key}' is not present in any selected table")

    parent_name = parent_name or f"{entity_key}_HUB"
    while parent_name in tables:                       # avoid a name collision
        parent_name += "_"

    # distinct entity ids across every child table
    ids = set()
    for t in children:
        ids |= set(tables[t][entity_key].dropna().unique())
    parent_df = pd.DataFrame({entity_key: sorted(ids, key=lambda v: (str(type(v)), str(v)))})

    lifted = _invariant_columns(tables, entity_key, children) if lift_invariant else {}
    # attach one invariant value per entity to the parent, and drop those
    # columns from the children so they live in exactly one place.
    new_tables = {t: df.copy() for t, df in tables.items()}
    for col, src in lifted.items():
        vals = (new_tables[src][[entity_key, col]].dropna(subset=[entity_key])
                .drop_duplicates(subset=[entity_key]).set_index(entity_key)[col])
        parent_df[col] = parent_df[entity_key].map(vals)
        for t in children:
            if col in new_tables[t].columns:
                new_tables[t] = new_tables[t].drop(columns=[col])

    new_tables = {parent_name: parent_df, **new_tables}

    # metadata: detect, then force the key roles + relationships
    meta = Metadata.detect_from_dataframes(new_tables)
    md = meta.to_dict()
    md["tables"][parent_name]["columns"][entity_key] = {"sdtype": "id"}
    md["tables"][parent_name]["primary_key"] = entity_key

    child_primary_keys = child_primary_keys or {}
    rels = []
    for t in children:
        md["tables"][t]["columns"][entity_key] = {"sdtype": "id"}
        pk = child_primary_keys.get(t)
        if pk and pk in new_tables[t].columns and new_tables[t][pk].is_unique:
            md["tables"][t]["columns"][pk] = {"sdtype": "id"}
            md["tables"][t]["primary_key"] = pk
        rels.append({
            "parent_table_name": parent_name, "parent_primary_key": entity_key,
            "child_table_name": t, "child_foreign_key": entity_key,
        })
    md["relationships"] = rels

    metadata = Metadata.load_from_dict(md)
    info = {"parent": parent_name, "children": children,
            "lifted_columns": lifted, "n_entities": len(parent_df)}
    return new_tables, metadata, rels, info
