"""synth_eval.pii — PII column detection and fake-value substitution.

The reduce-fit pipeline refills non-modelled columns by bootstrapping from the
real column — which puts *real* names / emails / phones into the synthetic
output (shuffled across rows, but the strings are real).  This module detects
those columns and, by default, replaces them with Faker-generated values that
never existed in the source, preserving each column's missing rate.

Policies (chosen per column in the UI):
  * ``fake``    — plausible generated values, zero real strings (default)
  * ``shuffle`` — the old behaviour: bootstrap real values across rows
  * ``drop``    — remove the column from the synthetic output entirely
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

#: column-name tokens per kind; checked in this order, first hit wins.
#: NB deliberately no bare FIRST/LAST/MIDDLE tokens — LAST_VERIFIED_TRANSIT and
#: friends are not names; real name columns virtually always carry NAME or NM.
_TOKENS = [
    ("email",  ("EMAIL", "E_MAIL")),
    ("phone",  ("PHONE", "TELEPHONE", "_TEL", "TEL_", "FAX", "MOBILE", "CELL")),
    ("postal", ("POSTAL", "ZIP")),
    ("street", ("ADDR_LINE", "ADDRESS_LINE", "STREET", "_ADDR", "ADDR_")),
    ("name",   ("NAME", "_NM", "NM_", "SURNAME", "GIVEN", "_USER", "USER_")),
]
#: kinds that only make sense for string columns (a numeric column can hold a
#: phone or a zip, but never a name / email / street address)
_STR_ONLY = {"name", "email", "street"}
#: suffixes that mark a column as a code/flag/date — never PII, whatever the
#: rest of the name says (PREFIX_NAME_TP_CD is a type code, not a name).
_NOT_PII = re.compile(r"(_CD|_CODE|_ID|_IND|_DT|_DATE|_TP|_TYPE|_CT|_QTY|_AMT|_PCT)$", re.I)

_RE_EMAIL  = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
_RE_PHONE  = re.compile(r"^[\d\s\-\+\(\)\.x]{7,}$")
_RE_POSTAL = re.compile(r"^([A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d|\d{5}(-\d{4})?)$")
_RE_STREET = re.compile(r"^\d+\s+\S+.*\b(ST|STREET|AVE|AVENUE|RD|ROAD|DR|DRIVE|BLVD|"
                        r"CRES|CRESCENT|CT|COURT|WAY|LANE|PL|PLACE|TRAIL|CIR)\b\.?$", re.I)

#: low bar -- only needs to rule out columns that are clearly NOT person names
#: (job titles, product names, ...), not confirm every real name column.
_NAME_CONFIRM_THRESHOLD = 0.15


def _confirms_person_name(sample: pd.Series) -> bool:
    """Check sampled values against Faker's own name corpus (content, not
    column name) -- catches columns like OCCUPATION_NAME ("Registered Nurse")
    that trip the ``NAME`` token but hold something else entirely. Generalizes
    across schemas since it never depends on this dataset's naming
    conventions, only the values themselves.
    """
    from faker.providers.person.en_US import Provider as _PersonProvider

    vals = sample.dropna().astype(str).str.strip()
    if len(vals) < 5:
        return True  # too little data to rule anything out -- trust the token
    first_names = {n.upper() for n in _PersonProvider.first_names}
    last_names = {n.upper() for n in _PersonProvider.last_names}

    def is_name(v: str) -> bool:
        toks = [t.upper() for t in v.split() if t]
        if not toks:
            return False
        if len(toks) == 1:
            return toks[0] in first_names or toks[0] in last_names
        return toks[0] in first_names or toks[-1] in last_names

    return float(vals.map(is_name).mean()) >= _NAME_CONFIRM_THRESHOLD


def _value_kind(sample: pd.Series) -> Optional[str]:
    """Classify a column by the *shape* of its values (>=60% of a sample must match)."""
    vals = sample.dropna().astype(str).str.strip()
    if len(vals) < 5:
        return None
    def frac(rx):
        return float(vals.str.match(rx).mean())
    if frac(_RE_EMAIL) >= 0.6:
        return "email"
    if frac(_RE_POSTAL) >= 0.6:
        return "postal"
    if frac(_RE_STREET) >= 0.6:
        return "street"
    # phone last: the pattern is loose, so require digits to dominate
    if frac(_RE_PHONE) >= 0.6 and float(vals.str.count(r"\d").median() or 0) >= 7:
        return "phone"
    return None


def detect_pii(df: pd.DataFrame, modelable: Optional[List[str]] = None,
               sample_n: int = 60) -> Dict[str, str]:
    """Return ``{column: kind}`` for likely-PII columns.

    Detection = column-name tokens first, then value-shape regexes on a sample
    (value checks only for object columns *outside* the modelable set, so a
    low-cardinality code column can never be misread as a postal code).
    """
    modelable = set(modelable or [])
    out: Dict[str, str] = {}
    for c in df.columns:
        cu = str(c).upper()
        if _NOT_PII.search(cu):
            continue
        kind = next((k for k, toks in _TOKENS if any(t in cu for t in toks)), None)
        if kind in _STR_ONLY and df[c].dtype != object:
            kind = None                      # numeric column can't be a name/email/street
        if kind == "name" and df[c].dtype == object:
            s = df[c].head(2000)
            n_notna = s.notna().sum()
            sample = s.sample(min(sample_n, n_notna), random_state=0) if n_notna else s
            if not _confirms_person_name(sample):
                kind = None                  # column-name token, but values aren't person names
        if kind is None and c not in modelable and df[c].dtype == object:
            try:
                kind = _value_kind(df[c].head(2000).sample(
                    min(sample_n, df[c].head(2000).notna().sum() or 1), random_state=0)
                    if df[c].notna().any() else df[c])
            except Exception:
                kind = None
        if kind:
            out[c] = kind
    return out


def fake_series(kind: str, n: int, like: Optional[pd.Series] = None,
                seed: int = 0, column_name: str = "") -> pd.Series:
    """``n`` Faker values of ``kind``, preserving ``like``'s missing rate.

    Deterministic for a given (kind, n, seed, column_name).  ``column_name``
    refines names: FIRST/GIVEN -> first names, LAST/SURNAME -> last names.
    """
    from faker import Faker

    fk = Faker("en_CA")
    fk.seed_instance(seed + (hash(column_name) % 100003))
    cu = column_name.upper()
    if kind == "name":
        if "FIRST" in cu or "GIVEN" in cu or "MIDDLE" in cu:
            gen = fk.first_name
        elif "LAST" in cu or "SURNAME" in cu:
            gen = fk.last_name
        else:
            gen = fk.name
    elif kind == "email":
        gen = fk.email
    elif kind == "phone":
        gen = fk.phone_number
    elif kind == "postal":
        gen = fk.postcode
    elif kind == "street":
        gen = fk.street_address
    else:  # unknown kind: opaque but harmless
        gen = lambda: fk.bothify("????####")  # noqa: E731
    vals = np.array([gen() for _ in range(n)], dtype=object)
    miss = float(like.isna().mean()) if like is not None and len(like) else 0.0
    if miss > 0 and n:
        rng = np.random.default_rng(seed)
        vals[rng.random(n) < miss] = np.nan
    return pd.Series(vals, dtype=object)


def apply_pii_plan(df: pd.DataFrame, plan: Dict[str, tuple], real: pd.DataFrame,
                   seed: int = 0) -> pd.DataFrame:
    """Apply ``{col: (action, kind)}`` to one synthetic table.

    ``fake`` replaces the column's values; ``drop`` removes the column;
    anything else (``shuffle``) leaves the refilled bootstrap untouched.
    """
    for c, (action, kind) in (plan or {}).items():
        if c not in df.columns:
            continue
        if action == "drop":
            df = df.drop(columns=[c])
        elif action == "fake":
            df[c] = fake_series(kind, len(df), real[c] if c in real.columns else None,
                                seed, c).to_numpy()
    return df
