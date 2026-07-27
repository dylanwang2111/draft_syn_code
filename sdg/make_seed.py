#!/usr/bin/env python
"""Regenerate sdg/seed/*.csv — a realistic MDM-style sample dataset.

Keeps the exact column headers of the original extract but fills them with
coherent values: real ISO timestamps with chained SCD versions
(IDP_EFFECTIVE_DATE < IDP_END_DATE = next version's start), Faker (en_CA)
names consistent across CONTACT and PERSONNAME, small skewed type-code
vocabularies, and cross-field logic (age <-> marital status <-> children,
inactive clients get INACTIVATED_DT/LEFT_DT, deceased flag <-> date, prefix
<-> gender, surname change history for some married customers).

Deterministic: python sdg/make_seed.py  always produces the same files.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

SEED = 42
N_CUST = 1000
NOW = datetime(2026, 6, 30)
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed")

rng = np.random.default_rng(SEED)
fk = Faker("en_CA")
fk.seed_instance(SEED)

USERS = ["cusadmin", "batchsys", "MECH", "migr8", "svc_mdm"]
USER_P = [0.38, 0.27, 0.18, 0.09, 0.08]
PROVINCES = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010]  # ON..NL
PROV_P = [0.36, 0.22, 0.13, 0.10, 0.07, 0.04, 0.03, 0.02, 0.02, 0.01]


def pick(vals, p=None):
    return vals[int(rng.choice(len(vals), p=p))]


def maybe(p, val):
    return val if rng.random() < p else ""


def dt_between(a: datetime, b: datetime) -> datetime:
    if b <= a:
        return a
    return a + timedelta(seconds=float(rng.random()) * (b - a).total_seconds())


def ts(d):
    return d.strftime("%Y-%m-%d %H:%M:%S") if isinstance(d, datetime) else ""


def day(d):
    return d.strftime("%Y-%m-%d") if isinstance(d, datetime) else ""


def transit():
    return int(rng.integers(1000, 9999))


def version_chain(start: datetime, n: int):
    """n SCD versions: [(effective, end|None)]; each end = next effective."""
    effs = sorted(dt_between(start, NOW - timedelta(days=30)) for _ in range(n))
    effs[0] = start
    out = []
    for i, e in enumerate(effs):
        out.append((e, effs[i + 1] if i + 1 < len(effs) else None))
    return out


# ---------------------------------------------------------------- customers
customers = []
for _ in range(N_CUST):
    gender = pick([1, 2], [0.49, 0.51])
    age = int(np.clip(rng.normal(47, 16), 18, 92))
    birth = datetime(NOW.year - age, int(rng.integers(1, 13)), int(rng.integers(1, 29)))
    since = dt_between(max(birth + timedelta(days=16 * 365), datetime(1996, 1, 1)),
                       datetime(2025, 6, 1))
    # marital status: young -> single, mid -> married, old adds widowed/divorced
    if age < 28:
        marital = pick([131865, 131866, 131867], [0.80, 0.17, 0.03])
    elif age < 60:
        marital = pick([131865, 131866, 131867, 131868], [0.22, 0.58, 0.17, 0.03])
    else:
        marital = pick([131865, 131866, 131867, 131868], [0.10, 0.52, 0.20, 0.18])
    children = 0
    if age >= 25:
        lam = 1.7 if marital in (131866, 131868) else 0.5
        children = int(min(rng.poisson(lam), 5))
    if age >= 65:
        empl = pick([503, 501, 502], [0.78, 0.14, 0.08])         # mostly retired
    elif age <= 24:
        empl = pick([504, 501, 502], [0.55, 0.38, 0.07])         # mostly student
    else:
        empl = pick([501, 502, 503, 504], [0.72, 0.18, 0.06, 0.04])
    prov = pick(PROVINCES, PROV_P)
    lang = pick([703793, 703794], [0.32, 0.68]) if prov == 2002 \
        else pick([703793, 703794], [0.93, 0.07])                # QC skews French
    first = fk.first_name_male() if gender == 1 else fk.first_name_female()
    customers.append({
        "cont_id": int(rng.integers(10**9, 10**10)),
        "gender": gender, "age": age, "birth": birth, "since": since,
        "marital": marital, "children": children, "empl": empl,
        "prov": prov, "lang": lang,
        "first": first, "middle": maybe(0.45, fk.first_name()),
        "last": fk.last_name(),
        "occup": int(rng.integers(348820, 348841)),
        "edu": pick(["", 150122, 150123, 150124, 150125], [0.15, 0.20, 0.35, 0.22, 0.08]),
        "deceased": (age > 68 and rng.random() < 0.20),
        "inactive": rng.random() < 0.12,
        "transit": transit(),
    })


def audit_cols(eff, end):
    return {
        "IDP_WAREHOUSE_ID": int(rng.integers(10**12, 10**13)),
        "IDP_AUDIT_ID": int(rng.integers(10**6, 10**7)),
        "IDP_EFFECTIVE_DATE": ts(eff),
        "IDP_END_DATE": ts(end) if end else "",
        "IDP_DELETE_DATE": "",
    }


def update_cols(eff):
    upd = dt_between(eff, NOW)
    return upd, {
        "LAST_UPDATE_DT": ts(upd),
        "LAST_UPDATE_USER": pick(USERS, USER_P),
        "LAST_UPDATE_TX_ID": int(rng.integers(10**17, 10**18)),
    }


# ------------------------------------------------------------------ CONTACT
contact_rows = []
for c in customers:
    if rng.random() > 0.96:            # a few customers exist only in child tables
        continue
    n_ver = pick([1, 2, 3, 4], [0.45, 0.32, 0.16, 0.07])
    for vi, (eff, end) in enumerate(version_chain(c["since"], n_ver)):
        current = end is None
        inactive = c["inactive"] and current
        upd, ucols = update_cols(eff)
        left = dt_between(eff, NOW) if inactive else None
        active_solicit = "Y" if (not inactive and rng.random() < 0.62) else "N"
        row = {
            **audit_cols(eff, end),
            "CONT_ID": c["cont_id"],
            "ACCE_COMP_TP_CD": pick([100301, 100302, 100303], [0.55, 0.35, 0.10]),
            "PREF_LANG_TP_CD": c["lang"],
            "CREATED_DT": ts(c["since"]),
            "INACTIVATED_DT": ts(left) if inactive else "",
            "CONTACT_NAME": f"{c['first']} {c['last']}".upper(),
            "PERSON_ORG_CODE": "P",
            "SOLICIT_IND": active_solicit,
            "CONFIDENTIAL_IND": maybe(0.06, "Y"),
            "CLIENT_IMP_TP_CD": maybe(0.08, pick([100601, 100602])),
            "CLIENT_ST_TP_CD": 100002 if inactive else 100001,
            "CLIENT_POTEN_TP_CD": pick([100201, 100202, 100203], [0.25, 0.55, 0.20]),
            "RPTING_FREQ_TP_CD": maybe(0.4, pick([100401, 100402])),
            "LAST_STATEMENT_DT": ts(dt_between(eff, NOW)) if not inactive else "",
            "PROVIDED_BY_CONT": "",
            "ALERT_IND": "Y" if rng.random() < 0.04 else "N",
            **ucols,
            "DO_NOT_DELETE_IND": maybe(0.05, "Y"),
            "LAST_USED_DT": ts(dt_between(eff, NOW)),
            "LAST_VERIFIED_DT": maybe(0.7, ts(dt_between(eff, NOW))),
            "SOURCE_IDENT_TP_CD": pick([3794, 3795, 3796], [0.5, 0.35, 0.15]),
            "SINCE_DT": ts(c["since"]),
            "LEFT_DT": ts(left) if inactive else "",
            "ACCESS_TOKEN_VALUE": maybe(0.6, int(rng.integers(100, 999))),
            "PENDING_CDC_IND": "Y" if rng.random() < 0.18 else "N",
            "X_OFFICIAL_LANGUAGE_TP_CD": 100000 if c["lang"] == 703793 else 100001,
            "X_LARGE_CASH_TXN_RPT_IND": "Y" if rng.random() < 0.03 else "N",
            "X_CTRY_RES_TP_CD": pick([1002, 1010, 1044], [0.93, 0.05, 0.02]),
            "X_PROV_RES_TP_CD": c["prov"],
            "X_CREATED_DT": ts(c["since"]),
            "X_SRC_SYS_LAST_UPD_USER": pick(USERS, USER_P),
            "X_CREATEDBY_USER": pick(USERS, USER_P),
            "X_SRC_SYS_LAST_UPD_DT": ts(upd),
            "X_PRIM_REL": "Y" if vi == 0 and rng.random() < 0.8 else "N",
            "X_BSN_GRP_TP_CD": maybe(0.25, pick([301, 302, 303])),
            "X_OFAC_SCREEN_IND": "Y" if rng.random() < 0.85 else "N",
            "X_OFAC_SCREEN_DT": "",
            "X_BEN_OWNERSHIP_IND": maybe(0.3, pick(["Y", "N"])),
            "X_LCTR_EXEMPT_UPDATE_DT": maybe(0.04, ts(dt_between(eff, NOW))),
            "X_LAST_VERIFIED_TRANSIT": c["transit"],
            "X_CRSP_LANG_TP_CD": maybe(0.3, c["lang"]),
            "X_ICPM_AC_IND": "Y" if rng.random() < 0.07 else "N",
        }
        if row["X_OFAC_SCREEN_IND"] == "Y":
            row["X_OFAC_SCREEN_DT"] = ts(dt_between(eff, NOW))
        contact_rows.append(row)

# ------------------------------------------------------------------- PERSON
person_rows = []
for c in customers:
    if rng.random() > 0.72:            # partial parent coverage, on purpose
        continue
    n_ver = pick([1, 2], [0.7, 0.3])
    for vi, (eff, end) in enumerate(version_chain(c["since"], n_ver)):
        current = end is None
        # older versions of a married/divorced customer may show single
        marital = 131865 if (not current and c["marital"] in (131866, 131867)
                             and rng.random() < 0.6) else c["marital"]
        deceased = c["deceased"] and current
        dec_dt = dt_between(max(eff, NOW - timedelta(days=1500)), NOW) if deceased else None
        upd, ucols = update_cols(eff)
        person_rows.append({
            **audit_cols(eff, end),
            "CONT_ID": c["cont_id"],
            "MARITAL_ST_TP_CD": marital,
            "BIRTHPLACE_TP_CD": maybe(0.35, int(rng.integers(11001, 11011))),
            "CITIZENSHIP_TP_CD": pick([150123, 150124, ""], [0.85, 0.06, 0.09]),
            "HIGHEST_EDU_TP_CD": c["edu"],
            "AGE_VER_DOC_TP_CD": maybe(0.25, pick([160001, 160002, 160003])),
            "GENDER_TP_CODE": c["gender"],
            "BIRTH_DT": day(c["birth"]),
            "DECEASED_DT": day(dec_dt) if deceased else "",
            "CHILDREN_CT": c["children"],
            "DISAB_START_DT": "",
            "DISAB_END_DT": "",
            "USER_IND": "N",
            **ucols,
            "X_DECEASED_IND": "Y" if deceased else "N",
            "X_OCCUPATION_TP_CD": c["occup"],
            "X_COMPANY_TP_CD": maybe(0.2, int(rng.integers(600, 700))),
            "X_EMPL_TP_CD": c["empl"],
            "X_SEC_CITIZENSHIP_TP_CD": maybe(0.05, 150124),
            "X_CITIZENSHIP_LAST_UPD_DT": maybe(0.3, ts(dt_between(eff, NOW))),
            "X_IMPT_FMLY_NAME": "",
            "X_US_ALIEN_RES_IND": "Y" if rng.random() < 0.03 else "N",
            "X_CHILDREN_CT_CAPTURED_DT": ts(dt_between(eff, NOW)) if c["children"] else "",
            "X_RESIDENCY_TP_CD": pick([701, 702], [0.9, 0.1]),
            "X_SRC_SYS_LAST_UPD_DT": ts(upd),
            "X_SRC_SYS_LAST_UPD_USER": pick(USERS, USER_P),
            "X_SRC_SYS_LAST_UPD_TRANSIT": transit(),
            "X_SRC_SYS_LAST_UPD_USER_NAME": maybe(0.5, fk.name()),
            "X_INCOME_STABILITY_TP_CD": maybe(0.4, pick([170001, 170002, 170003])),
        })
    if rng.random() < 0.02:
        disab = dt_between(c["since"], NOW)
        person_rows[-1]["DISAB_START_DT"] = day(disab)
        if rng.random() < 0.5:
            person_rows[-1]["DISAB_END_DT"] = day(dt_between(disab, NOW))

# --------------------------------------------------------------- PERSONNAME
PREFIX = {1: [("MR", 108001)], 2: [("MS", 108003), ("MRS", 108002)]}


def phon(s):
    return s.upper() if s else ""


def name_row(c, eff, end, first, middle, last, usage, fka=""):
    upd, ucols = update_cols(eff)
    pfx_desc, pfx_cd = pick(PREFIX[c["gender"]]) if rng.random() < 0.6 else ("", "")
    third = maybe(0.06, fk.first_name())
    gen = maybe(0.03, pick(["JR", "SR", "III"]))
    return {
        **audit_cols(eff, end),
        "PERSON_NAME_ID": int(rng.integers(10**8, 10**9)),
        "PREFIX_NAME_TP_CD": pfx_cd,
        "PREFIX_DESC": pfx_desc,
        "NAME_USAGE_TP_CD": usage,
        "GIVEN_NAME_ONE": first,
        "GIVEN_NAME_TWO": middle,
        "GIVEN_NAME_THREE": third,
        "GIVEN_NAME_FOUR": "",
        "LAST_NAME": last,
        "GENERATION_TP_CD": 109001 if gen else "",
        "SUFFIX_DESC": gen,
        "START_DT": ts(eff),
        "END_DT": ts(end) if end else "",
        "CONT_ID": c["cont_id"],
        "USE_STANDARD_IND": "Y" if rng.random() < 0.9 else "N",
        **ucols,
        "LAST_USED_DT": ts(dt_between(eff, NOW)),
        "LAST_VERIFIED_DT": maybe(0.6, ts(dt_between(eff, NOW))),
        "SOURCE_IDENT_TP_CD": pick([3794, 3795, 3796], [0.5, 0.35, 0.15]),
        "P_LAST_NAME": phon(last),
        "P_GIVEN_NAME_ONE": phon(first),
        "P_GIVEN_NAME_TWO": phon(middle),
        "P_GIVEN_NAME_THREE": phon(third),
        "P_GIVEN_NAME_FOUR": "",
        "X_CREATEDBY_USER": pick(USERS, USER_P),
        "X_CREATED_DT": ts(c["since"]),
        "X_SRC_SYS_LAST_UPD_USER": pick(USERS, USER_P),
        "X_SRC_SYS_LAST_UPD_DT": ts(upd),
        "X_FKA_NAME": fka,
        "X_LAST_VERIFIED_USER": maybe(0.5, pick(USERS, USER_P)),
        "X_LAST_VERIFIED_TRANSIT": c["transit"],
        "X_SRC_SYS_LAST_UPD_TRANSIT": transit(),
    }


personname_rows = []
for c in customers:
    if rng.random() > 0.85:
        continue
    # ~18% of married customers carry a name-change history (maiden name first)
    if c["marital"] == 131866 and rng.random() < 0.18:
        maiden = fk.last_name()
        change = dt_between(c["since"] + timedelta(days=120), NOW - timedelta(days=60))
        personname_rows.append(name_row(c, c["since"], change,
                                        c["first"], c["middle"], maiden, 1))
        personname_rows.append(name_row(c, change, None, c["first"], c["middle"],
                                        c["last"], 1, fka=f"{c['first']} {maiden}"))
    else:
        personname_rows.append(name_row(c, c["since"], None,
                                        c["first"], c["middle"], c["last"], 1))
    if rng.random() < 0.08:            # alias / preferred-name record
        personname_rows.append(name_row(c, dt_between(c["since"], NOW), None,
                                        fk.first_name(), "", c["last"], 2))

# -------------------------------------------------------------------- write
HEADERS = {
    "CONTACT": list(contact_rows[0].keys()),
    "PERSON": list(person_rows[0].keys()),
    "PERSONNAME": list(personname_rows[0].keys()),
}
for name, rows in [("CONTACT", contact_rows), ("PERSON", person_rows),
                   ("PERSONNAME", personname_rows)]:
    df = pd.DataFrame(rows, columns=HEADERS[name])
    df.to_csv(os.path.join(HERE, f"{name}.csv"), index=False)
    print(f"{name}: {len(df)} rows, {df['CONT_ID'].nunique()} customers")
