"""
generate_fake_seed.py
=====================
Generate FAKE seed CSVs for sdg/seed/{CONTACT,PERSON,PERSONNAME}.csv that mimic
the real warehouse schema (column names, value styles, missingness) without
containing any real data.  Value styles reproduced on purpose:

  * long IDs stored as floats (the Excel-mangled `2.68856E+17` style)
  * date columns holding time-like strings such as `43:45.2` / `00:00.0`
  * *_TP_CD / *_CODE type codes drawn from small skewed vocabularies
  * *_IND Y/N flags with blanks
  * gibberish masked names (`GPHGT IQWD`, `Jqsln`, `Vfw`, ...)

A few cross-column dependencies are injected so Column Pair Trends and the
TSTR ML-efficacy metrics have real signal to preserve.

Only numpy + pandas required:  python3 generate_fake_seed.py [n_rows]
"""

import os
import string
import sys

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1600
OUT_DIR = os.path.join("sdg", "seed")


# ----------------------------------------------------------------- helpers
def big_id(n):
    """Excel-mangled long identifier: float around 1e17."""
    return RNG.uniform(1e17, 9.5e17, n).round(-12)


def med_id(n):
    return RNG.uniform(1e12, 9.9e12, n).round(-7)


def audit_id(n):
    return RNG.integers(100000, 9999999, n)


def dtstr(n, missing=0.0):
    """Excel-mangled date -> 'MM:SS.d' string."""
    mm = RNG.integers(0, 60, n)
    ss = RNG.integers(0, 60, n)
    d = RNG.integers(0, 10, n)
    out = pd.Series([f"{a:02d}:{b:02d}.{c}" for a, b, c in zip(mm, ss, d)])
    return mask(out, missing)


def const_dt(n):
    return pd.Series(["00:00.0"] * n)


def mask(s, frac):
    """Blank out a fraction of values."""
    s = pd.Series(s).copy()
    if frac > 0:
        idx = RNG.random(len(s)) < frac
        s[idx] = np.nan
    return s


def codes(n, vocab, p=None, missing=0.0):
    return mask(pd.Series(RNG.choice(vocab, n, p=p)), missing)


def ind(n, p_yes=0.5, missing=0.0):
    return codes(n, ["Y", "N"], [p_yes, 1 - p_yes], missing)


def gibberish(n, lo=3, hi=8, upper=False, missing=0.0):
    letters = string.ascii_uppercase
    out = []
    for _ in range(n):
        L = RNG.integers(lo, hi + 1)
        w = "".join(RNG.choice(list(letters), L))
        out.append(w if upper else w.capitalize())
    return mask(pd.Series(out), missing)


def users(n, missing=0.0):
    return codes(n, ["cusadmin", "MECH", "Startup.", "batchsys", "migr8"],
                 [0.7, 0.1, 0.08, 0.07, 0.05], missing)


# ----------------------------------------------------------------- CONTACT
def make_contact(n):
    # latent client segment drives several columns (signal for pair trends/TSTR)
    seg = RNG.choice([0, 1, 2], n, p=[0.55, 0.3, 0.15])

    client_st = np.select([seg == 0, seg == 1, seg == 2], [100000, 100001, 100002])
    solicit = np.where(RNG.random(n) < np.select([seg == 0, seg == 1, seg == 2], [0.8, 0.4, 0.1]), "Y", "N")
    alert = np.where(RNG.random(n) < np.select([seg == 0, seg == 1, seg == 2], [0.05, 0.25, 0.6]), "Y", "N")
    # numeric columns correlated with segment
    access_token = (RNG.normal(0, 40, n) + np.select([seg == 0, seg == 1, seg == 2], [120, 400, 900])).clip(1).round(0)
    transit = (RNG.normal(0, 400, n) + np.select([seg == 0, seg == 1, seg == 2], [2500, 4500, 7500])).clip(1000, 9999).round(0)

    df = pd.DataFrame({
        "IDP_WAREHOUSE_ID": med_id(n),
        "IDP_AUDIT_ID": audit_id(n),
        "IDP_EFFECTIVE_DATE": const_dt(n),
        "IDP_END_DATE": const_dt(n),
        "IDP_DELETE_DATE": mask([np.nan] * n, 0),
        "CONT_ID": big_id(n),
        "ACCE_COMP_TP_CD": codes(n, [100301, 100302, 100303], [0.5, 0.3, 0.2], 0.35),
        "PREF_LANG_TP_CD": codes(n, [703794, 703795], [0.75, 0.25], 0.1),
        "CREATED_DT": dtstr(n),
        "INACTIVATED_DT": dtstr(n, 0.85),
        "CONTACT_NAME": (gibberish(n, 4, 7, upper=True) + " " + gibberish(n, 3, 6, upper=True)),
        "PERSON_ORG_CODE": codes(n, ["P", "O"], [0.9, 0.1]),
        "SOLICIT_IND": mask(pd.Series(solicit), 0.05),
        "CONFIDENTIAL_IND": codes(n, ["B", "Y", "N"], [0.6, 0.15, 0.25], 0.05),
        "CLIENT_IMP_TP_CD": codes(n, [100100, 100101, 100102, 100103], None, 0.5),
        "CLIENT_ST_TP_CD": pd.Series(client_st),
        "CLIENT_POTEN_TP_CD": codes(n, [100201, 100202, 100203], [0.4, 0.4, 0.2], 0.4),
        "RPTING_FREQ_TP_CD": codes(n, [100401, 100402], [0.7, 0.3], 0.6),
        "LAST_STATEMENT_DT": dtstr(n, 0.5),
        "PROVIDED_BY_CONT": mask(big_id(n), 0.9),
        "ALERT_IND": mask(pd.Series(alert), 0.05),
        "LAST_UPDATE_DT": dtstr(n),
        "LAST_UPDATE_USER": users(n),
        "LAST_UPDATE_TX_ID": big_id(n),
        "DO_NOT_DELETE_IND": ind(n, 0.1, 0.7),
        "LAST_USED_DT": dtstr(n, 0.6),
        "LAST_VERIFIED_DT": dtstr(n, 0.4),
        "SOURCE_IDENT_TP_CD": codes(n, [1, 2, 3], [0.6, 0.3, 0.1], 0.3),
        "SINCE_DT": dtstr(n, 0.2),
        "LEFT_DT": dtstr(n, 0.9),
        "ACCESS_TOKEN_VALUE": pd.Series(access_token),
        "PENDING_CDC_IND": ind(n, 0.6, 0.02),
        "X_OFFICIAL_LANGUAGE_TP_CD": codes(n, [100000, 100001], [0.8, 0.2], 0.05),
        "X_LARGE_CASH_TXN_RPT_IND": codes(n, [888888, 888889], [0.85, 0.15], 0.1),
        "X_CTRY_RES_TP_CD": codes(n, [1001, 1002, 1003, 1004], [0.7, 0.15, 0.1, 0.05], 0.1),
        "X_PROV_RES_TP_CD": codes(n, list(range(2001, 2011)), None, 0.15),
        "X_CREATED_DT": dtstr(n),
        "X_SRC_SYS_LAST_UPD_USER": users(n, 0.1),
        "X_CREATEDBY_USER": users(n, 0.1),
        "X_SRC_SYS_LAST_UPD_DT": dtstr(n, 0.1),
        "X_PRIM_REL": ind(n, 0.7, 0.4),
        "X_BSN_GRP_TP_CD": codes(n, [301, 302, 303], [0.5, 0.3, 0.2], 0.5),
        "X_OFAC_SCREEN_IND": ind(n, 0.9, 0.3),
        "X_OFAC_SCREEN_DT": dtstr(n, 0.6),
        "X_BEN_OWNERSHIP_IND": ind(n, 0.3, 0.5),
        "X_LCTR_EXEMPT_UPDATE_DT": dtstr(n, 0.9),
        "X_LAST_VERIFIED_TRANSIT": pd.Series(transit),
        "X_CRSP_LANG_TP_CD": codes(n, [100000, 100001], [0.8, 0.2], 0.4),
        "X_ICPM_AC_IND": ind(n, 0.2, 0.6),
    })
    return df


# ----------------------------------------------------------------- PERSON
def make_person(n):
    # marital status is the natural TSTR target; tie children count,
    # education, occupation and employment type to it
    marital = RNG.choice([131866, 131867, 131868, 131869], n, p=[0.45, 0.35, 0.12, 0.08])
    p_children = np.select(
        [marital == 131866, marital == 131867, marital == 131868, marital == 131869],
        [0.15, 0.75, 0.55, 0.4])
    children = np.where(RNG.random(n) < p_children, RNG.integers(1, 5, n), 0)
    edu = np.where(marital == 131866,
                   RNG.choice([150123, 150124, 150125], n, p=[0.5, 0.35, 0.15]),
                   RNG.choice([150123, 150124, 150125], n, p=[0.25, 0.45, 0.3]))
    gender = RNG.choice([1, 2], n, p=[0.52, 0.48])
    occ = np.where(gender == 1,
                   RNG.choice([348822, 348823, 348824], n, p=[0.5, 0.3, 0.2]),
                   RNG.choice([348822, 348823, 348824], n, p=[0.25, 0.45, 0.3]))
    empl = np.where(children > 0,
                    RNG.choice([501, 502], n, p=[0.8, 0.2]),
                    RNG.choice([501, 502], n, p=[0.5, 0.5]))
    transit = (RNG.normal(0, 800, n) + 3000 + children * 400).clip(1000, 9999).round(0)

    df = pd.DataFrame({
        "IDP_WAREHOUSE_ID": med_id(n),
        "IDP_AUDIT_ID": audit_id(n),
        "IDP_EFFECTIVE_DATE": const_dt(n),
        "IDP_END_DATE": const_dt(n),
        "IDP_DELETE_DATE": mask([np.nan] * n, 0),
        "CONT_ID": big_id(n),
        "MARITAL_ST_TP_CD": pd.Series(marital),
        "BIRTHPLACE_TP_CD": codes(n, [11001, 11002, 11003], [0.6, 0.25, 0.15], 0.55),
        "CITIZENSHIP_TP_CD": codes(n, [150123, 150126, 150127], [0.8, 0.12, 0.08], 0.1),
        "HIGHEST_EDU_TP_CD": mask(pd.Series(edu), 0.25),
        "AGE_VER_DOC_TP_CD": codes(n, [601, 602], [0.7, 0.3], 0.7),
        "GENDER_TP_CODE": pd.Series(gender),
        "BIRTH_DT": dtstr(n, 0.05),
        "DECEASED_DT": dtstr(n, 0.97),
        "CHILDREN_CT": pd.Series(children),
        "DISAB_START_DT": dtstr(n, 0.95),
        "DISAB_END_DT": dtstr(n, 0.97),
        "USER_IND": ind(n, 0.5, 0.4),
        "LAST_UPDATE_DT": dtstr(n),
        "LAST_UPDATE_USER": users(n),
        "LAST_UPDATE_TX_ID": big_id(n),
        "X_DECEASED_IND": ind(n, 0.03, 0.5),
        "X_OCCUPATION_TP_CD": mask(pd.Series(occ), 0.15),
        "X_COMPANY_TP_CD": codes(n, [100000, 100001, 100002], [0.6, 0.25, 0.15], 0.4),
        "X_EMPL_TP_CD": mask(pd.Series(empl), 0.2),
        "X_SEC_CITIZENSHIP_TP_CD": codes(n, [150126, 150127], [0.6, 0.4], 0.9),
        "X_CITIZENSHIP_LAST_UPD_DT": dtstr(n, 0.6),
        "X_IMPT_FMLY_NAME": gibberish(n, 3, 8, upper=True, missing=0.8),
        "X_US_ALIEN_RES_IND": ind(n, 0.05, 0.6),
        "X_CHILDREN_CT_CAPTURED_DT": dtstr(n, 0.7),
        "X_RESIDENCY_TP_CD": codes(n, [701, 702, 703], [0.75, 0.15, 0.1], 0.3),
        "X_SRC_SYS_LAST_UPD_DT": dtstr(n, 0.1),
        "X_SRC_SYS_LAST_UPD_USER": users(n, 0.1),
        "X_SRC_SYS_LAST_UPD_TRANSIT": pd.Series(transit),
        "X_SRC_SYS_LAST_UPD_USER_NAME": gibberish(n, 4, 9, missing=0.5),
        "X_INCOME_STABILITY_TP_CD": codes(n, [801, 802, 803], [0.5, 0.3, 0.2], 0.5),
    })
    return df


# --------------------------------------------------------------- PERSONNAME
def make_personname(n):
    usage = RNG.choice([1, 2, 3], n, p=[0.7, 0.2, 0.1])
    use_std = np.where(RNG.random(n) < np.select([usage == 1, usage == 2, usage == 3], [0.85, 0.4, 0.15]), "Y", "N")
    src = np.where(usage == 1,
                   RNG.choice([3794, 3795], n, p=[0.8, 0.2]),
                   RNG.choice([3794, 3795], n, p=[0.35, 0.65]))
    transit1 = (RNG.normal(0, 900, n) + np.select([usage == 1, usage == 2, usage == 3], [3000, 5000, 7000])).clip(1000, 9999).round(0)
    transit2 = (transit1 + RNG.normal(0, 500, n)).clip(1000, 9999).round(0)

    df = pd.DataFrame({
        "IDP_WAREHOUSE_ID": med_id(n),
        "IDP_AUDIT_ID": audit_id(n),
        "IDP_EFFECTIVE_DATE": const_dt(n),
        "IDP_END_DATE": const_dt(n),
        "IDP_DELETE_DATE": mask([np.nan] * n, 0),
        "PERSON_NAME_ID": big_id(n),
        "PREFIX_NAME_TP_CD": codes(n, [901, 902, 903], [0.5, 0.35, 0.15], 0.75),
        "PREFIX_DESC": codes(n, ["MR", "MS", "DR"], [0.5, 0.4, 0.1], 0.75),
        "NAME_USAGE_TP_CD": pd.Series(usage),
        "GIVEN_NAME_ONE": gibberish(n, 3, 7),
        "GIVEN_NAME_TWO": gibberish(n, 3, 7, missing=0.6),
        "GIVEN_NAME_THREE": gibberish(n, 3, 7, missing=0.9),
        "GIVEN_NAME_FOUR": gibberish(n, 3, 7, missing=0.97),
        "LAST_NAME": gibberish(n, 3, 8),
        "GENERATION_TP_CD": codes(n, [11, 12, 13], [0.6, 0.3, 0.1], 0.92),
        "SUFFIX_DESC": codes(n, ["JR", "SR", "III"], [0.5, 0.3, 0.2], 0.92),
        "START_DT": dtstr(n, 0.1),
        "END_DT": dtstr(n, 0.9),
        "CONT_ID": big_id(n),
        "USE_STANDARD_IND": mask(pd.Series(use_std), 0.02),
        "LAST_UPDATE_DT": dtstr(n),
        "LAST_UPDATE_USER": users(n),
        "LAST_UPDATE_TX_ID": big_id(n),
        "LAST_USED_DT": dtstr(n, 0.6),
        "LAST_VERIFIED_DT": dtstr(n, 0.5),
        "SOURCE_IDENT_TP_CD": mask(pd.Series(src), 0.2),
        "P_LAST_NAME": gibberish(n, 4, 8, upper=True, missing=0.3),
        "P_GIVEN_NAME_ONE": gibberish(n, 4, 8, upper=True, missing=0.3),
        "P_GIVEN_NAME_TWO": gibberish(n, 4, 8, upper=True, missing=0.8),
        "P_GIVEN_NAME_THREE": gibberish(n, 4, 8, upper=True, missing=0.95),
        "P_GIVEN_NAME_FOUR": gibberish(n, 4, 8, upper=True, missing=0.98),
        "X_CREATEDBY_USER": users(n, 0.1),
        "X_CREATED_DT": dtstr(n),
        "X_SRC_SYS_LAST_UPD_USER": users(n, 0.4),
        "X_SRC_SYS_LAST_UPD_DT": dtstr(n, 0.2),
        "X_FKA_NAME": gibberish(n, 4, 9, missing=0.93),
        "X_LAST_VERIFIED_USER": users(n, 0.7),
        "X_LAST_VERIFIED_TRANSIT": pd.Series(transit1),
        "X_SRC_SYS_LAST_UPD_TRANSIT": pd.Series(transit2),
    })
    return df


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for fname, maker in (("CONTACT.csv", make_contact),
                         ("PERSON.csv", make_person),
                         ("PERSONNAME.csv", make_personname)):
        df = maker(N)
        path = os.path.join(OUT_DIR, fname)
        df.to_csv(path, index=False)
        print(f"wrote {path}  shape={df.shape}")
