"""
GGM workforce knowledge base
============================
Answers workforce questions from the Resource / Headcount files, in code, from the data.

Facts about the data (confirmed on the real files, nothing assumed):
  * Every resource file is a SNAPSHOT: the August file is the workforce as on August, the July file as on
    July. A month is always read from its own file - a missing month is reported, never substituted.
    Comparisons use the previous loaded snapshot (e.g. Aug vs Jul), matched by employee number.
  * Bench = resources whose account (Client Name) contains the word "bench"
    (the data has 'ADSPL - Bench - India' and 'ADSPI - Bench_PH'). Same rule as the app's headcount metric.
  * Headcount counts distinct employees. Bench is reported separately ("excl. bench" = placed on client accounts).
  * Skills come from the PrimarySkills / SecondarySkills columns, split on commas, trimmed, case variants merged,
    counted once per employee.
  * Blank rows are ignored.

Privacy (rules set by the user):
  * Only employee NAMES are shown. Employee numbers, personal details and the personal e-mail are never shown.
  * The official e-mail, when shown, is masked to its first 3 characters (e.g. 'dip***').
  * Nothing personal is ever passed to the language model: it only receives aggregated tables (llm_tables()).

Pure pandas + regex (no Streamlit import) so it can be unit-tested.
"""

import re

import numpy as np
import pandas as pd

from knowledge_base import md_table
from local_folder_loader import MONTH_ORDER, find_months_in_text

WF_VERSION = "2026-10-09"

BENCH_RE = re.compile(r"(?<![a-z])bench(?![a-z])", re.I)   # identical to the app's bench rule
TOP_SKILLS_N = 10
TOP_SKILLS_FIRST_N = 4      # "top skills" counts each resource's FIRST 4 listed skills only (primary and secondary alike)
SHOW_MASKED_EMAIL = True    # False hides the masked e-mail column completely
EMAIL_VISIBLE_CHARS = 3
PRIMARY_SHOWN = 6           # skills shown per person in account rosters (full lists via a person lookup)
SECONDARY_SHOWN = 4
MAX_LIST_ROWS = 80

CANDIDATES = {
    "name": ["Candidate Name", "Employee Name", "Resource Name", "Name"],
    "client": ["Client Name", "Account", "Client"],
    "emp": ["Emp No", "Employee No", "Emp ID", "Employee ID"],
    "primary": ["PrimarySkills", "Primary Skills", "Primary Skill"],
    "secondary": ["SecondarySkills", "Secondary Skills", "Secondary Skill"],
    "email": ["Official Email ID", "Official Email"],          # official e-mail only, never the personal one
    "country": ["Country"], "region": ["Region"], "tower": ["Tower"], "lob": ["LOB"],
    "position": ["Offered Position"],
    "experience": ["TotalExpierenceYear", "TotalExperienceYear", "Total Experience"],
    "city": ["Project Location City"],
    "status": ["Staffing Reason"],
    "term": ["Termination Type"],
    "attrition": ["AttritionDate"],
    "start": ["Staffing Start Date"],
}
DIM_LABELS = {"client": "Account", "country": "Country", "region": "Region", "tower": "Tower",
              "lob": "LOB", "position": "Offered Position", "status": "Staffing Reason"}

WORKFORCE_GUIDE = [
    ("Headcount summary", "Headcount (excl. bench), bench, status split", "headcount  |  how many employees"),
    ("Headcount by dimension", "By account, country, region, tower, LOB, position or status", "headcount by account  |  employees by country"),
    ("Resources in an account", "Names, position, experience, location, status, skills, masked e-mail",
     "resources in ZS Associates  |  names of employees in Deloitte  |  employee details in Asian Development Bank"),
    ("Top skills", "Top primary and secondary skills (each resource's first 4 listed skills) for bench, placed, an account or everyone",
     "top skills among bench  |  top skills placed  |  skillset of ZS Associates"),
    ("Bench", "Bench roster (names, skills) and bench skills", "bench resources  |  who is on bench"),
    ("Who has a skill", "Names of resources with a skill (primary or secondary)", "resources with Snowflake skill"),
    ("One person", "Full details of a named resource (name must match the data)", "details of <full name>"),
    ("Status / exits", "Counts by Staffing Reason and Termination Type; names on request", "attrition  |  who are the closed resources"),
    ("Compare snapshots", "Previous snapshot vs latest: totals, by account, joiners, leavers, moves, bench",
     "compare headcount Aug vs Jul  |  joiners  |  leavers  |  compare bench"),
]


# ----------------------------------------------------------------------------
# Columns and data hygiene
# ----------------------------------------------------------------------------
def detect_resource_columns(df):
    """Real header for each field (case-insensitive), or None when the file does not have it."""
    out = {}
    low = {str(c).strip().lower(): c for c in df.columns}
    for key, names in CANDIDATES.items():
        out[key] = next((low[n.lower()] for n in names if n.lower() in low), None)
    return out


def _clean_text(v):
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return np.nan
    s = re.sub(r"\s+", " ", str(v)).strip()
    return s if s else np.nan


def _emp_key(v):
    if v is None or pd.isna(v):
        return None
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(v).strip()
    except (TypeError, ValueError):
        return str(v).strip()


def prepare_snapshot(df, cols):
    """
    One snapshot ready for reports. Adds internal columns (never displayed): _emp (employee key), _bench.
    Drops blank rows and exact duplicate rows. Returns (frame, info).
    """
    d = df.copy()
    for k in ("name", "client", "position", "status", "term", "country", "city", "tower", "lob", "region"):
        c = cols.get(k)
        if c in d.columns:
            d[c] = d[c].map(_clean_text)
    name, emp, client = cols["name"], cols["emp"], cols["client"]
    d["_emp"] = d[emp].map(_emp_key)
    blank = d["_emp"].isna() & d[name].isna()
    n_blank = int(blank.sum())
    d = d[~blank].copy()
    d["_emp"] = d["_emp"].where(d["_emp"].notna(), "name:" + d[name].astype(str).str.lower())
    before = len(d)
    d = d[~d.drop(columns=[c for c in d.columns if c.startswith("_")]).astype(str).duplicated()].copy()
    d["_bench"] = d[client].map(lambda v: bool(BENCH_RE.search(v)) if isinstance(v, str) else False)
    return d, {"blank_rows_dropped": n_blank, "duplicate_rows_dropped": before - len(d)}


def mask_email(v):
    """First 3 characters + '***'. Never returns the full address."""
    if v is None or (not isinstance(v, str) and pd.isna(v)) or not str(v).strip():
        return "-"
    return str(v).strip()[:EMAIL_VISIBLE_CHARS] + "***"


def split_skills(v):
    """Comma-separated skills -> list, trimmed, empties dropped, repeated skill (any case) once."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return []
    out, seen = [], set()
    for t in str(v).split(","):
        t = re.sub(r"\s+", " ", t).strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def _num(v):
    try:
        return "-" if v is None or pd.isna(v) else f"{float(v):.1f}"
    except (TypeError, ValueError):
        return "-"


def _skills_cell(v, shown):
    sk = split_skills(v)
    if not sk:
        return "-"
    return ", ".join(sk[:shown]) + (f" (+{len(sk) - shown} more)" if len(sk) > shown else "")


def _explode(df, col, emp="_emp", first_n=None):
    """One row per (employee, skill). first_n = keep only each person's first N listed skills (None = all)."""
    s = df[[emp, col]].copy()
    s["_sk"] = s[col].map(lambda v: split_skills(v)[:first_n] if first_n else split_skills(v))
    e = s.explode("_sk").dropna(subset=["_sk"])
    if e.empty:
        return e.assign(Skill=pd.Series(dtype="object"), _k=pd.Series(dtype="object"))
    e["_k"] = e["_sk"].str.lower()
    canon = e.groupby("_k")["_sk"].agg(lambda x: (lambda vc: vc[vc == vc.max()].index.min())(x.value_counts()))
    e["Skill"] = e["_k"].map(canon)
    return e.drop_duplicates([emp, "_k"])


def top_skills(df, col, n=TOP_SKILLS_N, first_n=TOP_SKILLS_FIRST_N):
    """Skill | Resources (distinct employees) | Share of the resources in df. Counts each person's first `first_n` skills."""
    total = df["_emp"].nunique()
    e = _explode(df, col, first_n=first_n)
    if e.empty or not total:
        return pd.DataFrame(columns=["Skill", "Resources", "Share"])
    g = e.groupby("Skill")["_emp"].nunique().rename("Resources").reset_index()
    g["Share"] = g["Resources"] / total * 100
    return g.sort_values(["Resources", "Skill"], ascending=[False, True]).head(n).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Matching accounts / people / skills in a question
# ----------------------------------------------------------------------------
_LEGAL = {"the", "ltd", "limited", "inc", "incorporated", "llc", "llp", "lp", "plc", "corp", "corporation", "co",
          "company", "gmbh", "ag", "sa", "nv", "bv", "pvt", "private"}
_GENERIC_FIRST = {"global", "group", "international", "services", "solutions", "technologies", "technology", "systems",
                  "india", "asia", "pacific", "digital", "data", "consulting", "business", "general", "national",
                  "first", "american", "united", "regional", "singapore", "philippines", "europe", "ascendion"}


def _ntok(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower().replace("&", " and ")).split()


def _acc_tok(s):
    return [w for w in _ntok(s) if w not in _LEGAL]


def match_accounts(question, accounts):
    """
    Accounts named in the question. An account matches when the question contains the START of its name
    (whole words): 'ZS Associates' -> 'ZS Associates India Private Limited'. For accounts sharing a first word the
    most specific match wins; several accounts can still match (all are shown, with their full names).
    """
    qj = " " + " ".join(_ntok(question)) + " "
    hits = {}
    for acc in accounts:
        t = _acc_tok(acc)
        for n in range(len(t), 0, -1):
            pre = " ".join(t[:n])
            if n == 1 and (len(pre) < 4 or pre in _GENERIC_FIRST):
                break
            if f" {pre} " in qj:
                hits[acc] = (n, t[0])
                break
    best = {}
    for acc, (n, first) in hits.items():
        best[first] = max(best.get(first, 0), n)
    return sorted([a for a, (n, first) in hits.items() if n == best[first]])


def match_people(question, names):
    """Resources whose FULL name (2+ words) appears in the question."""
    qj = " " + " ".join(_ntok(question)) + " "
    out = []
    for nm in names:
        t = _ntok(nm)
        if len(t) >= 2 and f" {' '.join(t)} " in qj:
            out.append(nm)
    return sorted(set(out))


_SKILL_MIN = {"ai", "ml", "qa", "bi", "ui", "ux", "hr", "c"}


def match_skills(question, vocab):
    """Known skills named in the question (longest phrases first, no overlaps). vocab: {normalised: display}."""
    qj = " " + " ".join(_ntok(question)) + " "
    found = []
    for key in sorted(vocab, key=len, reverse=True):
        if (len(key) >= 3 or key in _SKILL_MIN) and f" {key} " in qj:
            qj = qj.replace(f" {key} ", " ")
            found.append(vocab[key])
    return found


# ----------------------------------------------------------------------------
# Context
# ----------------------------------------------------------------------------
def make_context(res_df, selected_month, question="", source_col="Source_File"):
    """
    Snapshot context for one question.
      snapshot month = the later of two loaded months named in the question, else the one month named, else the selected month
      previous snapshot = the other named month, else the previous loaded snapshot.
    Returns {'ok': False, 'why': ...} when it cannot be built.
    """
    if res_df is None or len(res_df) == 0 or "Month" not in res_df.columns:
        return {"ok": False, "why": "No resource / headcount file is loaded. Upload the resource files to ask workforce questions."}
    cols = detect_resource_columns(res_df)
    missing = [k for k in ("name", "client", "emp") if not cols.get(k)]
    if missing:
        return {"ok": False, "why": "The resource file has no column for: " + ", ".join(missing)}
    months = [m for m in MONTH_ORDER if m in set(res_df["Month"])]
    named = find_months_in_text(question, months)
    named = sorted(set(named), key=MONTH_ORDER.index)
    snap_m = named[-1] if named else selected_month
    if snap_m not in months:
        return {"ok": False, "why": (f"No resource snapshot is loaded for {snap_m} (loaded: {', '.join(months)}). "
                                     f"A month is only read from its own file; no other month is substituted.")}
    i = months.index(snap_m)
    prev_m = named[-2] if len(named) >= 2 else (months[i - 1] if i > 0 else None)

    def _snap(m):
        sub = res_df[res_df["Month"] == m]
        return prepare_snapshot(sub, cols)

    snap, info = _snap(snap_m)
    prev, pinfo = _snap(prev_m) if prev_m else (None, None)
    files = {}
    if source_col in res_df.columns:
        files = res_df.groupby("Month")[source_col].first().to_dict()

    client, name = cols["client"], cols["name"]
    vocab = {}
    for c in (cols.get("primary"), cols.get("secondary")):
        if c in snap.columns:
            for s in _explode(snap, c)["Skill"].unique():
                k = " ".join(_ntok(s))
                if k:
                    vocab.setdefault(k, s)
    known = {"accounts": sorted(snap[client].dropna().unique()), "names": sorted(snap[name].dropna().unique()), "skills": vocab}
    return {"ok": True, "cols": cols, "snap": snap, "prev": prev, "snap_month": snap_m, "prev_month": prev_m,
            "months": months, "files": files, "info": info, "prev_info": pinfo, "known": known}


def resource_health(ctx):
    """Short data-quality lines for the UI."""
    if not ctx.get("ok"):
        return [ctx.get("why", "")]
    s, cols = ctx["snap"], ctx["cols"]
    lines = [f"Snapshot {ctx['snap_month']}: {s['_emp'].nunique()} employees from {ctx['files'].get(ctx['snap_month'], 'n/a')}."]
    bench = sorted(s.loc[s["_bench"], cols["client"]].dropna().unique())
    lines.append("Bench accounts detected: " + (", ".join(bench) if bench else "none") + ".")
    if ctx["info"]["blank_rows_dropped"] or ctx["info"]["duplicate_rows_dropped"]:
        lines.append(f"Ignored {ctx['info']['blank_rows_dropped']} blank and {ctx['info']['duplicate_rows_dropped']} duplicate rows.")
    both = set(s.loc[s["_bench"], "_emp"]) & set(s.loc[~s["_bench"], "_emp"])
    if both:
        lines.append(f"{len(both)} employee(s) appear on a bench account AND a client account (rows kept as in the data).")
    lines.append("Loaded snapshots: " + ", ".join(ctx["months"]) + ".")
    return lines


# ----------------------------------------------------------------------------
# Question routing
# ----------------------------------------------------------------------------
_WF = re.compile(r"\b(resource|resources|employee|employees|headcount|head count|people|manpower|workforce|staff|"
                 r"bench|benched|skill|skills|skillset|skill set|joiner|joiners|leaver|leavers|attrition|"
                 r"resign\w*|new hires?|hires?|team members?|associates)\b", re.I)
_COMPARE = re.compile(r"\b(compare|comparison|vs\.?|versus|differences?|changes?|changed|movement|moved|moves|joiners?|"
                      r"leavers?|new hires?|hires?|exits?|added|dropped|transfers?)\b", re.I)
_STATUS = re.compile(r"\b(attrition|resign\w*|terminat\w*|closed|notice|exits?|involuntary|voluntary)\b", re.I)
_SKILL = re.compile(r"\b(skill|skills|skillset|skill set|skilled|expertise|tech stack|proficien\w*|knowledge)\b", re.I)
_SKILL_FIND = re.compile(r"\b(who|which|list|names?|show|find|resources?|employees?|people|having|has|have|with|know|knows)\b", re.I)
_BENCHW = re.compile(r"\bbench(ed)?\b", re.I)
_PLACED = re.compile(r"\b(placed|deployed|billable|on accounts?|staffed|on client)\b", re.I)
_ROSTER = re.compile(r"\b(names?|list|details?|info|information|who|roster|members?|resources?|employees?|people|staff|team|contact|email|profile|skillset|skills?)\b", re.I)
_COUNT = re.compile(r"\b(headcount|head count|how many|count|number of|size|strength|total)\b", re.I)
_DIMS = [("client", re.compile(r"\b(accounts?|clients?)\b", re.I)), ("country", re.compile(r"\b(countr(y|ies))\b", re.I)),
         ("region", re.compile(r"\bregions?\b", re.I)), ("tower", re.compile(r"\btowers?\b", re.I)),
         ("lob", re.compile(r"\b(lob|line of business)\b", re.I)),
         ("position", re.compile(r"\b(positions?|roles?|designations?)\b", re.I)),
         ("status", re.compile(r"\b(status|staffing reason)\b", re.I))]
_BYWORD = re.compile(r"\b(by|per|each|every|wise|across|split|breakdown|distribution)\b", re.I)


_FIN = re.compile(r"\b(revenue|gpm|gp|margin|profit|cost|billing|ytd|qtd)\b", re.I)


def looks_workforce(question):
    return bool(_WF.search(question or ""))


def match_workforce(question, known=None):
    """Which workforce report(s) does the question ask for? [] = none (falls through)."""
    q = (question or "").lower()
    known = known or {}
    if match_people(question, known.get("names", ())):
        return ["wf_person"]
    if not _WF.search(q) or _FIN.search(q):      # money questions ("revenue per resource") belong to the finance path
        return []
    accounts = match_accounts(question, known.get("accounts", ()))
    bench = bool(_BENCHW.search(q))
    if _COMPARE.search(q):
        return ["wf_compare"]
    if _STATUS.search(q):
        return ["wf_status"]
    if _SKILL.search(q):
        if bench and re.search(r"\b(names?|who|list|resources?|employees?|people|details?|roster)\b", q) and not re.search(r"\btop\b", q):
            return ["wf_bench"]
        if known.get("skills") and not re.search(r"\btop\b", q) and _SKILL_FIND.search(q) and match_skills(question, known["skills"]) \
                and not accounts and not bench:
            return ["wf_skill_search"]
        return ["wf_top_skills"]
    if bench:
        return ["wf_summary"] if _COUNT.search(q) and not re.search(r"\b(names?|who|list|resources?|employees?|people|details?)\b", q) \
            else ["wf_bench"]
    dim = next((k for k, rx in _DIMS if rx.search(q)), None)
    if accounts:
        return ["wf_by_dim"] if _COUNT.search(q) and not re.search(r"\b(names?|who|list|details?|members?)\b", q) else ["wf_roster"]
    if dim and (_BYWORD.search(q) or _COUNT.search(q)):
        return ["wf_by_dim"]
    if _COUNT.search(q):
        return ["wf_summary"]
    if re.search(r"\b(names?|list|details?|who|members?|roster)\b", q):
        return ["wf_roster"]
    return []


# ----------------------------------------------------------------------------
# Report sections
# ----------------------------------------------------------------------------
def _fx(ctx, month=None):
    month = month or ctx["snap_month"]
    return ctx["files"].get(month, "n/a")


def _head(ctx, title):
    return f"**{title}** — {ctx['snap_month']} snapshot (as on {ctx['snap_month']}, file: {_fx(ctx)})"


def _counts(df):
    placed = df.loc[~df["_bench"], "_emp"].nunique()
    bench = df.loc[df["_bench"], "_emp"].nunique()
    return {"total": df["_emp"].nunique(), "placed": placed, "bench": bench}


def sec_summary(ctx):
    s, cols = ctx["snap"], ctx["cols"]
    c = _counts(s)
    rows = [("Headcount (excl. bench)", c["placed"]), ("Bench", c["bench"]), ("Total distinct employees", c["total"])]
    st = cols.get("status")
    if st in s.columns:
        for k, g in s.groupby(s[st].fillna("(blank)")):
            rows.append((f"Staffing Reason: {k}", g["_emp"].nunique()))
    notes = ["Headcount counts distinct employees; 'excl. bench' = employees on client accounts."]
    both = set(s.loc[s["_bench"], "_emp"]) & set(s.loc[~s["_bench"], "_emp"])
    if both:
        notes.append(f"{len(both)} employee(s) appear on both a bench account and a client account, so Headcount + Bench exceeds the total by {len(both)}.")
    return _head(ctx, "Workforce summary") + "\n\n" + md_table(["Metric", "Value"], rows) + "\n\n" + "\n".join(f"_{n}_" for n in notes)


def sec_by_dim(ctx, dim, accounts=None):
    s, cols = ctx["snap"], ctx["cols"]
    col = cols.get(dim)
    label = DIM_LABELS[dim]
    if not col or col not in s.columns:
        return f"**Headcount by {label}**\n\n_The resource file has no {label} column._"
    d = s if not accounts else s[s[cols["client"]].isin(accounts)]
    d = d.assign(_v=d[col].fillna("(blank)"))
    act = d[~d["_bench"]].groupby("_v")["_emp"].nunique()
    ben = d[d["_bench"]].groupby("_v")["_emp"].nunique()
    t = pd.DataFrame({"a": act, "b": ben}).fillna(0).astype(int)
    t["_s"] = t["a"] + t["b"]
    t = t.sort_index().sort_values("_s", ascending=False, kind="stable")      # biggest first, ties alphabetical
    rows = [(i, r["a"], r["b"]) for i, r in t.iterrows()]
    c = _counts(d)
    rows.append(("Total (distinct employees)", c["placed"], c["bench"]))
    note = (f"An employee on more than one {label.lower()} is counted under each, so rows can add up to more than the total."
            if int(t["a"].sum()) != c["placed"] or int(t["b"].sum()) != c["bench"] else "")
    return (_head(ctx, f"Headcount by {label}") + "\n\n" + md_table([label, "Headcount (excl. bench)", "Bench"], rows[:MAX_LIST_ROWS + 1])
            + (f"\n\n_Showing {MAX_LIST_ROWS} of {len(rows) - 1} rows._" if len(rows) - 1 > MAX_LIST_ROWS else "")
            + (f"\n\n_{note}_" if note else ""))


def _roster_rows(df, cols, names_only, include_account=False):
    nm, pos, st = cols["name"], cols.get("position"), cols.get("status")
    head = ["Name"] + (["Account"] if include_account else []) + ["Offered Position"] + (["Staffing Reason"] if st else [])
    if not names_only:
        head += ["Experience (yrs)", "Location", "Primary skills", "Secondary skills"]
        if SHOW_MASKED_EMAIL and cols.get("email"):
            head += ["Email (masked)"]
    rows = []
    for _, r in df.sort_values(nm, key=lambda s: s.astype(str).str.lower()).iterrows():
        row = [r[nm]] + ([r[cols["client"]]] if include_account else []) + [r[pos] if pos and pd.notna(r.get(pos)) else "-"]
        if st:
            row.append(r[st] if pd.notna(r.get(st)) else "-")
        if not names_only:
            row.append(_num(r.get(cols["experience"])) if cols.get("experience") else "-")
            loc = ", ".join(str(r[c]) for c in (cols.get("city"), cols.get("country")) if c and pd.notna(r.get(c)))
            row.append(loc or "-")
            row.append(_skills_cell(r.get(cols.get("primary")), PRIMARY_SHOWN) if cols.get("primary") else "-")
            row.append(_skills_cell(r.get(cols.get("secondary")), SECONDARY_SHOWN) if cols.get("secondary") else "-")
            if SHOW_MASKED_EMAIL and cols.get("email"):
                row.append(mask_email(r.get(cols["email"])))
        rows.append(row)
    return head, rows


def sec_roster(ctx, question, accounts):
    s, cols = ctx["snap"], ctx["cols"]
    if not accounts:
        accs = ctx["known"]["accounts"]
        return (_head(ctx, "Resources in an account") +
                "\n\n_Name the account, for example 'resources in ZS Associates'. Accounts in this snapshot:_\n\n" +
                "\n".join(f"- {a}" for a in accs[:MAX_LIST_ROWS]) +
                (f"\n\n_(+{len(accs) - MAX_LIST_ROWS} more)_" if len(accs) > MAX_LIST_ROWS else ""))
    names_only = bool(re.search(r"\bnames?\b", question, re.I)) and not re.search(
        r"\b(skill|skills|skillset|detail|details|email|contact|profile|info|information|experience|location)\b", question, re.I)
    parts = []
    for acc in accounts:
        d = s[s[cols["client"]] == acc]
        head, rows = _roster_rows(d, cols, names_only)
        shown = rows[:MAX_LIST_ROWS]
        parts.append(f"**{acc}** — {d['_emp'].nunique()} resources" + (" (bench account)" if d["_bench"].any() else "") + "\n\n" +
                     md_table(head, shown) + (f"\n\n_Showing {MAX_LIST_ROWS} of {len(rows)} rows._" if len(rows) > MAX_LIST_ROWS else ""))
        if not names_only and re.search(r"\b(skill|skills|skillset)\b", question, re.I):
            parts.append(sec_skill_tables(ctx, d, f"{acc}", TOP_SKILLS_N, with_head=False))
    foot = ("Employee numbers are never shown; the e-mail is masked to its first 3 characters. Skills lists are cut to the first "
            f"{PRIMARY_SHOWN} primary / {SECONDARY_SHOWN} secondary for display - ask for a person by full name to see all of them."
            if not names_only else "Employee numbers and e-mails are not shown.")
    return _head(ctx, "Resources in account") + "\n\n" + "\n\n".join(parts) + f"\n\n_{foot}_"


def sec_skill_tables(ctx, df, label, n, with_head=True):
    cols = ctx["cols"]
    total = df["_emp"].nunique()
    out = [(_head(ctx, f"Top skills — {label}") if with_head else f"**Top skills — {label}**") + f" ({total} resources)"]
    for key, title in (("primary", "Primary skills"), ("secondary", "Secondary skills")):
        c = cols.get(key)
        if not c or c not in df.columns:
            out.append(f"_{title}: the file has no such column._")
            continue
        t = top_skills(df, c, n)
        empty = int(df.groupby("_emp")[c].apply(lambda s: not any(split_skills(v) for v in s)).sum())
        if t.empty:
            out.append(f"**{title}** — none listed for these resources.")
            continue
        out.append(f"**{title}**\n\n" + md_table(["Skill", "Resources", "Share of resources"],
                                                 [(r["Skill"], int(r["Resources"]), f"{r['Share']:.1f}%") for _, r in t.iterrows()]))
        notes = []
        if empty:
            notes.append(f"{empty} of {total} resources have no {title.lower()} listed.")
        cut = int(df.groupby("_emp")[c].apply(lambda s: max((len(split_skills(v)) for v in s), default=0) > TOP_SKILLS_FIRST_N).sum())
        notes.append(f"Counts use each resource's first {TOP_SKILLS_FIRST_N} listed {title.lower()} only"
                     + (f" ({cut} resources list more; the rest are not counted here)." if cut else "."))
        if notes:
            out.append("_" + " ".join(notes) + "_")
    return "\n\n".join(out)


def sec_top_skills(ctx, question, accounts):
    s, cols = ctx["snap"], ctx["cols"]
    mt = re.search(r"\btop\s*(\d{1,2})\b", question, re.I)
    n = int(mt.group(1)) if mt else TOP_SKILLS_N
    if accounts:
        return "\n\n".join(sec_skill_tables(ctx, s[s[cols["client"]] == a], a, n) for a in accounts)
    if _BENCHW.search(question):
        return sec_skill_tables(ctx, s[s["_bench"]], "bench resources", n)
    if _PLACED.search(question):
        return sec_skill_tables(ctx, s[~s["_bench"]], "placed resources (client accounts, bench excluded)", n)
    return sec_skill_tables(ctx, s, "all resources in the snapshot (bench included)", n)


def sec_bench(ctx, question):
    s, cols = ctx["snap"], ctx["cols"]
    b = s[s["_bench"]]
    if b.empty:
        return _head(ctx, "Bench") + "\n\n_No resource is on a bench account in this snapshot._"
    names_only = bool(re.search(r"\bnames?\b", question, re.I)) and not _SKILL.search(question)
    head, rows = _roster_rows(b, cols, names_only, include_account=True)
    out = [_head(ctx, "Bench resources") + f" — {b['_emp'].nunique()} resources on " +
           ", ".join(sorted(b[cols['client']].dropna().unique())) + "\n\n" + md_table(head, rows)]
    if _SKILL.search(question) or not names_only:
        out.append(sec_skill_tables(ctx, b, "bench resources", TOP_SKILLS_N, with_head=False))
    out.append("_Bench = resources on accounts whose name contains 'bench'. Employee numbers are never shown; e-mail masked to 3 characters._")
    return "\n\n".join(out)


def sec_skill_search(ctx, question):
    s, cols = ctx["snap"], ctx["cols"]
    found = match_skills(question, ctx["known"]["skills"])
    if not found:
        return _head(ctx, "Who has a skill") + "\n\n_No skill from the data was recognised in the question._"
    parts = []
    for sk in found:
        keyl = sk.lower()
        mask = pd.Series(False, index=s.index)
        which = pd.Series("", index=s.index)
        for key, lab in (("primary", "Primary"), ("secondary", "Secondary")):
            c = cols.get(key)
            if c in s.columns:
                has = s[c].map(lambda v: keyl in [x.lower() for x in split_skills(v)])
                mask |= has
                which = which.where(~has, which + lab[0])
        d = s[mask]
        rows = []
        for _, r in d.sort_values(cols["name"], key=lambda x: x.astype(str).str.lower()).iterrows():
            tag = which[r.name]
            rows.append((r[cols["name"]], r[cols["client"]] + (" (bench)" if r["_bench"] else ""),
                         "Primary + Secondary" if tag == "PS" else ("Primary" if tag == "P" else "Secondary"),
                         r[cols["position"]] if cols.get("position") and pd.notna(r.get(cols["position"])) else "-"))
        parts.append(f"**{sk}** — {d['_emp'].nunique()} resources\n\n" +
                     (md_table(["Name", "Account", "Listed as", "Offered Position"], rows[:MAX_LIST_ROWS]) if rows else "_None._") +
                     (f"\n\n_Showing {MAX_LIST_ROWS} of {len(rows)} rows._" if len(rows) > MAX_LIST_ROWS else ""))
    return _head(ctx, "Resources with the skill") + "\n\n" + "\n\n".join(parts) + "\n\n_Skill matched as written in the Primary / Secondary skills columns (case ignored)._"


def sec_person(ctx, people):
    s, cols = ctx["snap"], ctx["cols"]
    out = []
    for p in people:
        d = s[s[cols["name"]] == p]
        for emp, g in d.groupby("_emp", sort=False):
            r = g.iloc[0]
            accts = "; ".join(f"{a}{' (bench)' if b else ''}" for a, b in zip(g[cols["client"]], g["_bench"]))

            def v(key):
                c = cols.get(key)
                return r[c] if c and c in g.columns and pd.notna(r[c]) else "-"
            rows = [("Name", p), ("Account", accts), ("Offered Position", v("position")), ("Staffing Reason", v("status")),
                    ("Termination Type", v("term")), ("Experience (yrs)", v("experience")),
                    ("Location", ", ".join(str(x) for x in (v("city"), v("country")) if x != "-") or "-"),
                    ("Primary skills", ", ".join(split_skills(r.get(cols.get("primary")))) or "-" if cols.get("primary") else "-"),
                    ("Secondary skills", ", ".join(split_skills(r.get(cols.get("secondary")))) or "-" if cols.get("secondary") else "-")]
            if SHOW_MASKED_EMAIL and cols.get("email"):
                rows.append(("Email (masked)", mask_email(r.get(cols["email"]))))
            out.append(md_table(["Field", "Value"], rows))
    return _head(ctx, "Resource details") + "\n\n" + "\n\n".join(out) + "\n\n_Employee numbers and personal details are never shown._"


def sec_status(ctx, question):
    s, cols = ctx["snap"], ctx["cols"]
    st, tt = cols.get("status"), cols.get("term")
    if not st or st not in s.columns:
        return _head(ctx, "Status") + "\n\n_The resource file has no Staffing Reason column._"
    t = s.groupby(s[st].fillna("(blank)"))["_emp"].nunique().sort_values(ascending=False)
    out = [_head(ctx, "Staffing status") + "\n\n" + md_table(["Staffing Reason", "Resources"], [(k, v) for k, v in t.items()])]
    if tt and tt in s.columns and s[tt].notna().any():
        t2 = s[s[tt].notna()].groupby(tt)["_emp"].nunique().sort_values(ascending=False)
        out.append("**Termination Type** (rows that have one)\n\n" + md_table(["Termination Type", "Resources"], [(k, v) for k, v in t2.items()]))
    if re.search(r"\b(who|names?|list|details?)\b", question, re.I):
        d = s[s[st].astype(str).str.lower() == "closed"]
        rows = []
        ad = cols.get("attrition")
        for _, r in d.sort_values(cols["name"], key=lambda x: x.astype(str).str.lower()).iterrows():
            dt = pd.to_datetime(r.get(ad), errors="coerce") if ad else pd.NaT
            rows.append((r[cols["name"]], r[cols["client"]], r[tt] if tt and pd.notna(r.get(tt)) else "-",
                         dt.strftime("%Y-%m-%d") if pd.notna(dt) else "-"))
        out.append(f"**Staffing Reason = Closed** ({d['_emp'].nunique()} resources)\n\n" +
                   md_table(["Name", "Account", "Termination Type", "Attrition Date"], rows[:MAX_LIST_ROWS]))
    out.append("_Labels are shown exactly as in the file; no interpretation is applied (e.g. 'Closed' is not treated as left)._")
    return "\n\n".join(out)


def sec_compare(ctx, question):
    s, p, cols = ctx["snap"], ctx["prev"], ctx["cols"]
    cm, pm = ctx["snap_month"], ctx["prev_month"]
    if p is None or not pm:
        return (_head(ctx, "Snapshot comparison") +
                f"\n\n_No earlier resource snapshot is loaded to compare with (loaded: {', '.join(ctx['months'])}). Upload the previous month's file._")
    q = question.lower()
    focus = ("bench" if _BENCHW.search(q) else "joiners" if re.search(r"joiner|new hire|hires?|added", q) else
             "leavers" if re.search(r"leaver|exit|attrition|resign|dropped", q) else
             "moves" if re.search(r"\bmove|moved|movement|transfer", q) else "all")
    client, name, st = cols["client"], cols["name"], cols.get("status")
    out = [f"**Workforce comparison — {cm} vs {pm}** (each month from its own snapshot: {_fx(ctx, cm)} vs {_fx(ctx, pm)})"]

    # totals
    cc, pc = _counts(s), _counts(p)
    rows = [("Headcount (excl. bench)", pc["placed"], cc["placed"], f"{cc['placed'] - pc['placed']:+d}"),
            ("Bench", pc["bench"], cc["bench"], f"{cc['bench'] - pc['bench']:+d}"),
            ("Total distinct employees", pc["total"], cc["total"], f"{cc['total'] - pc['total']:+d}")]
    if st in s.columns and st in p.columns:
        labs = sorted(set(s[st].dropna()) | set(p[st].dropna()))
        for l in labs:
            a = p[p[st] == l]["_emp"].nunique()
            b = s[s[st] == l]["_emp"].nunique()
            rows.append((f"Staffing Reason: {l}", a, b, f"{b - a:+d}"))
    out.append("**Totals**\n\n" + md_table(["Metric", pm, cm, "Change"], rows))

    ce, pe = set(s["_emp"]), set(p["_emp"])
    join_keys, leave_keys = ce - pe, pe - ce

    def _who(df, keys, extra_prev=False):
        d = df[df["_emp"].isin(keys)].sort_values(name, key=lambda x: x.astype(str).str.lower())
        pos = cols.get("position")
        return [(r[name], r[client], r[pos] if pos and pd.notna(r.get(pos)) else "-") for _, r in d.iterrows()]

    if focus in ("all", "joiners"):
        j = _who(s, join_keys)
        out.append(f"**Joiners** — in {cm} but not in {pm}: {len(join_keys)}\n\n" + (md_table(["Name", "Account", "Offered Position"], j[:MAX_LIST_ROWS]) if j else "_None._")
                   + (f"\n\n_Showing {MAX_LIST_ROWS} of {len(j)}._" if len(j) > MAX_LIST_ROWS else ""))
    if focus in ("all", "leavers"):
        l = _who(p, leave_keys)
        out.append(f"**Leavers** — in {pm} but not in {cm}: {len(leave_keys)}\n\n" + (md_table(["Name", f"Account ({pm})", "Offered Position"], l[:MAX_LIST_ROWS]) if l else "_None._")
                   + (f"\n\n_Showing {MAX_LIST_ROWS} of {len(l)}._" if len(l) > MAX_LIST_ROWS else ""))
    if focus in ("all", "moves"):
        sa = s.groupby("_emp")[client].apply(lambda x: sorted(set(x.dropna())))
        pa = p.groupby("_emp")[client].apply(lambda x: sorted(set(x.dropna())))
        nm = s.drop_duplicates("_emp").set_index("_emp")[name]
        mv = [(nm[e], " ; ".join(pa[e]), " ; ".join(sa[e])) for e in sorted(ce & pe, key=lambda k: str(nm[k]).lower())
              if sa[e] != pa[e]]
        out.append(f"**Account moves** — in both snapshots, different account(s): {len(mv)}\n\n" +
                   (md_table(["Name", f"{pm} account", f"{cm} account"], mv[:MAX_LIST_ROWS]) if mv else "_None._"))
    if focus == "all":
        ca = s.groupby(client)["_emp"].nunique()
        pa2 = p.groupby(client)["_emp"].nunique()
        t = pd.DataFrame({pm: pa2, cm: ca}).fillna(0).astype(int)
        t["d"] = t[cm] - t[pm]
        ch = t[t["d"] != 0].reindex(t[t["d"] != 0]["d"].abs().sort_values(ascending=False).index)
        out.append(f"**Headcount by account — accounts with a change** ({len(ch)} of {len(t)}; bench accounts included)\n\n" +
                   md_table(["Account", pm, cm, "Change"], [(i, r[pm], r[cm], f"{r['d']:+d}") for i, r in ch.head(MAX_LIST_ROWS).iterrows()]))
    if focus == "bench":
        for lab, df in ((pm, p), (cm, s)):
            b = df[df["_bench"]]
            out.append(f"**Bench on {lab}** — {b['_emp'].nunique()}\n\n" +
                       (md_table(["Name", "Account", "Offered Position"], [(r[name], r[client], r[cols['position']] if cols.get('position') and pd.notna(r.get(cols['position'])) else '-')
                                                                         for _, r in b.sort_values(name, key=lambda x: x.astype(str).str.lower()).iterrows()]) if len(b) else "_None._"))
    if _SKILL.search(q):
        scope_s, scope_p, lab = ((s[s["_bench"]], p[p["_bench"]], "bench") if _BENCHW.search(q) else (s[~s["_bench"]], p[~p["_bench"]], "placed resources"))
        c = cols.get("primary")
        if c in s.columns and c in p.columns:
            a, b = top_skills(scope_p, c, 200).set_index("Skill")["Resources"], top_skills(scope_s, c, 200).set_index("Skill")["Resources"]
            t = pd.DataFrame({pm: a, cm: b}).fillna(0).astype(int)
            t["_m"] = t[[pm, cm]].max(axis=1)
            t = t.sort_values(["_m"], ascending=False, kind="stable").head(TOP_SKILLS_N)
            out.append(f"**Top primary skills — {lab}: {pm} vs {cm}** (resources per skill; first {TOP_SKILLS_FIRST_N} listed skills per resource)\n\n" +
                       md_table(["Skill", pm, cm, "Change"], [(i, r[pm], r[cm], f"{r[cm] - r[pm]:+d}") for i, r in t.iterrows()]))
    out.append("_Employees are matched between the two snapshots by their employee number (never shown). Only names are displayed._")
    return "\n\n".join(out)


KIND = {"wf_summary", "wf_by_dim", "wf_roster", "wf_top_skills", "wf_skill_search", "wf_bench", "wf_person", "wf_status", "wf_compare"}


def build_workforce_report(ids, question, ctx):
    """Render a workforce report. ctx must be ok. Returns {'markdown', 'ids'}."""
    s = ctx["snap"]
    accounts = match_accounts(question, ctx["known"]["accounts"])
    out = []
    for i in ids:
        if i == "wf_summary":
            out.append(sec_summary(ctx))
        elif i == "wf_by_dim":
            dim = next((k for k, rx in _DIMS if rx.search(question)), "client")
            out.append(sec_by_dim(ctx, dim, accounts if accounts else None))
        elif i == "wf_roster":
            out.append(sec_roster(ctx, question, accounts))
        elif i == "wf_top_skills":
            out.append(sec_top_skills(ctx, question, accounts))
        elif i == "wf_skill_search":
            out.append(sec_skill_search(ctx, question))
        elif i == "wf_bench":
            out.append(sec_bench(ctx, question))
        elif i == "wf_person":
            out.append(sec_person(ctx, match_people(question, ctx["known"]["names"])))
        elif i == "wf_status":
            out.append(sec_status(ctx, question))
        elif i == "wf_compare":
            out.append(sec_compare(ctx, question))
    notes = [f"Matched account(s): {', '.join(accounts)}." ] if accounts and any(i in ids for i in ("wf_roster", "wf_top_skills", "wf_by_dim")) else []
    if ctx["info"]["blank_rows_dropped"] or ctx["info"]["duplicate_rows_dropped"]:
        notes.append(f"Ignored {ctx['info']['blank_rows_dropped']} blank and {ctx['info']['duplicate_rows_dropped']} duplicate rows in the {ctx['snap_month']} file.")
    md = "\n\n".join(out) + ("\n\n" + "\n".join(f"_{n}_" for n in notes) if notes else "")
    return {"markdown": md, "ids": ids}


# ----------------------------------------------------------------------------
# For the language model: aggregates only, NO names / ids / e-mails
# ----------------------------------------------------------------------------
def _csv(df):
    return df.round(2).to_csv(index=False).strip()


def llm_tables(ctx, top_n=15):
    """Aggregated workforce tables (counts only). Nothing personal is included."""
    if not ctx or not ctx.get("ok"):
        return []
    s, p, cols = ctx["snap"], ctx["prev"], ctx["cols"]
    cm, pm = ctx["snap_month"], ctx["prev_month"]
    out = [f"WORKFORCE TABLES - {cm} snapshot (as on {cm}; employee names, numbers and e-mails are NOT available to you)"]
    c = _counts(s)
    rows = [("Headcount_excl_bench", c["placed"]), ("Bench", c["bench"]), ("Total_distinct_employees", c["total"])]
    st = cols.get("status")
    if st in s.columns:
        rows += [(f"Staffing_Reason:{k}", g["_emp"].nunique()) for k, g in s.groupby(s[st].fillna("(blank)"))]
    if p is not None:
        pc = _counts(p)
        rows += [(f"{pm}_Headcount_excl_bench", pc["placed"]), (f"{pm}_Bench", pc["bench"]), (f"{pm}_Total_distinct_employees", pc["total"])]
    out.append("TABLE W1 - HEADCOUNT SUMMARY (distinct employees; bench = accounts named like 'bench')\n" + _csv(pd.DataFrame(rows, columns=["Metric", "Value"])))
    client = cols["client"]
    a = s.groupby(client)["_emp"].nunique().rename("Headcount").reset_index()
    a["Type"] = a[client].map(lambda v: "Bench" if BENCH_RE.search(str(v)) else "Client")
    a = a.sort_values(["Headcount", client], ascending=[False, True])
    out.append(f"TABLE W2 - HEADCOUNT BY ACCOUNT ({'top ' + str(top_n) + ' of ' + str(len(a)) if len(a) > top_n else 'all ' + str(len(a))})\n" + _csv(a.head(top_n)))
    if cols.get("country") in s.columns:
        g = s.groupby(s[cols["country"]].fillna("(blank)"))["_emp"].nunique().rename("Headcount").reset_index().sort_values("Headcount", ascending=False)
        out.append("TABLE W3 - HEADCOUNT BY COUNTRY\n" + _csv(g))
    for key, lab in (("primary", "Primary"), ("secondary", "Secondary")):
        c_ = cols.get(key)
        if c_ in s.columns:
            t = top_skills(s[~s["_bench"]], c_, top_n)
            if len(t):
                out.append(f"TABLE W4{key[0].upper()} - TOP {lab.upper()} SKILLS, PLACED RESOURCES (resources per skill; first {TOP_SKILLS_FIRST_N} listed skills per resource; top {top_n})\n" + _csv(t[["Skill", "Resources"]]))
            b = top_skills(s[s["_bench"]], c_, top_n)
            if len(b):
                out.append(f"TABLE W5{key[0].upper()} - TOP {lab.upper()} SKILLS, BENCH RESOURCES (first {TOP_SKILLS_FIRST_N} listed skills per resource)\n" + _csv(b[["Skill", "Resources"]]))
    if p is not None:
        pa = p.groupby(client)["_emp"].nunique()
        t = pd.DataFrame({pm: pa, cm: s.groupby(client)["_emp"].nunique()}).fillna(0).astype(int)
        t["Change"] = t[cm] - t[pm]
        t = t[t["Change"] != 0]
        t = t.reindex(t["Change"].abs().sort_values(ascending=False).index).head(top_n).reset_index().rename(columns={"index": client})
        joiners = len(set(s["_emp"]) - set(p["_emp"]))
        leavers = len(set(p["_emp"]) - set(s["_emp"]))
        out.append(f"TABLE W6 - {cm} vs {pm}: joiners {joiners}, leavers {leavers}; accounts with a headcount change (largest first)\n" + _csv(t))
    return out


def knowledge_text():
    """Workforce definitions for the model's prompt (no figures, no personal data)."""
    return "\n".join([
        "WORKFORCE KNOWLEDGE BASE (definitions - no figures):",
        "- Each resource file is a snapshot: the August file is the workforce as on August, the July file as on July. A month is read from its own file; compare uses the previous month's file.",
        "- Bench = resources on accounts whose name contains 'bench' (e.g. ADSPL - Bench - India). Headcount counts distinct employees; 'excl. bench' = placed on client accounts.",
        f"- Skills come from the PrimarySkills / SecondarySkills columns (comma-separated). Top-skill counts use each resource's first {TOP_SKILLS_FIRST_N} listed skills only.",
        "- You never receive employee names, numbers or e-mails. If asked for names or employee details, say they are produced by the knowledge-base reports (for example 'resources in <account>') and never invent any.",
    ])