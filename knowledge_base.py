"""
GGM knowledge base
==================
Turns the monthly reference analysis into repeatable reports.

* A recognised question (e.g. "YTD analysis", "top 5 accounts in each region",
  "accounts with GPM below 20%") is answered by CODE: every table is computed with pandas
  from the loaded data and rendered in the reference layout. No language model is involved,
  so the same question always gives the same layout and every figure is traceable to the data.
* Anything not recognised falls through to the normal model-based chat, which receives
  `knowledge_text()` (definitions and conventions only - it contains NO figures).
* `verify_figures()` checks that every decimal figure in a model answer exists in the data
  that was sent to the model.

Rules baked in (taken from the reference analysis, nothing assumed):
  - Every upload's Sheet1 holds all months from Jan to that file's latest month.
    YTD / QTD / quarter figures are computed from the LATEST file's Sheet1.
  - YTD = Jan .. as-of month.  QTD = first month of the as-of month's quarter .. as-of month.
    Quarters start in January: Q1 Jan-Mar, Q2 Apr-Jun, Q3 Jul-Sep, Q4 Oct-Dec.
  - GPM = GP / Revenue from summed values (never an average of monthly GPMs).
  - A single month uses that month's own file by default (handled by the loader).

Pure pandas + regex: no Streamlit import, so it can be unit-tested on its own.
"""

import re

import numpy as np
import pandas as pd

from local_folder_loader import MONTH_ORDER, find_months_in_text

KB_VERSION = "2026-10-07"

# ----------------------------------------------------------------------------
# Rules and thresholds. Each one comes from the reference analysis; change here only.
# ----------------------------------------------------------------------------
GPM_LOW_THRESHOLD = 20.0       # "Accounts with GPM < 20%"
GPM_FLAG_THRESHOLD = 30.0      # per-region table: "highlighted GPM <= 30%"
TOP_ACCOUNTS_N = 10            # "Top revenue accounts YTD" lists 10
TOP_PER_REGION_N = 5           # "Top 5 YTD revenue accounts in each region"
TOP_MONTH_N = 5                # single-month top accounts (reference showed 5 + SFI; here strictly top N)
TREND_MAX_ROWS = 10            # decline screen: largest revenue declines first
# The reference GPM<20% list omitted Maxicare (revenue 0.002). The reason is not stated, so NO minimum
# revenue is applied. Set e.g. 0.005 to hide tiny accounts once the rule is confirmed.
LOW_GPM_MIN_REVENUE = None
# The reference per-region table lists Digiterre and SFI as their own regions. 'source' = regions exactly
# as written in the source file (used ONLY by per-region reports). 'reporting' = after the Digiterre/SFI -> Europe rule.
REGION_BASIS = "source"
SOURCE_REGION_COL = "_Region_Source"   # internal column written before the Europe rule is applied
FLAG = "⚠️"

QUARTERS = {q: MONTH_ORDER[(q - 1) * 3:(q - 1) * 3 + 3] for q in (1, 2, 3, 4)}

# Shown in the UI so users know what the knowledge base can answer.
PLAYBOOK_GUIDE = [
    ("full_report", "Full analysis", "YTD analysis  |  QTD analysis  |  Q2 report",
     "Summary, top accounts, GPM < 20%, top 5 per region, single-month top accounts, decline screen, observations"),
    ("summary", "Summary", "YTD revenue and GPM  |  QTD summary",
     "Total revenue, total GP, overall GPM for the period"),
    ("top_accounts", "Top accounts", "Top 10 revenue accounts YTD  |  Top accounts in July",
     "Top accounts by revenue with GPM (default: selected month; add YTD / QTD / Q1-Q4)"),
    ("low_gpm", "Low-margin accounts", "Accounts with GPM below 20% YTD  |  clients with GPM < 25%",
     "Accounts under a GPM threshold (default 20%; a number in the question overrides it)"),
    ("region_top", "Top accounts per region", "Top 5 accounts in each region YTD",
     "Top N accounts per region; GPM <= 30% flagged; regions as written in the source file"),
    ("trend", "Decline trend", "Accounts showing revenue and margin decline  |  trend for Bank of America",
     "Accounts whose revenue AND GPM both fell vs the previous month, or the monthly series of a named account"),
]


# ----------------------------------------------------------------------------
# Periods
# ----------------------------------------------------------------------------
def quarter_of(month):
    return MONTH_ORDER.index(month) // 3 + 1


def make_period(kind, as_of, loaded_months, quarter=None):
    """
    kind: 'MONTH' | 'YTD' | 'QTD' | 'Q' (a named quarter).
    Returns dict(kind, label, short, months, missing, as_of): `months` are the loaded months in the
    period (calendar order); `missing` are expected months that are not loaded (never invented).
    """
    i = MONTH_ORDER.index(as_of)
    if kind == "YTD":
        expected = MONTH_ORDER[:i + 1]
    elif kind == "QTD":
        quarter = quarter_of(as_of)
        expected = [m for m in QUARTERS[quarter] if MONTH_ORDER.index(m) <= i]
    elif kind == "Q":
        expected = QUARTERS[int(quarter)]
    else:
        kind, expected = "MONTH", [as_of]
    months = [m for m in expected if m in loaded_months]
    missing = [m for m in expected if m not in loaded_months]
    span = (f"{months[0]}-{months[-1]}" if len(months) > 1 else (months[0] if months else ""))
    label = {"YTD": f"YTD ({span})", "QTD": f"QTD (Q{quarter}: {span})",
             "Q": f"Q{quarter} ({span})", "MONTH": as_of}[kind]
    short = {"YTD": "YTD", "QTD": "QTD", "Q": f"Q{quarter}", "MONTH": as_of}[kind]
    return {"kind": kind, "label": label, "short": short, "months": months, "missing": missing, "as_of": as_of}


_QTD_RE = re.compile(r"\bqtd\b|quarter[- ]to[- ]date", re.I)
_YTD_RE = re.compile(r"\bytd\b|year[- ]to[- ]date", re.I)
_Q_RE = re.compile(r"\b(?:q|quarter\s*)([1-4])\b", re.I)


def detect_period(question):
    """('QTD', None) | ('YTD', None) | ('Q', n) | None. Only explicit wording counts - nothing is guessed."""
    q = question or ""
    if _QTD_RE.search(q):
        return ("QTD", None)
    if _YTD_RE.search(q):
        return ("YTD", None)
    m = _Q_RE.search(q)
    if m:
        return ("Q", int(m.group(1)))
    return None


def resolve_period(question, selected_month, loaded_months):
    """Period for a question. YTD/QTD are 'as of' a month named in the question, else the selected month."""
    named = [m for m in find_months_in_text(question, loaded_months)]
    as_of = named[0] if named else selected_month
    det = detect_period(question)
    if det is None:
        return make_period("MONTH", as_of, loaded_months)
    kind, qn = det
    return make_period(kind, as_of, loaded_months, quarter=qn)


# ----------------------------------------------------------------------------
# Question matching
# ----------------------------------------------------------------------------
def _has_word(text, name):
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", text))


_REPORT_WORDS = re.compile(r"\b(analysis|analyses|report|review|overview|snapshot|scorecard)\b", re.I)
_REGION_EACH = re.compile(r"\b(each|every|per|by)\s+region\b|region[- ]?wise|regionwise", re.I)
_TOP_WORDS = re.compile(r"\b(top|best|largest|biggest|leading|highest)\b", re.I)
_ACCOUNT_WORDS = re.compile(r"\b(account|accounts|client|clients|customer|customers)\b", re.I)
_GPM_WORDS = re.compile(r"\b(gpm|margin|margins|gross margin)\b", re.I)
_LOW_CMP = re.compile(r"(?:<=|<|below|under|less than|lower than|beneath)\s*=?\s*(\d{1,2}(?:\.\d+)?)\s*%?", re.I)
_LOW_WORDS = re.compile(r"\b(low|poor|negative|weak|thin)\s+(gpm|margin|margins)\b", re.I)
_TREND_WORDS = re.compile(r"\b(trend|trends|declin\w*|decreas\w*|deteriorat\w*|falling|fell|dropp?\w*)\b", re.I)
_VALUE_WORDS = re.compile(r"\b(revenue|gp|gpm|margin|profit|performance|numbers|figures|summary|total)\b", re.I)


def match_playbooks(question, known=None):
    """
    Which knowledge-base reports does this question ask for? [] = none (falls through to the model).
    known = {'regions': set of lower-case region names, 'clients': set of lower-case client names}.
    A question that names a specific region or client is NOT a generic report and goes to the model,
    except the trend report for named accounts.
    """
    q = (question or "").lower()
    known = known or {}
    names_region = any(_has_word(q, r) for r in known.get("regions", ()) if r)
    named_clients = [c for c in known.get("clients", ()) if c and len(c) >= 3 and _has_word(q, c)]
    period = detect_period(question)

    if period and _REPORT_WORDS.search(q) and not names_region and not named_clients:
        return ["full_report"]
    if _TREND_WORDS.search(q) and not names_region and (named_clients or _ACCOUNT_WORDS.search(q)):
        return ["trend"]
    if names_region or named_clients:
        return []
    if _REGION_EACH.search(q) and _TOP_WORDS.search(q) and _ACCOUNT_WORDS.search(q):
        return ["region_top"]
    if _GPM_WORDS.search(q) and _ACCOUNT_WORDS.search(q) and (_LOW_CMP.search(q) or _LOW_WORDS.search(q)):
        return ["low_gpm"]
    if _TOP_WORDS.search(q) and _ACCOUNT_WORDS.search(q):
        return ["top_accounts"]
    if period and _VALUE_WORDS.search(q):
        return ["summary"]
    return []


def _top_n(question, default):
    m = re.search(r"\btop\s*(\d{1,2})\b", question or "", re.I)
    return int(m.group(1)) if m else default


# ----------------------------------------------------------------------------
# Computation helpers
# ----------------------------------------------------------------------------
def _grp(df, by, cols):
    """Sum revenue / cost / GP by `by`; GPM = GP / Revenue * 100 (NaN when revenue is 0)."""
    by = [by] if isinstance(by, str) else list(by)
    spec = {"Revenue": (cols["revenue"], "sum")}
    if cols.get("cost") and cols["cost"] in df.columns:
        spec["Cost"] = (cols["cost"], "sum")
    if cols.get("profit") and cols["profit"] in df.columns:
        spec["GP"] = (cols["profit"], "sum")
    g = df.groupby(by, dropna=True).agg(**spec).reset_index()
    if "GP" not in g.columns:
        g["GP"] = np.nan
    g["GPM"] = g["GP"] / g["Revenue"].where(g["Revenue"] != 0) * 100
    return g


def _frame(ctx, period):
    """Single month -> the default frame (own-file rule). YTD / QTD / quarter -> the LATEST file (all months)."""
    df = ctx["month_df"] if period["kind"] == "MONTH" else ctx["ytd_df"]
    return df[df[ctx["cols"]["month"]].isin(period["months"])]


def _region_col(df, ctx):
    if REGION_BASIS == "source" and SOURCE_REGION_COL in df.columns:
        return SOURCE_REGION_COL
    return ctx["cols"]["region"]


def md_table(headers, rows):
    esc = lambda v: str(v).replace("|", "\\|")
    out = ["| " + " | ".join(esc(h) for h in headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _f_rev(x):
    return f"{x:.3f}"


def _f_gpm(x, d=1):
    return "n/a" if pd.isna(x) else f"{x:.{d}f}%"


def _empty(title, why):
    return {"title": title, "md": f"**{title}**\n\n_{why}_", "df": pd.DataFrame()}


# ----------------------------------------------------------------------------
# Report sections (each returns {'title','md','df'})
# ----------------------------------------------------------------------------
def sec_summary(period, ctx):
    df = _frame(ctx, period)
    title = f"Summary - {period['label']}"
    if df.empty:
        return _empty(title, "No data loaded for this period.")
    rev = df[ctx["cols"]["revenue"]].sum()
    gp = df[ctx["cols"]["profit"]].sum() if ctx["cols"].get("profit") in df.columns else np.nan
    gpm = gp / rev * 100 if rev else np.nan
    rows = [("Total Revenue", f"{rev:.2f}M USD"), ("Total GP", "n/a" if pd.isna(gp) else f"{gp:.2f}M USD"),
            ("Overall GPM", _f_gpm(gpm, 2))]
    raw = pd.DataFrame({"Metric": [r[0] for r in rows], "Value": [r[1] for r in rows],
                        "_rev": rev, "_gp": gp, "_gpm": gpm})
    return {"title": title, "md": f"**{title}**\n\n" + md_table(["Metric", "Value"], rows), "df": raw}


def sec_top_accounts(period, ctx, n=None):
    n = n or (TOP_ACCOUNTS_N if period["kind"] != "MONTH" else TOP_MONTH_N)
    df = _frame(ctx, period)
    cl = ctx["cols"]["client"]
    title = f"Top revenue accounts - {period['label']}"
    if df.empty:
        return _empty(title, "No data loaded for this period.")
    g = _grp(df, cl, ctx["cols"])
    g = g[g["Revenue"] > 0].sort_values(["Revenue", cl], ascending=[False, True]).head(n)
    rows = [(r[cl], _f_rev(r["Revenue"]), _f_gpm(r["GPM"])) for _, r in g.iterrows()]
    return {"title": title, "md": f"**{title}** (top {len(rows)} by revenue)\n\n" +
            md_table(["Client", "Revenue USD M", "GPM"], rows), "df": g}


def sec_low_gpm(period, ctx, threshold=None):
    thr = GPM_LOW_THRESHOLD if threshold is None else threshold
    df = _frame(ctx, period)
    cl = ctx["cols"]["client"]
    title = f"Accounts with GPM < {thr:g}% - {period['label']}"
    if df.empty:
        return _empty(title, "No data loaded for this period.")
    g = _grp(df, cl, ctx["cols"])
    g = g[(g["Revenue"] > 0) & (g["GPM"] < thr)]
    if LOW_GPM_MIN_REVENUE is not None:
        g = g[g["Revenue"] >= LOW_GPM_MIN_REVENUE]
    g = g.sort_values(["GPM", cl])
    if g.empty:
        return _empty(title, f"No account with revenue > 0 has GPM below {thr:g}% in this period.")
    rows = [(r[cl], _f_rev(r["Revenue"]), _f_gpm(r["GPM"])) for _, r in g.iterrows()]
    note = ("Accounts of any size are included (no minimum-revenue rule is defined); lowest GPM first. "
            "Accounts with zero revenue have no GPM and are not listed.")
    return {"title": title, "md": f"**{title}** ({len(rows)} accounts)\n\n" +
            md_table(["Client", "Revenue USD M", "GPM"], rows) + f"\n\n_{note}_", "df": g}


def sec_region_top(period, ctx, n=None):
    n = n or TOP_PER_REGION_N
    df = _frame(ctx, period)
    cl = ctx["cols"]["client"]
    title = f"Top {n} {period['label']} revenue accounts in each region"
    if df.empty:
        return _empty(title, "No data loaded for this period.")
    rc = _region_col(df, ctx)
    g = _grp(df, [rc, cl], ctx["cols"])
    g = g[g["Revenue"] > 0]
    order = g.groupby(rc)["Revenue"].sum().sort_values(ascending=False).index.tolist()
    rows, parts = [], []
    for reg in order:
        top = g[g[rc] == reg].sort_values(["Revenue", cl], ascending=[False, True]).head(n)
        for _, r in top.iterrows():
            flag = f" {FLAG}" if pd.notna(r["GPM"]) and r["GPM"] <= GPM_FLAG_THRESHOLD else ""
            rows.append((reg, r[cl], _f_rev(r["Revenue"]), _f_gpm(r["GPM"]) + flag))
            parts.append(r)
    basis = ("Regions are shown exactly as written in the source file (Digiterre and SFI are separate regions here); "
             "regions are ordered by revenue."
             if rc == SOURCE_REGION_COL else "Regions are shown after the Digiterre/SFI -> Europe rule.")
    md = (f"**{title}** ({FLAG} = GPM <= {GPM_FLAG_THRESHOLD:g}%)\n\n" +
          md_table(["Region", "Top Account", "Revenue (USD M)", "GPM"], rows) + f"\n\n_{basis}_")
    return {"title": title, "md": md, "df": pd.DataFrame(parts)}


def sec_trend(period_month, ctx, question=""):
    """Decline screen for the as-of month vs the previous loaded month, or the monthly series of named accounts."""
    cols, cl = ctx["cols"], ctx["cols"]["client"]
    mdf = ctx["month_df"]
    loaded = [m for m in MONTH_ORDER if m in set(mdf[cols["month"]])]
    cur = period_month
    q = (question or "").lower()
    named = [c for c in ctx.get("known_clients", ()) if c and len(c) >= 3 and _has_word(q, c)]

    if named:
        title = f"Monthly trend - {', '.join(sorted(named))}"
        sub = mdf[mdf[cl].astype(str).str.lower().isin(named)]
        g = _grp(sub, [cl, cols["month"]], cols)
        if g.empty:
            return _empty(title, "The named account has no rows in the loaded data.")
        ms_all = [m for m in loaded if m in set(g[cols["month"]])]
        rows = []
        for c, gc in g.groupby(cl, sort=True):
            s_ = gc.set_index(cols["month"])
            rows.append((c, "Revenue USD M", *[_f_rev(s_.loc[m, "Revenue"]) if m in s_.index else "-" for m in ms_all]))
            rows.append((c, "GPM", *[_f_gpm(s_.loc[m, "GPM"]) if m in s_.index else "-" for m in ms_all]))
        return {"title": title, "md": f"**{title}**\n\n" + md_table(["Account", "Metric", *ms_all], rows) +
                "\n\n_'-' = the account has no row in that month._", "df": g}

    if cur not in loaded or loaded.index(cur) == 0:
        return _empty("Accounts showing revenue and margin decline", "There is no earlier month loaded to compare with.")
    prev = loaded[loaded.index(cur) - 1]
    title = f"Accounts showing revenue and margin decline - {cur} vs {prev}"
    rc = _region_col(mdf, ctx)
    a = _grp(mdf[mdf[cols["month"]] == cur], [rc, cl], cols).rename(columns={"Revenue": "R1", "GPM": "G1"})
    b = _grp(mdf[mdf[cols["month"]] == prev], [rc, cl], cols).rename(columns={"Revenue": "R0", "GPM": "G0"})
    m = a[[rc, cl, "R1", "G1"]].merge(b[[rc, cl, "R0", "G0"]], on=[rc, cl], how="inner")
    m = m[(m["R0"] > 0) & (m["R1"] > 0) & (m["R1"] < m["R0"] - 0.0005) & (m["G1"] < m["G0"] - 0.05)]
    m["Drop"] = m["R0"] - m["R1"]
    m = m.sort_values(["Drop", cl], ascending=[False, True])
    total = len(m)
    if total == 0:
        return _empty(title, "No account has both lower revenue and lower GPM than the previous month.")
    show = m.head(_top_n(question, TREND_MAX_ROWS))
    rows = [(r[rc], r[cl], f"{r['R0']:.3f}M ➜ {r['R1']:.3f}M", f"{r['G0']:.1f}% ➜ {r['G1']:.1f}%")
            for _, r in show.iterrows()]
    note = (f"Screen: revenue AND GPM both lower in {cur} than in {prev}; accounts with no revenue in either month are "
            f"not screened. Showing {len(rows)} of {total}, largest revenue decline first. Risk levels are NOT assigned: "
            f"no criteria have been defined.")
    return {"title": title, "md": f"**{title}**\n\n" + md_table(["Region", "Account", "Revenue Trend", "GPM Trend"], rows)
            + f"\n\n_{note}_", "df": m}


def observations(period, ctx):
    """Plain statements computed from the data (no model). Every number is derived in code."""
    df = _frame(ctx, period)
    cl = ctx["cols"]["client"]
    if df.empty:
        return []
    g = _grp(df, cl, ctx["cols"])
    total = g["Revenue"].sum()
    top = g[g["Revenue"] > 0].sort_values(["Revenue", cl], ascending=[False, True])
    out = []
    if len(top) and total:
        t = top.iloc[0]
        out.append(f"{t[cl]} is the largest revenue account in {period['label']}: {t['Revenue']:.3f} USD M, "
                   f"{t['Revenue'] / total * 100:.1f}% of total revenue ({total:.3f} USD M).")
    if len(top) >= 3 and total:
        t3 = top.head(3)
        out.append(f"The top 3 accounts ({', '.join(t3[cl])}) together contribute {t3['Revenue'].sum() / total * 100:.1f}% "
                   f"of {period['label']} revenue.")
    flagged = top.head(TOP_ACCOUNTS_N)
    flagged = flagged[flagged["GPM"] <= GPM_FLAG_THRESHOLD]
    if len(flagged):
        out.append(f"Among the top {min(TOP_ACCOUNTS_N, len(top))} accounts, GPM is at or below {GPM_FLAG_THRESHOLD:g}% for: " +
                   ", ".join(f"{r[cl]} ({r['GPM']:.1f}%)" for _, r in flagged.iterrows()) + ".")
    low = g[(g["Revenue"] > 0) & (g["GPM"] < GPM_LOW_THRESHOLD)]
    if LOW_GPM_MIN_REVENUE is not None:
        low = low[low["Revenue"] >= LOW_GPM_MIN_REVENUE]
    if len(low) and total:
        out.append(f"{len(low)} accounts have {period['label']} GPM below {GPM_LOW_THRESHOLD:g}%, with combined revenue of "
                   f"{low['Revenue'].sum():.3f} USD M ({low['Revenue'].sum() / total * 100:.1f}% of total revenue).")
    rc = _region_col(df, ctx)
    if rc in df.columns and total:
        rg = _grp(df, rc, ctx["cols"]).sort_values("Revenue", ascending=False)
        if len(rg):
            r0 = rg.iloc[0]
            out.append(f"{r0[rc]} is the largest region in {period['label']}: {r0['Revenue']:.3f} USD M "
                       f"({r0['Revenue'] / total * 100:.1f}% of total revenue).")
    return out


def data_notes(periods, ctx):
    """Where the numbers came from, and any month whose numbers differ between files (both shown)."""
    lines = []
    ytd_file = ctx.get("ytd_file")
    if any(p["kind"] != "MONTH" for p in periods):
        p = next(p for p in periods if p["kind"] != "MONTH")
        lines.append(f"{p['label']} is computed from the latest file's Sheet1 ({ytd_file or 'n/a'}), months {', '.join(p['months'])}.")
    mf = ctx.get("month_files", {})
    for p in periods:
        if p["kind"] == "MONTH" and p["as_of"] in mf:
            lines.append(f"{p['as_of']} figures come from {mf[p['as_of']]}.")
    for p in periods:
        if p["missing"]:
            lines.append(f"Months not loaded, so NOT included in {p['short']}: {', '.join(p['missing'])}.")
    mm = ctx.get("mismatch")
    if mm is not None and len(mm):
        relevant = set()
        for p in periods:
            relevant |= set(p["months"]) | {p["as_of"]}
        for _, r in mm[mm["Month"].isin(relevant)].iterrows():
            lines.append(f"{r['Month']}: numbers differ between files - {r['Default_Source']} {r['Revenue_Default']:.3f} vs "
                         f"{r['Other_Source']} {r['Revenue_Other']:.3f} (revenue, USD M).")
    return lines


def build_report(ids, question, ctx):
    """Render the requested knowledge-base reports. Returns {'markdown', 'sections', 'ids'}."""
    selected = ctx["selected_month"]
    loaded = ctx["loaded_months"]
    base = resolve_period(question, selected, loaded)
    month_p = make_period("MONTH", base["as_of"], loaded)
    sections, periods, header = [], [base], None

    if "full_report" in ids:
        ytd = base if base["kind"] != "MONTH" else make_period("YTD", base["as_of"], loaded)
        periods = [ytd, month_p]
        header = f"**Analysis on GGM {ytd['label']} and {month_p['as_of']} revenue and GPM**"
        sections = [sec_summary(ytd, ctx), sec_top_accounts(ytd, ctx), sec_low_gpm(ytd, ctx), sec_region_top(ytd, ctx),
                    sec_top_accounts(month_p, ctx), sec_trend(base["as_of"], ctx, question)]
        obs = observations(ytd, ctx)
    else:
        obs = []
        for i in ids:
            if i == "summary":
                sections.append(sec_summary(base, ctx))
            elif i == "top_accounts":
                sections.append(sec_top_accounts(base, ctx, _top_n(question, None)))
            elif i == "low_gpm":
                m = _LOW_CMP.search(question or "")
                sections.append(sec_low_gpm(base, ctx, float(m.group(1)) if m else None))
            elif i == "region_top":
                sections.append(sec_region_top(base, ctx, _top_n(question, None)))
            elif i == "trend":
                sections.append(sec_trend(base["as_of"], ctx, question))

    md = [header] if header else []
    md += [s["md"] for s in sections]
    if obs:
        md.append("**Observations** _(computed from the tables above)_\n\n" + "\n".join(f"- {o}" for o in obs))
    notes = data_notes(periods, ctx)
    if notes:
        md.append("**Data basis**\n\n" + "\n".join(f"- {n}" for n in notes))
    return {"markdown": "\n\n".join(md), "sections": sections, "ids": ids, "observations": obs, "notes": notes}


# ----------------------------------------------------------------------------
# For the model-based chat
# ----------------------------------------------------------------------------
def knowledge_text():
    """Definitions and conventions for the system prompt. Contains NO figures, on purpose."""
    return "\n".join([
        "KNOWLEDGE BASE (definitions and conventions - contain no figures):",
        "- Each uploaded file's Sheet1 holds every month from Jan up to that file's latest month.",
        "- YTD = Jan to the selected month. QTD = first month of the selected month's quarter to the selected month. "
        "Quarters start in January: Q1 Jan-Mar, Q2 Apr-Jun, Q3 Jul-Sep, Q4 Oct-Dec.",
        "- YTD / QTD / quarter figures come from the LATEST file's Sheet1 (all months) - see TABLE P1-P3 when present.",
        "- GPM = GP / Revenue from summed values, never an average of monthly GPMs.",
        f"- Low-margin accounts are those with GPM below {GPM_LOW_THRESHOLD:g}%; in per-region tables GPM at or below "
        f"{GPM_FLAG_THRESHOLD:g}% is flagged.",
        "- Per-region reports use regions as written in the source file; every other answer reports Digiterre and SFI under Europe.",
        "- Risk levels (Critical / High / Medium) are NOT defined: never assign them.",
    ])


_DEC = re.compile(r"(?<![\w.])(-?\d+\.\d+)(?![\w.]*\d)")
_ANY_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def verify_figures(answer, evidence):
    """
    Decimal figures in `answer` that do not appear in `evidence` (the data sent to the model), compared at the
    precision they were quoted with (17.70 matches 17.696). Integers / years are ignored. Returns a list of strings.
    """
    ev = [float(x) for x in _ANY_NUM.findall(evidence or "")]
    bad = []
    for m in _DEC.finditer(answer or ""):
        tok = m.group(1)
        dec = len(tok.split(".")[1])
        v = abs(float(tok))
        tol = 0.5 * 10 ** (-dec) + 1e-9
        if not any(abs(v - abs(e)) <= tol for e in ev):
            bad.append(tok)
    return list(dict.fromkeys(bad))