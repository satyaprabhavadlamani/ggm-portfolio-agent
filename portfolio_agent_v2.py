import difflib
import os
import re
import traceback

# Corporate networks often re-sign HTTPS traffic with a company certificate that Python
# doesn't trust but Windows does. truststore makes Python use the OS trust store.
# Certificate verification stays ON. Must run before any HTTP client is created.
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass  # not installed / unsupported platform: fall back to Python's default certificates

import numpy as np
import pandas as pd
import streamlit as st

# Fuzzy matching: rapidfuzz (maintained, prebuilt wheels) -> fuzzywuzzy -> difflib (always available)
try:
    from rapidfuzz import fuzz as _fuzz
except ImportError:
    try:
        from fuzzywuzzy import fuzz as _fuzz
    except ImportError:
        _fuzz = None

from local_folder_loader import (
    SOURCE_COL,
    YEAR_COL,
    apply_circle_filter,
    assign_months,
    find_months_in_text,
    get_available_months,
    infer_month_from_text,
    infer_year_from_text,
    load_monthly_data_from_folder,
    month_mismatches,
    select_authoritative,
)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
CIRCLE_NAME = "Data and Insights"      # value of the 'Circle' column to keep
YEAR = "2026"
DEFAULT_BASE_PATH = r"C:\Users\satyaprabha.v\OneDrive - ascendion\Documents\GGM Data\GGM Portfolio Analytics"
DEFAULT_MODELS = ["llama-3.1-8b-instant", "openai/gpt-oss-20b"]  # verify against Groq's current model list
TOP_N = 20          # max rows per table sent to the model (keeps prompts within Groq token limits)
HISTORY_TURNS = 6   # recent chat messages sent to the model so follow-ups keep context
REGION_CLIENT_CAP = 40      # max clients listed per region in TABLE 4B (also scaled down with top_n)
MAX_OUTPUT_TOKENS = 1024    # max_tokens requested from the model
# Total tokens (prompt + reply) one request may use. ASSUMPTION - not verified against Groq's current
# per-model limits. Override with [groq] token_budget in secrets or the GROQ_TOKEN_BUDGET env var.
MODEL_TOKEN_BUDGET = 6000
CHARS_PER_TOKEN = 3.0       # conservative estimate for number-heavy CSV text
YEARS = "2026"              # year folder(s) to load: "2026", "2025,2026" or "all" (override: [paths] years / GGM_YEARS)

SYNONYM_MAP = {
    'revenue': ['sales', 'top line', 'topline', 'income', 'earnings', 'throughput', 'billing', 'invoiced', 'turnover', 'receipts'],
    'profit': ['gp', 'gross profit', 'margin', 'bottomline', 'earnings', 'gains', 'net', 'returns', 'net profit'],
    'gpm': ['margin', 'profitability', 'margin %', 'efficiency', 'yield', 'margin percentage', 'gross margin'],
    'headcount': ['staff', 'team', 'people', 'employees', 'strength', 'workforce', 'fte', 'personnel', 'manpower', 'bench'],
    'account': ['client', 'customer', 'partner', 'engagement', 'company', 'vendor', 'portfolio', 'business'],
    'region': ['geography', 'location', 'zone', 'area', 'market', 'territory', 'geographic'],
    'cost': ['expense', 'cost of sales', 'cogs', 'operational cost', 'spending', 'burn'],
    'performance': ['how is', 'status', 'doing', 'outlook', 'trajectory', 'trend', 'performance'],
}

CLIENT_EXACT = ['client name', 'client', 'customer', 'account name', 'account']


# ----------------------------------------------------------------------------
# Helpers (no Streamlit UI calls below this line until the UI section)
# ----------------------------------------------------------------------------
def get_secret(section, key, default=None):
    """st.secrets raises if no secrets file exists; treat that as 'not set'."""
    try:
        return st.secrets.get(section, {}).get(key, default)
    except Exception:
        return default


def _find_col(df, exact=(), contains=(), exclude=()):
    """Exact (case-insensitive) name match first, then substring match."""
    cols = [str(c) for c in df.columns]
    for c in cols:
        if c.lower().strip() in exact:
            return c
    for c in cols:
        low = c.lower()
        if any(x in low for x in exclude):
            continue
        if any(t in low for t in contains):
            return c
    return None


def detect_column_mapping(df_cir, df_res=None):
    column_map = {
        'revenue': _find_col(df_cir, exact=['revenue usd m', 'revenue'],
                             contains=['revenue', 'sales', 'topline', 'billing'], exclude=['%']),
        'cost': _find_col(df_cir, exact=['cost usd m', 'cost'],
                          contains=['cost', 'cogs', 'expense'], exclude=['%']),
        'profit': _find_col(df_cir, exact=['gp usd m', 'gross profit'],
                            contains=['gross profit', 'gp ', 'profit'], exclude=['%']),
        'gpm': _find_col(df_cir, exact=['gpm %', 'gpm', 'gpm%'],
                         contains=['gpm', 'margin', 'profitability']),
        'client': _find_col(df_cir, exact=CLIENT_EXACT,
                            contains=['client', 'customer', 'account', 'company']),
        'region': _find_col(df_cir, exact=['region'],
                            contains=['region', 'geography', 'territory', 'location']),
        'res_client': None,
        'res_region': None,
        'res_emp': None,
    }

    if df_res is not None and len(df_res) > 0:
        column_map['res_client'] = _find_col(df_res, exact=CLIENT_EXACT,
                                             contains=['client', 'customer', 'account'])
        column_map['res_region'] = _find_col(df_res, exact=['region'],
                                             contains=['region', 'geography', 'location'])
        column_map['res_emp'] = _find_col(df_res, exact=['emp no', 'employee id', 'emp id', 'resource_id'],
                                          contains=['emp no', 'employee id'])
    return column_map


def filter_circle(df):
    """Keep only the Data and Insights circle when the file has a 'Circle' column.
    Returns (df, applied: bool)."""
    if df is not None and 'Circle' in df.columns:
        mask = df['Circle'].astype(str).str.strip() == CIRCLE_NAME
        if mask.any():
            return df[mask].copy(), True
    return df, False


def clean_circle_df(df, column_map):
    """Coerce money columns to numbers and drop 'Total' summary rows (they double-count).
    Returns (df, dropped_row_count)."""
    df = df.copy()
    for key in ('revenue', 'cost', 'profit', 'gpm'):
        col = column_map.get(key)
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    dropped = 0
    client = column_map.get('client')
    if client and client in df.columns:
        mask = df[client].astype(str).str.match(r'^\s*(grand\s+)?(sub\s*)?total\s*$', case=False)
        dropped = int(mask.sum())
        df = df[~mask]
    return df, dropped


def normalize_labels(df, columns):
    """
    Clean text labels (client / region) BEFORE any grouping, so 'Europe' and 'Europe ' (or
    'Bank of America' / 'Bank Of America') are one group, not two.
      - trims, collapses repeated spaces, turns non-breaking spaces into spaces
      - merges case variants into the most frequent spelling (ties: alphabetical)
      - blank strings become missing values
    Returns (df, merged) where merged = {column: number of spelling variants merged away}.
    """
    merged = {}
    if df is None or df.empty:
        return df, merged
    df = df.copy()
    for col in columns:
        if not col or col not in df.columns:
            continue
        s = df[col]
        cleaned = (s.astype(str)
                    .str.replace('\u00a0', ' ', regex=False)
                    .str.replace(r'\s+', ' ', regex=True)
                    .str.strip())
        cleaned = cleaned.where(s.notna() & (cleaned != ''), np.nan)
        valid = cleaned.notna()
        if not valid.any():
            continue
        key = cleaned.str.lower()
        pairs = pd.DataFrame({'k': key[valid], 'v': cleaned[valid]})

        def _most_common(x):
            vc = x.value_counts()
            return vc[vc == vc.max()].index.min()

        canon = pairs.groupby('k')['v'].agg(_most_common)
        before = int(s.dropna().astype(str).nunique())
        df[col] = key.map(canon)
        after = int(df[col].nunique())
        if before > after:
            merged[col] = before - after
    return df, merged


def get_column_explanations(column_map):
    """Generate AI-readable explanations of detected columns"""
    explanations = []

    if column_map.get('revenue'):
        explanations.append(f"Revenue Column: '{column_map['revenue']}' - Total sales/income (VPs may say: sales, topline, billing)")
    if column_map.get('cost'):
        explanations.append(f"Cost Column: '{column_map['cost']}' - Operating costs/expenses (VPs may say: expense, COGS)")
    if column_map.get('profit'):
        explanations.append(f"Profit Column: '{column_map['profit']}' - Gross profit (VPs may say: GP, earnings, net profit)")
    if column_map.get('gpm'):
        explanations.append(f"Margin Column: '{column_map['gpm']}' - Profitability % (VPs may say: margin, efficiency)")
    if column_map.get('client'):
        explanations.append(f"Client Column: '{column_map['client']}' - Account/customer names (VPs may say: account, customer, company)")
    if column_map.get('region'):
        explanations.append(f"Region Column: '{column_map['region']}' - Geographic location (VPs may say: geography, territory)")
    if column_map.get('res_emp'):
        explanations.append(f"Headcount (resource file) employee ID: '{column_map['res_emp']}'")
    if column_map.get('res_client'):
        explanations.append(f"Headcount (resource file) client: '{column_map['res_client']}'")
    if column_map.get('res_region'):
        explanations.append(f"Headcount (resource file) region: '{column_map['res_region']}'")

    return "\n".join(f"- {e}" for e in explanations) if explanations else ""


# 'bench' as a whole word only - so "Bench", "Internal Bench", "Bench-Pool" match but "Benchmark Corp" does not
_BENCH_RE = re.compile(r'(?<![a-z])bench(?![a-z])', re.IGNORECASE)


def identify_bench_resources(df_res, column_map):
    """
    Identify bench resources: where client/account field contains the whole word 'bench' (case-insensitive).
    Bench resources are allocated to internal bench, not active client accounts.
    Returns (active_res_df, bench_res_df) - two dataframes separated by bench status.
    """
    if df_res is None or df_res.empty:
        return df_res, pd.DataFrame()
    
    client_col = column_map.get('res_client')
    if not client_col or client_col not in df_res.columns:
        return df_res, pd.DataFrame()
    
    df_res = df_res.copy()
    is_bench = df_res[client_col].astype(str).str.contains(_BENCH_RE, na=False)
    
    return df_res[~is_bench].copy(), df_res[is_bench].copy()


_LEGAL_SUFFIXES = {
    'the', 'ltd', 'limited', 'inc', 'incorporated', 'llc', 'llp', 'lp', 'plc', 'corp', 'corporation',
    'co', 'company', 'gmbh', 'ag', 'sa', 'nv', 'bv', 'pvt', 'private',
}


def _name_key(name):
    """Exact-match key: whitespace-collapsed, lower-case."""
    return re.sub(r'\s+', ' ', str(name)).strip().lower()


def _name_tokens(name):
    """Fuzzy-match form: lower-case, punctuation removed, legal suffixes (Ltd, Inc, ...) dropped."""
    t = re.sub(r'[^a-z0-9]+', ' ', str(name).lower().replace('&', ' and '))
    return ' '.join(w for w in t.split() if w not in _LEGAL_SUFFIXES)


def _token_sort_score(a, b):
    """Order-insensitive similarity 0-100. Extra words LOWER the score ('AXA' vs 'AXA XL' ~ 67)."""
    if _fuzz is not None:
        return _fuzz.token_sort_ratio(a, b)
    ta, tb = ' '.join(sorted(a.split())), ' '.join(sorted(b.split()))
    return int(round(difflib.SequenceMatcher(None, ta, tb).ratio() * 100))


def _token_set_score(a, b):
    """Lenient score: 100 when one name's words are a subset of the other's. Used ONLY to suggest reviews."""
    if _fuzz is not None:
        return _fuzz.token_set_ratio(a, b)
    sa, sb = set(a.split()), set(b.split())
    if sa and sb and (sa <= sb or sb <= sa):
        return 100
    return _token_sort_score(a, b)


def validate_client_matching(df_cir, df_res, column_map, log=None, fuzzy_threshold=85, review_threshold=90):
    """
    Validate that client names in the revenue file match the resource file.
    Returns (matched_clients, revenue_only, resource_only) for data quality checks.

    1. Exact match (case/whitespace-insensitive).
    2. Fuzzy auto-match: token_sort_ratio >= fuzzy_threshold on cleaned names, BEST pair first,
       one-to-one, deterministic. Subset names ("AXA" vs "AXA XL") do NOT auto-match.
    3. Lenient candidates (token_set_ratio >= review_threshold) are only SUGGESTED in the log.
    Bench rows are not clients and are excluded from the resource side.
    """
    log = log if log is not None else []

    # Handle None or empty dataframes
    if df_cir is None or df_res is None or len(df_cir) == 0 or len(df_res) == 0:
        return set(), set(), set()

    client_cir_col = column_map.get('client')
    client_res_col = column_map.get('res_client')

    if not client_cir_col or not client_res_col:
        return set(), set(), set()

    if client_cir_col not in df_cir.columns or client_res_col not in df_res.columns:
        return set(), set(), set()

    df_res_active, _bench = identify_bench_resources(df_res, column_map)

    def _by_key(series):
        out = {}
        for v in series.dropna().astype(str).str.strip().unique():
            if v and v.lower() != 'nan':
                out.setdefault(_name_key(v), v)
        return out

    rev_by_key = _by_key(df_cir[client_cir_col])
    res_by_key = _by_key(df_res_active[client_res_col])

    # 1. Exact matches
    exact = set(rev_by_key) & set(res_by_key)
    matched = {rev_by_key[k] for k in exact}
    revenue_unmatched = {v for k, v in rev_by_key.items() if k not in exact}
    resource_unmatched = {v for k, v in res_by_key.items() if k not in exact}

    # 2. Fuzzy auto-match: score every pair, take best first, each name used at most once
    rev_tok = {c: _name_tokens(c) for c in revenue_unmatched}
    res_tok = {c: _name_tokens(c) for c in resource_unmatched}
    pairs = []
    for rc in sorted(revenue_unmatched):
        for sc in sorted(resource_unmatched):
            if rev_tok[rc] and res_tok[sc]:
                ta, tb = set(rev_tok[rc].split()), set(res_tok[sc].split())
                if ta < tb or tb < ta:      # one name just has EXTRA words ('Santander' / 'Santander UK'):
                    continue                # possibly a different entity -> suggested for review, never auto-matched
                pairs.append((_token_sort_score(rev_tok[rc], res_tok[sc]), rc, sc))
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))

    fuzzy_matched = {}
    used_res = set()
    for score, rc, sc in pairs:
        if score < fuzzy_threshold:
            break
        if rc in fuzzy_matched or sc in used_res:
            continue
        fuzzy_matched[rc] = (sc, score)
        used_res.add(sc)

    for rc, (sc, _score) in fuzzy_matched.items():
        revenue_unmatched.discard(rc)
        resource_unmatched.discard(sc)
        matched.add(rc)

    # 3. Suggest (never auto-apply) lenient candidates among what is still unmatched
    review = {}
    for rc in sorted(revenue_unmatched):
        best = None
        for sc in sorted(resource_unmatched):
            if rev_tok[rc] and res_tok[sc]:
                sc_score = _token_set_score(rev_tok[rc], res_tok[sc])
                if sc_score >= review_threshold and (best is None or sc_score > best[1]):
                    best = (sc, sc_score)
        if best:
            review[rc] = best

    if fuzzy_matched:
        items = sorted(fuzzy_matched.items())
        matches_str = ", ".join([f"'{k}' ≈ '{v[0]}' ({v[1]:.0f}%)" for k, v in items[:3]])
        log.append(f"✓ Fuzzy matched (≥{fuzzy_threshold}% name similarity): {matches_str}" +
                   (f" (+{len(items)-3} more)" if len(items) > 3 else ""))
    if review:
        items = sorted(review.items())
        review_str = ", ".join([f"'{k}' ? '{v[0]}'" for k, v in items[:5]])
        log.append(f"🔎 Possible matches to review (NOT auto-matched): {review_str}" +
                   (f" (+{len(items)-5} more)" if len(items) > 5 else ""))
    if revenue_unmatched:
        log.append(f"⚠️ In revenue but not in resource: {', '.join(sorted(revenue_unmatched)[:5])}" +
                   (f" (+{len(revenue_unmatched)-5} more)" if len(revenue_unmatched) > 5 else ""))
    if resource_unmatched:
        log.append(f"⚠️ In resource but not in revenue: {', '.join(sorted(resource_unmatched)[:5])}" +
                   (f" (+{len(resource_unmatched)-5} more)" if len(resource_unmatched) > 5 else ""))

    return matched, revenue_unmatched, resource_unmatched


def apply_region_overrides(df, column_map, data_type='cir'):
    """
    Apply hardcoded region mappings for specific clients AND regions.
    - Digiterre (client OR region) → Europe
    - SFI (client OR region) → Europe
    Ignores the source region/country and always uses the override.
    
    Args:
        df: DataFrame (Circle Wise or Resource)
        column_map: Column name mapping
        data_type: 'cir' for Circle Wise, 'res' for Resource data
    
    Returns:
        DataFrame with region overrides applied
    """
    if df is None or df.empty:
        return df
    
    # Determine which columns to use based on data type
    if data_type == 'cir':
        client_col = column_map.get('client')
        region_col = column_map.get('region')
    else:  # data_type == 'res'
        client_col = column_map.get('res_client')
        region_col = column_map.get('res_region')
    
    if not client_col or not region_col:
        return df
    
    if client_col not in df.columns or region_col not in df.columns:
        return df
    
    df = df.copy()

    # Apply region overrides for specific clients
    # CRITICAL: These MUST always map to Europe regardless of source data
    overrides = {
        'Digiterre': 'Europe',
        'SFI': 'Europe',
    }
    
    for override_name, target_region in overrides.items():
        # Match in CLIENT column (handles Digiterre/SFI as client names)
        client_mask = df[client_col].astype(str).str.strip().str.lower() == override_name.lower()
        if client_mask.any():
            df.loc[client_mask, region_col] = target_region
        
        # ALSO match in REGION column (handles Digiterre/SFI as region values in source data)
        region_mask = df[region_col].astype(str).str.strip().str.lower() == override_name.lower()
        if region_mask.any():
            df.loc[region_mask, region_col] = target_region
    
    return df


def get_available_metrics(df_cir, df_res, column_map):
    """Headline numbers for the selected month. df_cir / df_res must already be month-filtered."""
    def total(key):
        col = column_map.get(key)
        return float(df_cir[col].sum()) if col and col in df_cir.columns else 0.0

    metrics = {'revenue': total('revenue'), 'cost': total('cost'), 'profit': total('profit')}
    metrics['gpm'] = (metrics['profit'] / metrics['revenue'] * 100) if metrics['revenue'] else 0.0

    # Separate active resources from bench resources
    df_res_active, df_res_bench = identify_bench_resources(df_res, column_map)
    
    headcount_active = 0
    headcount_bench = 0
    if df_res_active is not None and len(df_res_active) > 0:
        emp = column_map.get('res_emp')
        headcount_active = int(df_res_active[emp].nunique()) if emp and emp in df_res_active.columns else len(df_res_active)
    if df_res_bench is not None and len(df_res_bench) > 0:
        emp = column_map.get('res_emp')
        headcount_bench = int(df_res_bench[emp].nunique()) if emp and emp in df_res_bench.columns else len(df_res_bench)
    
    metrics['headcount'] = headcount_active
    metrics['headcount_bench'] = headcount_bench

    client = column_map.get('client')
    metrics['accounts'] = int(df_cir[client].nunique()) if client and client in df_cir.columns else 0
    return metrics


# ---- Pre-computed tables: the model reads these, it does not do the maths ----
def _agg(df, by, column_map):
    spec = {}
    for key, label in (('revenue', 'Revenue_USDm'), ('cost', 'Cost_USDm'), ('profit', 'GP_USDm')):
        col = column_map.get(key)
        if col and col in df.columns:
            spec[label] = (col, 'sum')
    if not spec or df.empty:
        return pd.DataFrame()
    out = df.groupby(by, dropna=True).agg(**spec).reset_index()
    if 'Revenue_USDm' in out.columns and 'GP_USDm' in out.columns:
        out['GPM_pct'] = np.where(out['Revenue_USDm'] != 0,
                                  out['GP_USDm'] / out['Revenue_USDm'] * 100, np.nan)
    return out


def _csv(df):
    return df.round(3).to_csv(index=False).strip()


def _hc(df, emp):
    """Distinct employees (or row count when there is no employee-id column)."""
    if df is None or len(df) == 0:
        return 0
    return int(df[emp].nunique()) if emp and emp in df.columns else len(df)


def _tables_client_region(cur, month, column_map, top_n):
    """TABLE 2 (by client), 3 (by region) and 4B (client-by-region with revenue) for ONE month.
    Returns {'2': text, '3': text, '4B': text} - only the ones that can be built."""
    parts = {}
    client, region = column_map.get('client'), column_map.get('region')

    for label, col in (('CLIENT', client), ('REGION', region)):
        if not col or col not in cur.columns:
            continue
        t = _agg(cur, col, column_map)
        if t.empty:
            continue
        t = t.sort_values('Revenue_USDm' if 'Revenue_USDm' in t.columns else t.columns[1], ascending=False)
        limit = top_n if label == 'CLIENT' else max(top_n, 30)   # regions are few: effectively never cut
        scope = f"top {limit} of {len(t)}" if len(t) > limit else f"all {len(t)}"
        n = 2 if label == 'CLIENT' else 3
        parts[str(n)] = f"TABLE {n} - {month} BY {label} ({scope}, sorted by revenue)\n" + _csv(t.head(limit))

    # 4B. Client-by-region WITH revenue/GP/GPM, so "revenue of the accounts in Europe" is a lookup, not maths
    if client and region and client in cur.columns and region in cur.columns:
        rc = _agg(cur, [region, client], column_map)
        if not rc.empty and 'Revenue_USDm' in rc.columns:
            cap = min(REGION_CLIENT_CAP, top_n * 2)
            rc = rc.sort_values([region, 'Revenue_USDm'], ascending=[True, False])
            shown = rc.groupby(region, sort=False).head(cap)
            scope = (f"all {len(rc)} rows" if len(shown) == len(rc)
                     else f"top {cap} clients per region - {len(shown)} of {len(rc)} rows shown, list is truncated")
            parts['4B'] = (f"TABLE 4B - {month} CLIENT-BY-REGION: every client's revenue/GP/GPM inside its region ({scope}). "
                           f"Use it to list a region's accounts and their revenue; region totals are in TABLE 3\n" + _csv(shown))
        elif rc.empty:
            m = cur[[client, region]].drop_duplicates().sort_values([region, client])
            if not m.empty:
                parts['4B'] = f"TABLE 4B - {month} CLIENT-TO-REGION MAPPING (no revenue column detected)\n" + _csv(m)
    return parts


def _tables_headcount_month(res_cur, month, column_map, top_n):
    """TABLE 6 (active headcount by client) and 7 (by region, bench shown separately) for ONE month."""
    parts = {}
    if res_cur is None or len(res_cur) == 0:
        return parts
    emp = column_map.get('res_emp')
    emp = emp if emp in res_cur.columns else None
    active, bench = identify_bench_resources(res_cur, column_map)

    def _grouped(frame, col):
        if frame is None or len(frame) == 0 or col not in frame.columns:
            return pd.Series(dtype='int64')
        return frame.groupby(col)[emp].nunique() if emp else frame.groupby(col).size()

    col = column_map.get('res_client')
    if col and col in res_cur.columns:
        g = _grouped(active, col)
        if len(g):
            g = g.rename('Headcount').reset_index().sort_values('Headcount', ascending=False)
            scope = f"top {top_n} of {len(g)}" if len(g) > top_n else f"all {len(g)}"
            parts['6'] = f"TABLE 6 - {month} ACTIVE HEADCOUNT BY CLIENT ({scope}; bench excluded)\n" + _csv(g.head(top_n))

    col = column_map.get('res_region')
    if col and col in res_cur.columns:
        g = pd.DataFrame({'Active_Headcount': _grouped(active, col),
                          'Bench_Headcount': _grouped(bench, col)}).fillna(0).astype(int)
        if len(g):
            g.index.name = col
            g = g.reset_index().sort_values('Active_Headcount', ascending=False)
            scope = f"top {top_n} of {len(g)}" if len(g) > top_n else f"all {len(g)}"
            parts['7'] = (f"TABLE 7 - {month} HEADCOUNT BY REGION ({scope}; Active_Headcount excludes bench, "
                          f"Bench_Headcount is separate)\n" + _csv(g.head(top_n)))
    return parts


def _mismatch_parts(mismatches, client_cap=15):
    """TABLE M / M2: months whose numbers differ between files - BOTH values side by side."""
    if not mismatches:
        return []
    mm_month, mm_client = mismatches
    if mm_month is None or len(mm_month) == 0:
        return []
    cols = ['Month', 'Default_Source', 'Other_Source'] + [c for c in mm_month.columns
                                                          if c.split('_')[0] in ('Revenue', 'Cost', 'GP')]
    out = ["TABLE M - MONTHS WHOSE NUMBERS DIFFER BETWEEN FILES (USD m). Default_* = the file that reports that "
           "month (used by default); Other_* = the same month as found in another file. Show BOTH when comparing; "
           "never blend\n" + _csv(mm_month[cols])]
    if mm_client is not None and len(mm_client):
        out.append("TABLE M2 - CLIENTS BEHIND THE DIFFERENCES IN TABLE M (largest first)\n" + _csv(mm_client.head(client_cap)))
    return out


_MULTI_RE = re.compile(
    r'\b(overall|ytd|year[- ]to[- ]date|all months|cumulative|trend|so far|compar\w*|vs\.?|versus|chang\w*|growth|'
    r'previous month|last month|prior month|month[- ]over[- ]month|mom|each month|by month|monthly|across months|history|since)\b',
    re.IGNORECASE)


def wants_multi_month(question, extra_months=None):
    """True when the question needs more than the selected month: it names another month, compares, or asks overall/YTD."""
    return bool(extra_months) or bool(_MULTI_RE.search(question or ''))


def describe_sources(df):
    """'Jan-Jul: fileA / Aug: fileB' - which file is the default for each month (runs of consecutive months)."""
    if df is None or df.empty or SOURCE_COL not in df.columns or 'Month' not in df.columns:
        return ""
    by_month = df.groupby('Month')[SOURCE_COL].first()
    runs = []
    for m in get_available_months(df):
        f = by_month.get(m)
        if runs and runs[-1][1] == f:
            runs[-1][2] = m
        else:
            runs.append([m, f, m])
    return "\n".join(f"- {a if a == b else a + '-' + b}: {f}" for a, f, b in runs)


def build_data_tables(df_cir_raw, df_res_raw, selected_month, column_map, top_n=TOP_N, extra_months=None,
                      mismatches=None, multi_month=False):
    """All figures the model is allowed to quote, computed with pandas.
    extra_months: other loaded months named in the user's question - they get their own
    client / region / client-by-region / headcount tables so 'region revenue in July' is answerable."""
    parts = []
    months = get_available_months(df_cir_raw)
    cur = df_cir_raw[df_cir_raw['Month'] == selected_month]
    client, rev = column_map.get('client'), column_map.get('revenue')

    # 1. Totals by month
    trend = _agg(df_cir_raw, 'Month', column_map)
    if not trend.empty:
        trend['Month'] = pd.Categorical(trend['Month'], categories=months, ordered=True)
        trend = trend.sort_values('Month')
        trend_out = trend.copy()
        trend_out['Month'] = trend_out['Month'].astype(str)
        note = ""
        if len(months) > 1:
            tot = {'Month': 'ALL_MONTHS'}
            for c in ('Revenue_USDm', 'Cost_USDm', 'GP_USDm'):
                if c in trend.columns:
                    tot[c] = trend[c].sum()
            if 'Revenue_USDm' in tot and 'GP_USDm' in tot:
                tot['GPM_pct'] = tot['GP_USDm'] / tot['Revenue_USDm'] * 100 if tot['Revenue_USDm'] else np.nan
            trend_out = pd.concat([trend_out, pd.DataFrame([tot])], ignore_index=True)
            note = "; last row ALL_MONTHS = all loaded months combined"
        parts.append(f"TABLE 1 - TOTALS BY MONTH (each month from its own default file{note}; GPM_pct = GP / Revenue)\n"
                     + _csv(trend_out))
    parts += _mismatch_parts(mismatches)

    # 2, 3 & 4B. Selected month by client / region / client-by-region
    main = _tables_client_region(cur, selected_month, column_map, top_n)
    parts += [main[k] for k in ('2', '3') if k in main]

    # 4. Revenue by client across months, with change vs previous month
    if client and rev and client in df_cir_raw.columns and len(months) > 1:
        pv = df_cir_raw.pivot_table(index=client, columns='Month', values=rev, aggfunc='sum')
        pv = pv.reindex(columns=[m for m in months if m in pv.columns])
        if selected_month in pv.columns:
            idx = months.index(selected_month)
            if idx > 0 and months[idx - 1] in pv.columns:
                prev = months[idx - 1]
                delta = pv[selected_month] - pv[prev]
                pv[f'Chg_{selected_month}_vs_{prev}_USDm'] = delta
                pv['Chg_pct'] = np.where(pv[prev].fillna(0) != 0, delta / pv[prev] * 100, np.nan)
            pv['Total_all_months'] = pv[[m for m in months if m in pv.columns]].sum(axis=1)
            pv = pv.sort_values(selected_month, ascending=False).head(top_n).reset_index()
            parts.append(f"TABLE 4 - REVENUE USD m BY CLIENT ACROSS MONTHS (top {top_n} by {selected_month}; "
                         f"each month from its own default file)\n" + _csv(pv))

    if '4B' in main:
        parts.append(main['4B'])

    # 5-7. Headcount (active and bench are ALWAYS reported separately, matching the snapshot figure)
    has_res = df_res_raw is not None and len(df_res_raw) > 0 and 'Month' in df_res_raw.columns
    if has_res:
        emp = column_map.get('res_emp')
        emp = emp if emp in df_res_raw.columns else None

        rows = []
        for m in get_available_months(df_res_raw):
            active, bench = identify_bench_resources(df_res_raw[df_res_raw['Month'] == m], column_map)
            rows.append((m, _hc(active, emp), _hc(bench, emp)))
        parts.append("TABLE 5 - HEADCOUNT BY MONTH (Active_Headcount excludes bench; Bench_Headcount is separate)\n" +
                     _csv(pd.DataFrame(rows, columns=['Month', 'Active_Headcount', 'Bench_Headcount'])))

        hc = _tables_headcount_month(df_res_raw[df_res_raw['Month'] == selected_month],
                                     selected_month, column_map, top_n)
        parts += [hc[k] for k in ('6', '7') if k in hc]

    # Overall / YTD / comparison questions: every loaded month, each from its own default file
    if multi_month and len(months) > 1:
        region = column_map.get('region')
        if region and rev and region in df_cir_raw.columns:
            pr = df_cir_raw.pivot_table(index=region, columns='Month', values=rev, aggfunc='sum')
            pr = pr.reindex(columns=[m for m in months if m in pr.columns])
            pr['Total_all_months'] = pr.sum(axis=1)
            pr = pr.sort_values('Total_all_months', ascending=False).reset_index()
            parts.append("TABLE 8 - REVENUE USD m BY REGION ACROSS MONTHS (each month from its own default file)\n" + _csv(pr))
        if client and client in df_cir_raw.columns:
            t = _agg(df_cir_raw, client, column_map)
            if not t.empty:
                t = t.sort_values('Revenue_USDm' if 'Revenue_USDm' in t.columns else t.columns[1], ascending=False)
                scope = f"top {top_n} of {len(t)}" if len(t) > top_n else f"all {len(t)}"
                parts.append(f"TABLE 9 - ALL LOADED MONTHS COMBINED ({months[0]}-{months[-1]}) BY CLIENT "
                             f"({scope}, sorted by revenue)\n" + _csv(t.head(top_n)))

    # Other months named in the question
    for m in extra_months or []:
        if m == selected_month or m not in months:
            continue
        block = [f"=== ADDITIONAL MONTH REQUESTED IN THE QUESTION: {m} (same layout as TABLES 2, 3, 4B, 6, 7 above) ==="]
        ex = _tables_client_region(df_cir_raw[df_cir_raw['Month'] == m], m, column_map, top_n)
        block += [ex[k] for k in ('2', '3', '4B') if k in ex]
        if has_res:
            exh = _tables_headcount_month(df_res_raw[df_res_raw['Month'] == m], m, column_map, top_n)
            block += [exh[k] for k in ('6', '7') if k in exh]
        parts.append("\n\n".join(block))

    return "\n\n".join(parts)


def build_chart_frames(cir_raw, selected_month, column_map, top_n=10):
    """Frames behind the dashboard charts (same maths as the tables the model reads)."""
    months = get_available_months(cir_raw)
    frames = {}

    trend = _agg(cir_raw, 'Month', column_map)
    if not trend.empty:
        trend['Month'] = pd.Categorical(trend['Month'], categories=months, ordered=True)
        frames['trend'] = trend.sort_values('Month')

    cur = cir_raw[cir_raw['Month'] == selected_month]
    for key in ('client', 'region'):
        col = column_map.get(key)
        if col and col in cur.columns:
            t = _agg(cur, col, column_map)
            if not t.empty and 'Revenue_USDm' in t.columns:
                frames[key] = t.sort_values('Revenue_USDm', ascending=False).head(top_n)
    return frames


def generate_dynamic_context(df_cir, df_res, metrics, df_cir_raw, selected_month, column_map=None,
                             data_tables="", extra_months=None, source_note=""):
    """Context injected into the system prompt."""
    lines = ["=" * 60, "GGM PORTFOLIO ANALYSIS CONTEXT", "=" * 60, ""]

    if column_map:
        lines += ["DETECTED COLUMNS:", get_column_explanations(column_map), ""]

    # Year: prefer the loader's year column, then a 'Year' column from the file, else the YEAR constant
    year = YEAR
    if df_cir_raw is not None and not df_cir_raw.empty:
        for ycol in (YEAR_COL, 'Year'):
            if ycol in df_cir_raw.columns:
                vals = df_cir_raw[ycol].dropna().unique()
                if len(vals) > 0:
                    try:
                        year = str(int(vals[0]))
                    except (TypeError, ValueError):
                        year = str(vals[0])
                    break

    snapshot_period = f"{selected_month} {year}"
    extra_months = [m for m in (extra_months or []) if m != selected_month]

    lines += [
        f"SNAPSHOT - {snapshot_period} (USD millions unless stated):",
        f"- Revenue: {metrics.get('revenue', 0):.3f}",
        f"- Gross Profit: {metrics.get('profit', 0):.3f}",
        f"- GPM: {metrics.get('gpm', 0):.2f}%",
        f"- Headcount (active, bench excluded): {metrics.get('headcount', 0)}",
    ]
    if metrics.get('headcount_bench', 0) > 0:
        lines.append(f"- Bench headcount (separate from active): {metrics.get('headcount_bench', 0)}")
    lines += [f"- Active accounts: {metrics.get('accounts', 0)}", ""]

    if df_cir_raw is not None and 'Month' in df_cir_raw.columns:
        months_loaded = get_available_months(df_cir_raw)
        lines += [f"MONTHS LOADED: {', '.join(months_loaded)} ({year})", ""]

    lines += ["CRITICAL CONTEXT:"]
    lines += [f"- User has FILTERED to month: {selected_month}"]
    lines += [f"- Default: answer for {selected_month} ONLY (e.g. 'what is revenue?' = {selected_month} revenue)"]
    lines += ["- If the user names a different month, or asks to compare / vs / change / previous month, use the tables for those months"]
    if extra_months:
        lines += [f"- This question names additional month(s): {', '.join(extra_months)} - their tables are included below"]
    lines += [""]

    if source_note:
        lines += ["DATA SOURCES (default file for each month - single-month answers use ONLY these):", source_note, ""]

    lines += ["VP TERMINOLOGY:"]
    lines += [f"- {metric}: {', '.join(syns[:5])}" for metric, syns in SYNONYM_MAP.items()]
    lines += [""]

    # Explicit context about Europe clients (helps answer "which clients in Europe?")
    if df_cir is not None and not df_cir.empty and column_map and column_map.get('client') and column_map.get('region'):
        client_col = column_map.get('client')
        region_col = column_map.get('region')
        if client_col in df_cir.columns and region_col in df_cir.columns:
            is_europe = df_cir[region_col].astype(str).str.strip().str.lower() == 'europe'
            europe_clients = df_cir.loc[is_europe, client_col].dropna().astype(str).unique()
            if len(europe_clients) > 0:
                lines += [f"EUROPE REGION CLIENTS (for {selected_month}): {', '.join(sorted(europe_clients))}"]
                lines += [""]

    if data_tables:
        lines += ["DATA TABLES (pre-computed with pandas - the only source for figures):", data_tables, ""]

    return "\n".join(lines)


def estimate_tokens(text):
    """Rough token estimate (no tokenizer dependency). Deliberately conservative for CSV-heavy text."""
    return int(len(text) / CHARS_PER_TOKEN) + 1


def build_system_prompt(dynamic_context, hint_line=""):
    return f"""You are a senior portfolio analyst at GGM Data & Insights, answering questions from VPs.

{dynamic_context}

RULES
1. Answer ONLY from the snapshot and DATA TABLES above. Quote figures exactly as shown (USD millions; GPM as %).
2. MONTH FILTER RULE (CRITICAL): The user has selected a month. Answer for that month ONLY, UNLESS the question names another month or asks for "comparison", "vs", "change", or "previous month". Examples:
   - Q: "What is our revenue?" -> A: Revenue for the selected month only
   - Q: "Compare August to July" -> A: Use Table 4 to show both months
   - Q: "Revenue by region in July" -> A: Use the tables under "ADDITIONAL MONTH REQUESTED" for July
   Default: single-month analysis.
3. Do NOT calculate totals, averages, growth rates or rankings yourself. Use the pre-computed columns (e.g. Chg_pct) and the order the tables are sorted in. Region totals are in TABLE 3; a region's accounts with revenue/GP/GPM are in TABLE 4B. If a figure you need is not in the tables, say it is not available and name the data that would be needed.
4. Do NOT assume, guess, or estimate any numbers. All figures must come directly from the tables. If you cannot find a number in the data, say "not available in the current data" rather than approximating or deriving unstated values.
5. Translate VP wording to metrics (sales -> revenue, GP -> profit, margin -> GPM, team -> headcount, client/customer -> account).
6. For questions about accounts/clients in a region (e.g. "account-level revenue in Europe"), read that region's rows from TABLE 4B (client, Revenue_USDm, GP_USDm, GPM_pct) and quote the region total from TABLE 3. If TABLE 4B says its list is truncated, say so.
7. Tables marked "top N of M" are truncated - make no claims about accounts that are not shown.
8. Headcount: "Active" excludes bench; bench is reported separately. Never add or mix them unless asked. Client names in the headcount data may be spelled differently from the financial data - flag a mismatch rather than guess.
9. Be concise and executive-ready: lead with the answer, add 2-4 supporting points, finish with one suggested follow-up.
10. SOURCES (CRITICAL): each month's default figures come from the file that reports that month (see DATA SOURCES) - e.g. August from the August file. A single-month question uses ONLY that month's default figures. Comparison / overall / YTD questions combine months using their own default figures (TABLES 1, 4, 8, 9). If a month is listed in TABLE M, its numbers differ between files: when comparing, state BOTH values separately, labelled by file (Default vs Other), say which one is the default, and never average or blend them. If a single-month question is about a month listed in TABLE M, add one short note.
11. NAMES AND REGIONS: use client and region names exactly as written in the tables. Never add, rename, merge, group or infer clients, and never fill gaps from general knowledge. For a region, list only that region's rows from TABLE 4B. Say "top N" only when a table title says so; otherwise say how many rows the table has.{hint_line}"""


def assemble_prompt(df_cir, df_res, metrics, cir_raw, res_raw, selected_month, column_map,
                    extra_months, history, hint_line, token_budget, extras=None):
    """
    Build the system prompt and keep the whole request inside the token budget.
    Order of trimming: shrink every table (TOP_N -> 15 -> 10 -> 7 -> 5 rows), then drop the
    oldest chat turns. The current question is never dropped.
    Returns a dict: system_prompt, tables, top_n, est_tokens, history, trimmed_rows,
    trimmed_history, over_budget, input_budget.
    """
    extras = extras or {}
    input_budget = max(int(token_budget) - MAX_OUTPUT_TOKENS, 1500)
    steps = [TOP_N] + [n for n in (15, 10, 7, 5) if n < TOP_N]
    history = list(history)

    def _hist_tokens(h):
        return sum(estimate_tokens(m['content']) for m in h)

    n = steps[0]
    tables = system_prompt = ""
    base = 0
    for n in steps:
        tables = build_data_tables(cir_raw, res_raw, selected_month, column_map, top_n=n, extra_months=extra_months,
                                   mismatches=extras.get('mismatches'), multi_month=extras.get('multi_month', False))
        ctx = generate_dynamic_context(df_cir, df_res, metrics, cir_raw, selected_month, column_map, tables, extra_months,
                                       source_note=extras.get('source_note', ''))
        system_prompt = build_system_prompt(ctx, hint_line)
        base = estimate_tokens(system_prompt)
        if base + _hist_tokens(history) <= input_budget:
            break

    dropped = 0
    while len(history) > 1 and base + _hist_tokens(history) > input_budget:
        history = history[1:]
        dropped += 1

    total = base + _hist_tokens(history)
    return {
        "system_prompt": system_prompt, "tables": tables, "top_n": n, "est_tokens": total,
        "history": history, "trimmed_rows": n < steps[0], "trimmed_history": dropped,
        "over_budget": total > input_budget, "input_budget": input_budget,
    }


_LIMIT_HINTS = ('413', 'request too large', 'tokens per minute', 'rate_limit', 'rate limit',
                'too many tokens', 'context length', 'context_length', 'reduce the length')


def _friendly_groq_error(e):
    """Turn token-limit / rate-limit failures into a clear message; other errors pass through unchanged."""
    msg = str(e)
    if any(h in msg.lower() for h in _LIMIT_HINTS):
        return "token/rate limit exceeded for this model"
    return msg


def get_groq_response(client, messages, models):
    """Try each model in turn. Returns (text, model_used, errors) - text is None if all failed."""
    errors = []
    for model in models:
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, max_tokens=MAX_OUTPUT_TOKENS, temperature=0.2
            )
            text = response.choices[0].message.content
            if text and text.strip():
                return text, model, errors
            errors.append(f"{model}: empty response")
        except Exception as e:
            # Groq's "Connection error." hides the real reason (proxy, SSL, DNS) in __cause__
            cause = f" | cause: {type(e.__cause__).__name__}: {e.__cause__}" if e.__cause__ else ""
            friendly = _friendly_groq_error(e)
            detail = f" | details: {str(e)[:300]}" if friendly != str(e) else ""
            errors.append(f"{model}: {friendly}{detail}{cause}")
    return None, None, errors


def suggest_questions(df_cir, column_map=None, selected_month=None):
    """Generate context-aware follow-up questions for selected month"""
    suggestions = []
    month_phrase = f"in {selected_month}" if selected_month else ""

    if column_map and column_map.get('client'):
        suggestions.append(f"👥 Which client has the highest revenue {month_phrase}?")
    if column_map and column_map.get('region'):
        suggestions.append(f"🗺️ What's the revenue breakdown by region {month_phrase}?")
    if column_map and column_map.get('gpm'):
        suggestions.append(f"💰 Which accounts are most profitable {month_phrase}?")
    
    # Add comparison question if data supports it
    suggestions.extend([
        f"📊 Revenue summary by client {month_phrase}",
        f"📈 Compare {selected_month} to previous month" if selected_month else "📈 Compare months",
        "🎯 Which region drives most revenue?",
    ])
    return suggestions[:5]


def translate_vp_language(query):
    """Translate VP business language to metric names"""
    query_lower = query.lower()
    translations = []
    for metric, synonyms in SYNONYM_MAP.items():
        if any(s in query_lower for s in synonyms):
            translations.append(metric)
    return translations


def _read_uploads(files, kind, log):
    """Read uploaded Excel files into one frame. Month AND year come from the file name
    (year falls back to YEAR). The Circle filter is applied exactly like the folder loader does."""
    by_period = {}
    default_year = int(YEAR) if str(YEAR).isdigit() else None
    for f in files or []:
        # Read by sheet name "Sheet1" (source of truth), with fallback to index 0
        try:
            df = pd.read_excel(f, sheet_name='Sheet1', header=0)
        except ValueError:
            log.append(f"{kind}: sheet 'Sheet1' not found in '{f.name}', using first sheet...")
            df = pd.read_excel(f, sheet_name=0, header=0)
        df.columns = [str(c).strip() for c in df.columns]
        month = infer_month_from_text(f.name) or 'Uploaded'
        year = infer_year_from_text(f.name, default=default_year)
        # Month per row comes from the file's own Month column (a monthly file also holds earlier months);
        # the file-name month is only the file's report month / fallback.
        assign_months(df, month, f"{kind} '{f.name}'", log)
        month = df['Report_Month'].iloc[0] if len(df) else month
        if (year, month) in by_period:
            log.append(f"{kind}: more than one file reporting '{month} {year}' - '{f.name}' replaced the earlier one. "
                       f"Put the month in the file name (e.g. '... Aug 2026.xlsx') to load several months.")
        df[SOURCE_COL] = f.name
        df[YEAR_COL] = year

        if 'Circle' in df.columns:
            df = apply_circle_filter(df, 'Circle', CIRCLE_NAME, f"{kind} '{f.name}'", log)
            if len(df) == 0:
                log.append(f"⚠️ {kind}: '{f.name}' had no rows matching Circle='{CIRCLE_NAME}' - file skipped")
                continue
        by_period[(year, month)] = df
    return pd.concat(by_period.values(), ignore_index=True) if by_period else pd.DataFrame()


# ============================================================================
# UI
# ============================================================================
st.set_page_config(page_title="GGM D&I Portfolio Agent", layout="wide")
st.title("📊 GGM D&I Portfolio Agent")

# Everything that must survive a rerun lives in session_state
for _k, _v in {"messages": [], "pending_prompt": None, "local_data": None,
               "upload_data": None, "upload_sig": None}.items():
    st.session_state.setdefault(_k, _v)

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Data Source")

base_path = get_secret("paths", "base_path") or os.environ.get("GGM_BASE_PATH") or DEFAULT_BASE_PATH
years_cfg = get_secret("paths", "years") or os.environ.get("GGM_YEARS") or YEARS
local_ok = os.path.isdir(base_path)

source = st.sidebar.radio("Choose source:", ["Local Folder (Auto-Load)", "Manual Upload"],
                          index=0 if local_ok else 1, key="data_source")

if source.startswith("Local"):
    if not local_ok:
        st.sidebar.warning("The local data folder isn't available on this machine "
                           "(expected on Streamlit Cloud). Use Manual Upload.")
    else:
        st.sidebar.caption("📁 Auto-loading from the GGM Portfolio Analytics folder")
        if st.sidebar.button("🔄 Load Historical Data"):
            load_log = []
            with st.spinner("Scanning folders and loading data..."):
                try:
                    cir, res = load_monthly_data_from_folder(
                        base_path, years_cfg, circle_name='', 
                        circle_column='Circle', circle_value=CIRCLE_NAME, 
                        log=load_log
                    )
                    st.session_state.local_data = {"cir": cir, "res": res, "log": load_log}
                except Exception as e:
                    st.session_state.local_data = None
                    st.sidebar.error(f"❌ Error: {e}")
    data = st.session_state.local_data

else:
    st.sidebar.caption("📁 Upload one Circle Wise file per month. Put the month in the file name "
                       "(e.g. '... Aug 2026.xlsx') so trends work.")
    circle_files = st.sidebar.file_uploader("Circle Wise Data (required)", type=['xlsx'],
                                            accept_multiple_files=True)
    resource_files = st.sidebar.file_uploader("Resource / Headcount Data (optional)", type=['xlsx'],
                                              accept_multiple_files=True)
    sig = tuple((f.name, f.size) for f in (list(circle_files or []) + list(resource_files or [])))

    if not circle_files:
        st.session_state.upload_data, st.session_state.upload_sig = None, None
    elif sig != st.session_state.upload_sig:
        up_log = []
        try:
            # Circle filter is applied per file inside _read_uploads - same rule as the folder loader
            cir_uploaded = _read_uploads(circle_files, "Circle Wise", up_log)
            res_uploaded = _read_uploads(resource_files, "Resource", up_log)
            if resource_files and res_uploaded.empty:
                st.sidebar.warning("Resource file(s) loaded no rows - check the Circle value / file names "
                                   "(see Data Quality Log).")
            st.session_state.upload_data = {"cir": cir_uploaded, "res": res_uploaded, "log": up_log}
            st.session_state.upload_sig = sig
        except Exception as e:
            st.session_state.upload_data = None
            st.sidebar.error(f"❌ Error reading files: {e}")
    data = st.session_state.upload_data

# ---- Nothing loaded yet ----
if data is None or data["cir"] is None or data["cir"].empty:
    if data is not None:
        st.error("❌ No Circle Wise data was loaded. Check the folder structure and file names.")
        for line in data.get("log", []):
            st.caption(f"• {line}")
    else:
        st.info("📁 Upload Circle Wise data or click 'Load Historical Data' to start")
    st.stop()

# ---- Prepare data (runs every rerun, cheap) ----
cir_all = data["cir"]
res_all = data["res"] if data["res"] is not None else pd.DataFrame()

# Year: the loader tags every row; with several years loaded, one year is analysed at a time
# so the same month in different years can never be summed together.
selected_year = None
if YEAR_COL in cir_all.columns:
    years = sorted(cir_all[YEAR_COL].dropna().unique().tolist(), key=str)
    if years:
        if len(years) > 1:
            selected_year = st.sidebar.selectbox("Select year:", options=years, index=len(years) - 1,
                                                 key="year_" + "_".join(map(str, years)))
        else:
            selected_year = years[0]
        cir_all = cir_all[cir_all[YEAR_COL] == selected_year].copy()
        if YEAR_COL in res_all.columns:
            res_all = res_all[res_all[YEAR_COL] == selected_year].copy()

cir_raw, circle_filtered = filter_circle(cir_all)
res_raw = res_all
column_map = detect_column_mapping(cir_raw, res_raw)
cir_raw, dropped_rows = clean_circle_df(cir_raw, column_map)

# Clean client/region labels BEFORE overrides and every groupby (trim, collapse spaces, merge case variants)
cir_raw, norm_cir = normalize_labels(cir_raw, [column_map.get('client'), column_map.get('region')])
res_raw, norm_res = normalize_labels(res_raw, [column_map.get('res_client'), column_map.get('res_region')])

# Apply region overrides to raw data for both Circle Wise and Resource (affects all aggregations)
cir_raw = apply_region_overrides(cir_raw, column_map, data_type='cir')
res_raw = apply_region_overrides(res_raw, column_map, data_type='res')

# A monthly file also holds earlier months, so a month can appear in several files. Default = the file that
# REPORTS the month (August -> the August file). Same month in other files is kept aside ONLY to show differences.
cir_raw, cir_other = select_authoritative(cir_raw)
res_raw, _res_other = select_authoritative(res_raw)
mm_month, mm_client = month_mismatches(cir_raw, cir_other, column_map.get('revenue'), column_map.get('cost'),
                                       column_map.get('profit'), column_map.get('client'))
source_note = describe_sources(cir_raw)

months = get_available_months(cir_raw)
selected_month = st.sidebar.selectbox("Select month for dashboard:", options=months,
                                      index=len(months) - 1, key=f"month_{selected_year}_" + "_".join(months))
st.sidebar.success(f"✅ Data loaded - {f'{selected_year} ' if selected_year is not None else ''}months: {', '.join(months)}")

if mm_month is not None and not mm_month.empty:
    _pairs = sorted({f"{r.Month} ({r.Default_Source} vs {r.Other_Source})" for r in mm_month.itertuples()})
    st.warning("⚠️ **Numbers differ between files for:** " + ", ".join(_pairs) +
               ". The default is the file that reports the month; both sets of numbers are shown under "
               "'Detected columns & data checks'.")

# CRITICAL DATA VALIDATION - Show exactly what data is being used (AFTER month selector)
with st.expander("🔍 DATA VALIDATION - Exact Figures for Selected Month"):
    debug_lines = []
    debug_lines.append(f"✓ Columns: client='{column_map.get('client')}', region='{column_map.get('region')}', revenue='{column_map.get('revenue')}'")
    debug_lines.append(f"✓ Filtering for Month = '{selected_month}'")
    
    if 'Month' in cir_raw.columns:
        selected_data = cir_raw[cir_raw['Month'].astype(str).str.strip() == selected_month]
        debug_lines.append(f"✓ Rows in {selected_month}: {len(selected_data)} (from {len(cir_raw)} total)")
        if SOURCE_COL in selected_data.columns:
            debug_lines.append(f"✓ {selected_month} figures come from: {', '.join(sorted(selected_data[SOURCE_COL].astype(str).unique()))}")
        
        rev_col = column_map.get('revenue')
        region_col = column_map.get('region')
        
        if rev_col and rev_col in selected_data.columns:
            total_revenue = selected_data[rev_col].sum()
            debug_lines.append(f"✓ Total {selected_month} Revenue: ${total_revenue:.3f}m")
        
        if region_col and region_col in selected_data.columns:
            debug_lines.append(f"  Regions in {selected_month}:")
            for reg in sorted(selected_data[region_col].unique()):
                reg_rev = selected_data[selected_data[region_col].astype(str).str.strip() == str(reg)][rev_col].sum() if rev_col else 0
                reg_count = len(selected_data[selected_data[region_col].astype(str).str.strip() == str(reg)])
                debug_lines.append(f"    - {reg}: ${reg_rev:.3f}m ({reg_count} rows)")
    
    for line in debug_lines:
        st.caption(line)

df_cir = cir_raw[cir_raw["Month"] == selected_month].copy()
df_res = res_raw[res_raw["Month"] == selected_month].copy() if "Month" in res_raw.columns else res_raw

if not column_map.get("revenue"):
    st.error("Couldn't detect a revenue column. Columns found: " + ", ".join(map(str, cir_raw.columns)))

metrics = get_available_metrics(df_cir, df_res, column_map)

groq_api_key = get_secret("groq", "api_key") or os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    st.warning("⚠️ **Groq API key not configured** - add it to .streamlit/secrets.toml (local) or the Streamlit Cloud Secrets settings.")
models = list(get_secret("groq", "models") or DEFAULT_MODELS)
try:
    token_budget = int(get_secret("groq", "token_budget") or os.environ.get("GROQ_TOKEN_BUDGET") or MODEL_TOKEN_BUDGET)
except (TypeError, ValueError):
    token_budget = MODEL_TOKEN_BUDGET

col1, col2, col3, col4 = st.columns(4)
col1.metric("Revenue", f"${metrics['revenue']:.3f}M")
col2.metric("Gross Profit", f"${metrics['profit']:.3f}M", f"GPM: {metrics['gpm']:.2f}%")
col3.metric("Active Headcount", metrics["headcount"])
if metrics.get('headcount_bench', 0) > 0:
    col4.metric("Bench Headcount", metrics.get("headcount_bench", 0))
else:
    col4.metric("Accounts", metrics["accounts"])

with st.expander("📊 Charts", expanded=True):
    try:
        import plotly.express as px
    except ImportError:
        px = None
        st.info("Charts need plotly - add `plotly` to requirements.txt.")

    if px is not None:
        frames = build_chart_frames(cir_raw, selected_month, column_map)
        tab_trend, tab_client, tab_region = st.tabs(["Monthly trend", "By client", "By region"])

        with tab_trend:
            if 'trend' in frames and len(months) > 1:
                t = frames['trend']
                value_cols = [c for c in ('Revenue_USDm', 'GP_USDm') if c in t.columns]
                long = t.melt(id_vars='Month', value_vars=value_cols, var_name='Metric', value_name='USD m')
                fig = px.bar(long, x='Month', y='USD m', color='Metric', barmode='group', text_auto='.2f')
                fig.update_layout(legend_title_text='', margin=dict(t=20))
                st.plotly_chart(fig, use_container_width=True)
                if 'GPM_pct' in t.columns:
                    st.caption("GPM %: " + " · ".join(f"{m}: {g:.1f}%" for m, g in zip(t['Month'], t['GPM_pct']) if pd.notna(g)))
            else:
                st.caption("Load at least two months to see a trend.")

        for tab, key, label in ((tab_client, 'client', 'client'), (tab_region, 'region', 'region')):
            with tab:
                if key in frames:
                    t = frames[key]
                    name_col = column_map[key]
                    hover = [c for c in ('GP_USDm', 'GPM_pct') if c in t.columns]
                    fig = px.bar(t.iloc[::-1], x='Revenue_USDm', y=name_col, orientation='h',
                                 hover_data=hover, text_auto='.2f',
                                 labels={'Revenue_USDm': 'Revenue USD m'})
                    fig.update_layout(margin=dict(t=20), yaxis_title='')
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"{selected_month} revenue, top {len(t)} {label}s. Hover for GP and GPM %.")
                else:
                    st.caption(f"No {label} column detected.")

with st.expander("📋 Detected columns & data checks"):
    st.markdown(get_column_explanations(column_map) or "No columns detected.")
    chk = cir_raw.groupby("Month").size().rename("Circle rows").to_frame()
    if "Month" in res_raw.columns:
        chk["Resource rows"] = res_raw.groupby("Month").size()
    st.dataframe(chk.reindex(months))
    st.caption(f"Circle filter ('{CIRCLE_NAME}') applied: {'yes' if circle_filtered else 'NO - no matching Circle column/value, all rows used'}")
    st.caption(f"'Total' summary rows removed: {dropped_rows}")
    cleanup = {**{f"revenue file / {k}": v for k, v in norm_cir.items()},
               **{f"resource file / {k}": v for k, v in norm_res.items()}}
    if cleanup:
        st.caption("Label cleanup merged spelling/spacing variants - " +
                   ", ".join(f"{k}: {v}" for k, v in cleanup.items()))
    if selected_year is not None:
        st.caption(f"Year analysed: {selected_year}")
    if source_note:
        st.caption("Default source file per month:")
        for line in source_note.split("\n"):
            st.caption(line)
    st.subheader("Months that differ between files")
    if mm_month is not None and not mm_month.empty:
        st.caption("Default_* = file that reports the month (used by default). Other_* = same month in another file. USD m.")
        st.dataframe(mm_month.round(3))
        if mm_client is not None and not mm_client.empty:
            st.caption("Clients behind the differences (revenue):")
            st.dataframe(mm_client.round(3))
    else:
        st.caption("No differences found between files for the same month.")
    
    # Client matching validation
    validation_log = []
    matched, rev_only, res_only = validate_client_matching(cir_raw, res_raw, column_map, validation_log)
    st.caption(f"**Client matching:** {len(matched)} match both files")
    if validation_log:
        st.warning("⚠️ Client name mismatches detected:")
        for line in validation_log:
            st.caption(f"• {line}")
    
    st.subheader("Data Quality Log")
    for line in data.get("log", []):
        st.caption(f"• {line}")

st.divider()
st.subheader("💬 Ask Questions")
st.info(f"📍 **Context: All answers are for {selected_month} data by default.** Ask about trends, comparisons, or other months explicitly if needed.")
st.caption(f"E.g., 'What is revenue?' = {selected_month} revenue | 'Compare to July' = month-over-month comparison")

with st.expander("💡 Suggested questions"):
    for idx, suggestion in enumerate(suggest_questions(df_cir, column_map, selected_month), 1):
        if st.button(suggestion, key=f"suggest_{idx}"):
            st.session_state.pending_prompt = suggestion
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

typed = st.chat_input("Ask about revenue, accounts, regions, margins... (VP language welcome!)")
prompt = st.session_state.pending_prompt or typed
st.session_state.pending_prompt = None

if prompt:
    # The turn is saved to chat history ONLY after the model answers, so a failed call
    # never leaves a dangling user message that would be re-sent with the next question.
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not groq_api_key:
            st.error("❌ Groq API key not configured.")
        else:
            with st.spinner("🔍 Analyzing your question..."):
                try:
                    from groq import Groq
                    client = Groq(api_key=groq_api_key)

                    # Other loaded months named in the question (e.g. "region revenue in July") get their own tables
                    extra_months = [m for m in find_months_in_text(prompt, months) if m != selected_month]
                    hint = translate_vp_language(prompt)
                    hint_line = f"\nMetric terms detected in this question: {', '.join(hint)}." if hint else ""

                    prior = st.session_state.messages[-(HISTORY_TURNS - 1):] if HISTORY_TURNS > 1 else []
                    history = [{"role": m["role"], "content": m["content"]} for m in prior]
                    history.append({"role": "user", "content": prompt})

                    extras = {
                        "mismatches": (mm_month, mm_client),
                        "multi_month": wants_multi_month(prompt, extra_months),
                        "source_note": source_note,
                    }
                    pr = assemble_prompt(df_cir, df_res, metrics, cir_raw, res_raw, selected_month, column_map,
                                         extra_months, history, hint_line, token_budget, extras)
                    messages = [{"role": "system", "content": pr["system_prompt"]}] + pr["history"]

                    # Show what data is being sent to model
                    with st.expander("📊 Data Tables Sent to Model"):
                        st.code(pr["tables"], language="text")
                    if pr["trimmed_rows"] or pr["trimmed_history"]:
                        st.caption(f"ℹ️ Context trimmed to fit the model's token budget: tables cut to top {pr['top_n']} rows"
                                   + (f", {pr['trimmed_history']} older chat message(s) left out" if pr["trimmed_history"] else "")
                                   + ". Ask about a specific account/region if it isn't shown.")
                    if pr["over_budget"]:
                        st.warning(f"⚠️ This request (~{pr['est_tokens']} tokens) is still above the configured input budget "
                                   f"(~{pr['input_budget']}). The model may reject it - raise [groq] token_budget in secrets "
                                   f"if your plan allows more, or ask a narrower question.")

                    answer, used_model, errors = get_groq_response(client, messages, models)

                    if answer:
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.markdown(answer)
                        months_note = f" + {', '.join(extra_months)}" if extra_months else ""
                        src_files = (sorted(df_cir[SOURCE_COL].astype(str).unique())
                                     if SOURCE_COL in df_cir.columns else [])
                        src_txt = f" · {selected_month} source: {', '.join(src_files)}" if src_files else ""
                        st.caption(f"Model: {used_model} · Figures computed from {selected_month}{months_note} data and loaded months{src_txt}")
                        if errors:
                            with st.expander("Fallback details"):
                                for e in errors:
                                    st.caption(e)
                    else:
                        st.error("❌ No response from any configured model. Your question was not saved - please resend it.")
                        if any("token/rate limit" in e for e in errors):
                            st.info("The request hit a model token/rate limit. Wait a minute and retry, ask a narrower "
                                    "question, or lower the amount of data (see [groq] token_budget).")
                        with st.expander("Details"):
                            for e in errors:
                                st.caption(e)

                except Exception as e:
                    st.error(f"❌ Error: {e}  (your question was not saved - please resend it)")
                    with st.expander("Technical details"):
                        st.code(traceback.format_exc())

if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    st.caption("💡 **Next steps:**")
    followups = suggest_questions(df_cir, column_map, selected_month)[:3]
    for idx, (col, suggestion) in enumerate(zip(st.columns(len(followups)), followups)):
        with col:
            if st.button(suggestion, key=f"followup_{idx}"):
                st.session_state.pending_prompt = suggestion
                st.rerun()