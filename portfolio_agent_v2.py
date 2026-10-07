import os
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

from fuzzywuzzy import fuzz

from local_folder_loader import (
    load_monthly_data_from_folder,
    get_available_months,
    infer_month_from_text,
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


def identify_bench_resources(df_res, column_map):
    """
    Identify bench resources: where client/account field contains 'bench' (case-insensitive).
    Bench resources are allocated to internal bench, not active client accounts.
    Returns (active_res_df, bench_res_df) - two dataframes separated by bench status.
    """
    if df_res is None or df_res.empty:
        return df_res, pd.DataFrame()
    
    client_col = column_map.get('res_client')
    if not client_col or client_col not in df_res.columns:
        return df_res, pd.DataFrame()
    
    df_res = df_res.copy()
    is_bench = df_res[client_col].astype(str).str.lower().str.contains('bench', na=False)
    
    return df_res[~is_bench].copy(), df_res[is_bench].copy()


def validate_client_matching(df_cir, df_res, column_map, log=None, fuzzy_threshold=85):
    """
    Validate that client names in revenue file match resource file (with fuzzy matching).
    Returns (matched_clients, revenue_only, resource_only) for data quality checks.
    
    Fuzzy matching: if a revenue client is >=85% similar to a resource client, they're considered matched.
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
    
    # Get unique clients from each source (strip whitespace for comparison)
    revenue_clients = set(df_cir[client_cir_col].astype(str).str.strip().unique())
    resource_clients = set(df_res[client_res_col].astype(str).str.strip().unique())
    
    # Remove NaN/None (ensure string conversion to avoid float.lower() error)
    revenue_clients = {c for c in revenue_clients if c and str(c).lower() != 'nan'}
    resource_clients = {c for c in resource_clients if c and str(c).lower() != 'nan'}
    
    # Exact matches first
    matched = revenue_clients & resource_clients
    revenue_unmatched = revenue_clients - resource_clients
    resource_unmatched = resource_clients - matched
    
    # Fuzzy matching on unmatched clients
    fuzzy_matched = {}
    for rev_client in revenue_unmatched.copy():
        for res_client in resource_unmatched.copy():
            similarity = fuzz.token_set_ratio(str(rev_client).lower(), str(res_client).lower())
            if similarity >= fuzzy_threshold:
                fuzzy_matched[rev_client] = res_client
                revenue_unmatched.discard(rev_client)
                resource_unmatched.discard(res_client)
                matched.add(rev_client)
                break
    
    if fuzzy_matched:
        matches_str = ", ".join([f"'{k}' ≈ '{v}'" for k, v in sorted(fuzzy_matched.items())[:3]])
        log.append(f"✓ Fuzzy matched ({fuzzy_threshold}% similarity): {matches_str}" + 
                   (f" (+{len(fuzzy_matched)-3} more)" if len(fuzzy_matched) > 3 else ""))
    
    if revenue_unmatched:
        log.append(f"⚠️ In revenue but not in resource: {', '.join(sorted(revenue_unmatched)[:5])}" + 
                   (f" (+{len(revenue_unmatched)-5} more)" if len(revenue_unmatched) > 5 else ""))
    if resource_unmatched:
        log.append(f"⚠️ In resource but not in revenue: {', '.join(sorted(resource_unmatched)[:5])}" + 
                   (f" (+{len(resource_unmatched)-5} more)" if len(resource_unmatched) > 5 else ""))
    
    return matched, revenue_unmatched, resource_unmatched


def apply_region_overrides(df, column_map, data_type='cir'):
    """
    Apply hardcoded region mappings for specific clients.
    - Digiterre → Europe
    - SFI → Europe
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
    overrides = {
        'Digiterre': 'Europe',
        'SFI': 'Europe',
    }
    
    for client_name, target_region in overrides.items():
        mask = df[client_col].astype(str).str.strip() == client_name
        if mask.any():
            df.loc[mask, region_col] = target_region
    
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


def build_data_tables(df_cir_raw, df_res_raw, selected_month, column_map, top_n=TOP_N):
    """All figures the model is allowed to quote, computed with pandas."""
    parts = []
    months = get_available_months(df_cir_raw)
    cur = df_cir_raw[df_cir_raw['Month'] == selected_month]
    client, region, rev = column_map.get('client'), column_map.get('region'), column_map.get('revenue')

    # 1. Totals by month
    trend = _agg(df_cir_raw, 'Month', column_map)
    if not trend.empty:
        trend['Month'] = pd.Categorical(trend['Month'], categories=months, ordered=True)
        trend = trend.sort_values('Month')
        parts.append("TABLE 1 - TOTALS BY MONTH (all loaded months; GPM_pct = GP / Revenue)\n" + _csv(trend))

    # 2 & 3. Selected month by client / region
    for label, col in (('CLIENT', client), ('REGION', region)):
        if not col or col not in cur.columns:
            continue
        t = _agg(cur, col, column_map)
        if t.empty:
            continue
        t = t.sort_values('Revenue_USDm' if 'Revenue_USDm' in t.columns else t.columns[1], ascending=False)
        scope = f"top {top_n} of {len(t)}" if len(t) > top_n else f"all {len(t)}"
        n = 2 if label == 'CLIENT' else 3
        parts.append(f"TABLE {n} - {selected_month} BY {label} ({scope}, sorted by revenue)\n" + _csv(t.head(top_n)))

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
            pv = pv.sort_values(selected_month, ascending=False).head(top_n).reset_index()
            parts.append(f"TABLE 4 - REVENUE USD m BY CLIENT ACROSS MONTHS (top {top_n} by {selected_month})\n" + _csv(pv))

    # 5-7. Headcount
    if df_res_raw is not None and len(df_res_raw) > 0 and 'Month' in df_res_raw.columns:
        emp = column_map.get('res_emp')
        emp = emp if emp in df_res_raw.columns else None

        rows = []
        for m in get_available_months(df_res_raw):
            sub = df_res_raw[df_res_raw['Month'] == m]
            rows.append((m, int(sub[emp].nunique()) if emp else len(sub)))
        parts.append("TABLE 5 - HEADCOUNT BY MONTH\n" + _csv(pd.DataFrame(rows, columns=['Month', 'Headcount'])))

        res_cur = df_res_raw[df_res_raw['Month'] == selected_month]
        for n, (label, key) in enumerate((('CLIENT', 'res_client'), ('REGION', 'res_region')), start=6):
            col = column_map.get(key)
            if col and col in res_cur.columns and len(res_cur):
                g = res_cur.groupby(col)[emp].nunique() if emp else res_cur.groupby(col).size()
                g = g.rename('Headcount').reset_index().sort_values('Headcount', ascending=False)
                scope = f"top {top_n} of {len(g)}" if len(g) > top_n else f"all {len(g)}"
                parts.append(f"TABLE {n} - {selected_month} HEADCOUNT BY {label} ({scope})\n" + _csv(g.head(top_n)))

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


def generate_dynamic_context(df_cir, df_res, metrics, df_cir_raw, selected_month, column_map=None, data_tables=""):
    """Context injected into the system prompt."""
    lines = ["=" * 60, "GGM PORTFOLIO ANALYSIS CONTEXT", "=" * 60, ""]

    if column_map:
        lines += ["DETECTED COLUMNS:", get_column_explanations(column_map), ""]

    # Extract year from data if available, default to YEAR constant
    year = YEAR  # Default from YEAR = "2026"
    if df_cir_raw is not None and not df_cir_raw.empty and 'Year' in df_cir_raw.columns:
        year_vals = df_cir_raw['Year'].dropna().unique()
        if len(year_vals) > 0:
            year = str(int(year_vals[0]))
    
    # Format snapshot with explicit year
    snapshot_period = f"{selected_month} {year}"
    
    lines += [
        f"SNAPSHOT - {snapshot_period} (USD millions unless stated):",
        f"- Revenue: {metrics.get('revenue', 0):.3f}",
        f"- Gross Profit: {metrics.get('profit', 0):.3f}",
        f"- GPM: {metrics.get('gpm', 0):.2f}%",
        f"- Headcount: {metrics.get('headcount', 0)}",
        f"- Active accounts: {metrics.get('accounts', 0)}",
        "",
    ]

    if df_cir_raw is not None and 'Month' in df_cir_raw.columns:
        months_loaded = get_available_months(df_cir_raw)
        lines += [f"MONTHS LOADED: {', '.join(months_loaded)} ({year})", ""]

    lines += ["VP TERMINOLOGY:"]
    lines += [f"- {metric}: {', '.join(syns[:5])}" for metric, syns in SYNONYM_MAP.items()]
    lines += [""]

    if data_tables:
        lines += ["DATA TABLES (pre-computed with pandas - the only source for figures):", data_tables, ""]

    return "\n".join(lines)


def get_groq_response(client, messages, models):
    """Try each model in turn. Returns (text, model_used, errors) - text is None if all failed."""
    errors = []
    for model in models:
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, max_tokens=1024, temperature=0.2
            )
            text = response.choices[0].message.content
            if text and text.strip():
                return text, model, errors
            errors.append(f"{model}: empty response")
        except Exception as e:
            # Groq's "Connection error." hides the real reason (proxy, SSL, DNS) in __cause__
            cause = f" | cause: {type(e.__cause__).__name__}: {e.__cause__}" if e.__cause__ else ""
            errors.append(f"{model}: {e}{cause}")
    return None, None, errors


def suggest_questions(df_cir, column_map=None):
    """Generate context-aware follow-up questions"""
    suggestions = []

    if column_map and column_map.get('revenue'):
        suggestions.append("📈 What's our revenue trend across months?")
    if column_map and column_map.get('client'):
        suggestions.append("👥 Which client has the highest revenue?")
    if column_map and column_map.get('region'):
        suggestions.append("🗺️ How does performance compare by region?")
    if column_map and column_map.get('gpm'):
        suggestions.append("💰 Which accounts are most profitable?")

    suggestions.extend([
        "📊 Show me a summary by account",
        "🎯 What are our top 5 priorities?",
        "⚠️ Which areas need improvement?",
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
    """Read uploaded Excel files into one frame; month comes from the file name."""
    by_month = {}
    for f in files or []:
        # Read by sheet name "Sheet1" (source of truth), with fallback to index 0
        try:
            df = pd.read_excel(f, sheet_name='Sheet1', header=0)
        except ValueError:
            log.append(f"{kind}: sheet 'Sheet1' not found in '{f.name}', using first sheet...")
            df = pd.read_excel(f, sheet_name=0, header=0)
        df.columns = [str(c).strip() for c in df.columns]
        month = infer_month_from_text(f.name) or 'Uploaded'
        if month in by_month:
            log.append(f"{kind}: more than one file for '{month}' - '{f.name}' replaced the earlier one. "
                       f"Put the month in the file name (e.g. '... Aug 2026.xlsx') to load several months.")
        df['Month'] = month
        by_month[month] = df
    return pd.concat(by_month.values(), ignore_index=True) if by_month else pd.DataFrame()


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
                        base_path, YEAR, circle_name='', 
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
            cir_uploaded = _read_uploads(circle_files, "Circle Wise", up_log)
            # Filter uploaded data to circle if 'Circle' column exists
            if 'Circle' in cir_uploaded.columns:
                before = len(cir_uploaded)
                cir_uploaded = cir_uploaded[cir_uploaded['Circle'].astype(str).str.strip() == CIRCLE_NAME].copy()
                up_log.append(f"Circle filter: {before} → {len(cir_uploaded)} rows")
            st.session_state.upload_data = {
                "cir": cir_uploaded,
                "res": _read_uploads(resource_files, "Resource", up_log),
                "log": up_log,
            }
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
cir_raw, circle_filtered = filter_circle(data["cir"])
res_raw = data["res"] if data["res"] is not None else pd.DataFrame()
column_map = detect_column_mapping(cir_raw, res_raw)
cir_raw, dropped_rows = clean_circle_df(cir_raw, column_map)

# Apply region overrides to raw data for both Circle Wise and Resource (affects all aggregations)
cir_raw = apply_region_overrides(cir_raw, column_map, data_type='cir')
res_raw = apply_region_overrides(res_raw, column_map, data_type='res')

# Debug: Show status of region overrides
with st.expander("🔍 DEBUG: Data Preparation Status"):
    debug_lines = []
    debug_lines.append(f"✓ Column mapping detected: client='{column_map.get('client')}', region='{column_map.get('region')}'")
    debug_lines.append(f"✓ Data rows: Circle={len(cir_raw)}, Resource={len(res_raw)}")
    
    if 'Month' in cir_raw.columns and column_map.get('client') and column_map.get('region'):
        client_col = column_map.get('client')
        region_col = column_map.get('region')
        
        digiterre_rows = cir_raw[cir_raw[client_col].astype(str).str.strip() == 'Digiterre']
        sfi_rows = cir_raw[cir_raw[client_col].astype(str).str.strip() == 'SFI']
        europe_rows = cir_raw[cir_raw[region_col].astype(str).str.strip() == 'Europe']
        
        debug_lines.append(f"✓ Digiterre rows: {len(digiterre_rows)} → Region={digiterre_rows[region_col].unique().tolist() if len(digiterre_rows) > 0 else 'N/A'}")
        debug_lines.append(f"✓ SFI rows: {len(sfi_rows)} → Region={sfi_rows[region_col].unique().tolist() if len(sfi_rows) > 0 else 'N/A'}")
        debug_lines.append(f"✓ Europe rows (after override): {len(europe_rows)}")
        debug_lines.append(f"  → Includes Digiterre: {('Digiterre' in europe_rows[client_col].values)}")
        debug_lines.append(f"  → Includes SFI: {('SFI' in europe_rows[client_col].values)}")
    
    for line in debug_lines:
        st.caption(line)

months = get_available_months(cir_raw)
selected_month = st.sidebar.selectbox("Select month for dashboard:", options=months,
                                      index=len(months) - 1, key="month_" + "_".join(months))
st.sidebar.success(f"✅ Data loaded - months: {', '.join(months)}")

df_cir = cir_raw[cir_raw["Month"] == selected_month].copy()
df_res = res_raw[res_raw["Month"] == selected_month].copy() if "Month" in res_raw.columns else res_raw

if not column_map.get("revenue"):
    st.error("Couldn't detect a revenue column. Columns found: " + ", ".join(map(str, cir_raw.columns)))

metrics = get_available_metrics(df_cir, df_res, column_map)

groq_api_key = get_secret("groq", "api_key") or os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    st.warning("⚠️ **Groq API key not configured** - add it to .streamlit/secrets.toml (local) or the Streamlit Cloud Secrets settings.")
models = list(get_secret("groq", "models") or DEFAULT_MODELS)

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

with st.expander("💡 Suggested questions"):
    for idx, suggestion in enumerate(suggest_questions(df_cir, column_map), 1):
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
    st.session_state.messages.append({"role": "user", "content": prompt})
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

                    data_tables = build_data_tables(cir_raw, res_raw, selected_month, column_map)
                    dynamic_context = generate_dynamic_context(
                        df_cir, df_res, metrics, cir_raw, selected_month, column_map, data_tables
                    )
                    
                    # Show what data is being sent to model
                    with st.expander("📊 Data Tables Sent to Model"):
                        st.code(data_tables, language="text")
                    hint = translate_vp_language(prompt)
                    hint_line = f"\nMetric terms detected in this question: {', '.join(hint)}." if hint else ""

                    system_prompt = f"""You are a senior portfolio analyst at GGM Data & Insights, answering questions from VPs.

{dynamic_context}

RULES
1. Answer ONLY from the snapshot and DATA TABLES above. Quote figures exactly as shown (USD millions; GPM as %).
2. Do NOT calculate totals, averages, growth rates or rankings yourself. Use the pre-computed columns (e.g. Chg_pct) and the order the tables are sorted in. If a figure you need is not in the tables, say it is not available and name the data that would be needed.
3. Do NOT assume, guess, or estimate any numbers. All figures must come directly from the tables. If you cannot find a number in the data, say "not available in the current data" rather than approximating or deriving unstated values.
4. Translate VP wording to metrics (sales -> revenue, GP -> profit, margin -> GPM, team -> headcount, client/customer -> account).
5. Tables marked "top N of M" are truncated - make no claims about accounts that are not shown.
6. Client names in the headcount data may be spelled differently from the financial data - flag a mismatch rather than guess.
7. Be concise and executive-ready: lead with the answer, add 2-4 supporting points, finish with one suggested follow-up.{hint_line}"""

                    history = [{"role": m["role"], "content": m["content"]}
                               for m in st.session_state.messages[-HISTORY_TURNS:]]
                    messages = [{"role": "system", "content": system_prompt}] + history

                    answer, used_model, errors = get_groq_response(client, messages, models)

                    if answer:
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.markdown(answer)
                        st.caption(f"Model: {used_model} · Figures computed from {selected_month} data and loaded months")
                        if errors:
                            with st.expander("Fallback details"):
                                for e in errors:
                                    st.caption(e)
                    else:
                        st.error("❌ No response from any configured model.")
                        with st.expander("Details"):
                            for e in errors:
                                st.caption(e)

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    with st.expander("Technical details"):
                        st.code(traceback.format_exc())

if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    st.caption("💡 **Next steps:**")
    followups = suggest_questions(df_cir, column_map)[:3]
    for idx, (col, suggestion) in enumerate(zip(st.columns(len(followups)), followups)):
        with col:
            if st.button(suggestion, key=f"followup_{idx}"):
                st.session_state.pending_prompt = suggestion
                st.rerun()