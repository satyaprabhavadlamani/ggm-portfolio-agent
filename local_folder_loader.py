"""
Local Folder Auto-Discovery Data Loader
Scans folder structure and loads all months automatically.
No Streamlit calls - UI integration happens in the app.
"""

import os
import re
import numpy as np
import pandas as pd

MONTHS = [
    ('january', 'Jan'), ('february', 'Feb'), ('march', 'Mar'), ('april', 'Apr'),
    ('may', 'May'), ('june', 'Jun'), ('july', 'Jul'), ('august', 'Aug'),
    ('september', 'Sep'), ('october', 'Oct'), ('november', 'Nov'), ('december', 'Dec'),
]
MONTH_ORDER = [abbr for _, abbr in MONTHS]

# Column added to every loaded frame so the same month in different years never collides.
YEAR_COL = 'Data_Year'
# Which file a row came from, and which month that file reports. A monthly file normally holds
# its own month PLUS earlier months, so one month can appear in several files.
SOURCE_COL = 'Source_File'
REPORT_COL = 'Report_Month'


def infer_month_from_text(text):
    """
    Find a month name or 3-letter abbreviation inside free text
    (folder or file name). Returns 'Jan'..'Dec' or None.

    'Resource headcount_Aug.xlsx' -> 'Aug'
    '2026 Circle Wise data - D & L July 2026.xlsx' -> 'Jul'
    'August' -> 'Aug'
    """
    t = str(text).lower()
    for full, abbr in MONTHS:
        for token in (full, abbr.lower()):
            if re.search(rf'(?<![a-z]){token}(?![a-z])', t):
                return abbr
    return None


def find_months_in_text(text, valid_months=None):
    """
    ALL months mentioned in free text (e.g. a chat question), in order of appearance.
    'Compare Aug to July' -> ['Aug', 'Jul']

    valid_months: optional list; only months in it are returned (e.g. the loaded months).
    The word "may" only counts when written 'May' (so "which accounts may churn" is ignored).
    """
    raw = str(text)
    low = raw.lower()
    found = []
    for full, abbr in MONTHS:
        for token in (full, abbr.lower()):
            for m in re.finditer(rf'(?<![a-z]){token}(?![a-z])', low):
                if token == 'may' and raw[m.start():m.end()] != 'May':
                    continue
                found.append((m.start(), abbr))
    found.sort()
    out = []
    for _, abbr in found:
        if abbr not in out:
            out.append(abbr)
    if valid_months is not None:
        out = [a for a in out if a in valid_months]
    return out


def infer_year_from_text(text, default=None):
    """First 4-digit year (2000-2099) in free text, as int; else `default`."""
    m = re.search(r'(?<!\d)(20\d{2})(?!\d)', str(text))
    return int(m.group(1)) if m else default


def apply_circle_filter(df, circle_column, circle_value, label, log):
    """
    Keep rows where circle_column == circle_value (whitespace-trimmed).
    No-op when the column is missing. Used by BOTH the folder loader and manual
    uploads so headcount/revenue are filtered identically in either mode.
    Returns the filtered frame (may be empty); logs before -> after.
    """
    if not (circle_column and circle_value) or circle_column not in df.columns:
        return df
    before = len(df)
    out = df[df[circle_column].astype(str).str.strip() == circle_value].copy()
    log.append(f"{label}: filtered {before} -> {len(out)} rows ({circle_column}='{circle_value}')")
    return out


def resolve_years(base_path, years):
    """
    Turn a year setting into a list of year-folder names.
    '2026' | 2026 | '2025,2026' | ['2025', '2026'] | 'all' (every 4-digit folder)
    """
    if isinstance(years, (list, tuple, set)):
        items = [str(y).strip() for y in years]
    else:
        items = [p.strip() for p in str(years).split(',') if p.strip()]
    if any(i.lower() == 'all' for i in items):
        items = sorted(d for d in os.listdir(base_path)
                       if d.isdigit() and len(d) == 4 and os.path.isdir(os.path.join(base_path, d)))
    return items


def normalize_month_value(v):
    """Month abbreviation ('Jan'..'Dec') from one cell: text ('Aug', 'August', 'Aug-26'),
    a date/Timestamp, or a number 1-12. None when it cannot be read."""
    try:
        if v is None or pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        iv = int(v)
        return MONTH_ORDER[iv - 1] if v == iv and 1 <= iv <= 12 else None
    if hasattr(v, 'month'):                       # datetime / Timestamp
        try:
            return MONTH_ORDER[int(v.month) - 1]
        except (TypeError, ValueError, IndexError):
            return None
    sv = str(v).strip()
    if sv.isdigit():
        iv = int(sv)
        return MONTH_ORDER[iv - 1] if 1 <= iv <= 12 else None
    return infer_month_from_text(sv)


def assign_months(df, report_month, label, log):
    """
    Set df['Month'] from the DATA ITSELF.
    A monthly workbook usually holds that month plus earlier months, so the file/folder name
    must NOT overwrite the Month column. The name is only a fallback (no usable Month column).
    Always adds REPORT_COL = the month the file reports. Modifies df in place; returns df.
    """
    df[REPORT_COL] = report_month
    if 'Month' in df.columns:
        parsed = df['Month'].map(normalize_month_value)
        ok = parsed.notna()
        if ok.any() and ok.mean() >= 0.9:
            bad = int((~ok).sum())
            if bad:
                log.append(f"{label}: {bad} row(s) have an unreadable Month value - counted as {report_month}")
            df['Month'] = parsed.where(ok, report_month)
            present = [m for m in MONTH_ORDER if m in set(df['Month'])]
            if report_month not in MONTH_ORDER and present:      # name had no month: file reports its latest month
                df[REPORT_COL] = present[-1]
                report_month = present[-1]
            if present != [report_month]:
                log.append(f"{label}: file holds months {', '.join(present)} - the file's own Month column is used")
            return df
        log.append(f"{label}: 'Month' column not readable ({int(ok.sum())}/{len(df)} rows) - using file month {report_month}")
    df['Month'] = report_month
    return df


def select_authoritative(df):
    """
    One month can sit in several files (the Aug file also holds Jan-Jul). The DEFAULT source
    of a month is the file that REPORTS that month (August -> the August file). If no such
    file is loaded, the earliest LATER file that contains it is used.
    Returns (default_rows, other_rows). other_rows = same months found in the other files,
    kept only so differences can be shown - they are never mixed into the default numbers.
    """
    if df is None or df.empty or not {'Month', REPORT_COL}.issubset(df.columns):
        return df, pd.DataFrame()
    idx = {m: i for i, m in enumerate(MONTH_ORDER)}
    d = (df[REPORT_COL].map(idx) - df['Month'].map(idx)).fillna(0)     # 0 = same month, >0 = later file
    rank = np.where(d == 0, 0, np.where(d > 0, d, 100 - d))
    keys = [k for k in (YEAR_COL, 'Month') if k in df.columns]
    tmp = df.assign(_rank=rank)
    best = tmp.groupby(keys)['_rank'].transform('min')
    keep = (tmp['_rank'] == best).to_numpy()
    return df[keep].copy(), df[~keep].copy()


def month_mismatches(auth, other, revenue_col, cost_col=None, profit_col=None, client_col=None,
                     tol=0.0005, top_clients=5):
    """
    Compare each month's default numbers with the same month in any other file.
    Returns (month_df, client_df); both empty when everything agrees (tolerance 0.0005 USD m).
    Both files' numbers are kept side by side (Default_* / Other_*), never blended.
    """
    empty = (pd.DataFrame(), pd.DataFrame())
    need = {SOURCE_COL, REPORT_COL, 'Month'}
    if auth is None or other is None or auth.empty or other.empty:
        return empty
    if not (need.issubset(auth.columns) and need.issubset(other.columns)):
        return empty
    measures = {'Revenue': revenue_col, 'Cost': cost_col, 'GP': profit_col}
    measures = {k: v for k, v in measures.items() if v and v in auth.columns and v in other.columns}
    if not measures:
        return empty
    keys = [k for k in (YEAR_COL, 'Month') if k in auth.columns and k in other.columns]

    def _tot(df, extra):
        agg = {lab: (col, 'sum') for lab, col in measures.items()}
        agg['File'] = (SOURCE_COL, 'first')
        agg['Report'] = (REPORT_COL, 'first')
        return df.groupby(keys + extra, dropna=False).agg(**agg).reset_index()

    a_t, o_t = _tot(auth, []), _tot(other, [REPORT_COL])
    m = o_t.drop(columns=[REPORT_COL]).merge(a_t, on=keys, suffixes=('_Other', '_Default'))
    if m.empty:
        return empty
    mask = np.zeros(len(m), dtype=bool)
    for lab in measures:
        m[f'{lab}_Diff'] = m[f'{lab}_Other'] - m[f'{lab}_Default']
        mask |= (m[f'{lab}_Diff'].abs().fillna(0) > tol).to_numpy()
    m = m[mask].copy()
    if m.empty:
        return empty
    m['Default_Source'] = m['Report_Default'].astype(str) + ' file'
    m['Other_Source'] = m['Report_Other'].astype(str) + ' file'
    m['_o'] = m['Month'].map(lambda x: MONTH_ORDER.index(x) if x in MONTH_ORDER else 999)
    m = m.sort_values(['_o', 'Other_Source']).drop(columns='_o')
    cols = (['Month', 'Default_Source', 'Other_Source'] +
            [f'{lab}_{sfx}' for lab in measures for sfx in ('Default', 'Other', 'Diff')] +
            ['File_Default', 'File_Other'])
    month_df = m[cols].reset_index(drop=True)

    client_df = pd.DataFrame()
    if client_col and revenue_col and client_col in auth.columns and client_col in other.columns:
        def _slice(df, row, rep=None):
            sel = np.ones(len(df), dtype=bool)
            for k in keys:
                sel &= (df[k] == row[k]).to_numpy()
            if rep is not None:
                sel &= (df[REPORT_COL] == rep).to_numpy()
            return df[sel]
        out = []
        for _, r in m.iterrows():
            a = _slice(auth, r).groupby(client_col)[revenue_col].sum().rename('Revenue_Default')
            o = _slice(other, r, r['Report_Other']).groupby(client_col)[revenue_col].sum().rename('Revenue_Other')
            c = pd.concat([a, o], axis=1).fillna(0.0)
            c['Revenue_Diff'] = c['Revenue_Other'] - c['Revenue_Default']
            c = c[c['Revenue_Diff'].abs() > tol]
            if c.empty:
                continue
            c = c.reindex(c['Revenue_Diff'].abs().sort_values(ascending=False).index).head(top_clients)
            c.index.name = 'Client'
            c = c.reset_index()
            c.insert(0, 'Month', r['Month'])
            c.insert(2, 'Default_Source', r['Default_Source'])
            c.insert(3, 'Other_Source', r['Other_Source'])
            out.append(c)
        if out:
            client_df = pd.concat(out, ignore_index=True)
    return month_df, client_df


def categorize_excel_file(filename):
    """
    Determine file type by name pattern.

    Returns:
        'circle'   - Circle Wise data (contains 'circle' or 'financial')
        'resource' - Resource/Headcount data (contains 'resource' or 'headcount')
        None       - Unknown
    """
    filename_lower = filename.lower()

    if 'circle' in filename_lower or 'financial' in filename_lower:
        return 'circle'
    elif 'resource' in filename_lower or 'headcount' in filename_lower:
        return 'resource'

    return None


def get_month_from_folder_name(folder_name):
    """
    Extract month abbreviation from a folder name.

    'August' -> 'Aug', 'July' -> 'Jul', 'Sep 2026' -> 'Sep'
    Falls back to the first 3 characters if no month is recognized.
    """
    return infer_month_from_text(folder_name) or str(folder_name)[:3].title()


def get_available_months(df):
    """
    Extract available months from dataframe, in calendar order.
    
    Returns sorted list: ['Jul', 'Aug', 'Sep', ...]
    """
    if df is None or df.empty or 'Month' not in df.columns:
        return []

    months = df['Month'].dropna().unique().tolist()
    return sorted(months, key=lambda x: MONTH_ORDER.index(x) if x in MONTH_ORDER else 999)


def load_monthly_data_from_folder(base_path, year, circle_name='', circle_column=None,
                                   circle_value=None, log=None):
    """
    Load Circle Wise and Resource data from monthly folders.

    Folder structure scanned:
        base_path/<year>/<Month folder>/[circle_subfolder/]*.xlsx

    Args:
        base_path:      Root folder (e.g. C:/Users/.../GGM Portfolio Analytics)
        year:           Year folder name ('2026'), several ('2025,2026' or a list), or 'all'
        circle_name:    Optional subfolder inside each month folder.
                        Leave '' when Excel files sit directly in month folder.
        circle_column:  Optional column name to filter by (e.g. 'Circle')
        circle_value:   Value to match in circle_column (e.g. 'Data and Insights')
        log:            Optional list. Messages (skipped files, errors, etc.)
                        are appended for UI display.

    Returns:
        (circle_wise_df, resource_df) - each has 'Month' ('Jul', 'Aug', ... taken from the
        file's own Month column), Report_Month (month the file reports), Source_File and
        Data_Year (int) columns.
        Both are empty DataFrames if nothing is found (never None).

    Notes:
        - Skips Excel lock files (~$...)
        - Month folders whose name is not a recognisable month are SKIPPED (logged),
          never turned into a fake month.
        - If several files compete for the same (type, year, month) - even across
          different folders such as 'August' and 'Aug_old' - only the most recent
          file is used, so a month is never double-counted. Skipped files are logged.
    """
    log = log if log is not None else []

    if not os.path.isdir(base_path):
        log.append(f"Base folder not found: {base_path}")
        return pd.DataFrame(), pd.DataFrame()

    years = resolve_years(base_path, year)
    if not years:
        log.append(f"No year folders to scan (setting: {year!r})")
        return pd.DataFrame(), pd.DataFrame()

    # ---- Pass 1: discover candidate files, keyed by (category, year, month) ----
    candidates = {}
    for yr in years:
        year_path = os.path.join(base_path, str(yr))
        if not os.path.isdir(year_path):
            log.append(f"Year folder not found: {year_path}")
            continue

        for month_folder in sorted(os.listdir(year_path)):
            month_path = os.path.join(year_path, month_folder)
            if not os.path.isdir(month_path):
                continue

            month_abbr = infer_month_from_text(month_folder)
            if month_abbr is None:
                log.append(f"{yr}/{month_folder}: folder name is not a recognisable month, skipped")
                continue

            scan_path = os.path.join(month_path, circle_name) if circle_name else month_path
            if not os.path.isdir(scan_path):
                log.append(f"{yr}/{month_folder}: sub-folder '{circle_name}' not found, month skipped")
                continue

            for file in os.listdir(scan_path):
                # Skip Excel lock files (~$...) and non-Excel files
                if file.startswith('~$') or not file.lower().endswith(('.xlsx', '.xlsm')):
                    continue

                category = categorize_excel_file(file)
                if category is None:
                    log.append(f"{month_abbr} {yr}: skipped '{file}' "
                               f"(name has no circle/financial/resource/headcount)")
                    continue

                full = os.path.join(scan_path, file)
                try:
                    mtime = os.path.getmtime(full)
                except OSError:
                    mtime = 0.0
                candidates.setdefault((category, yr, month_abbr), []).append((mtime, full, month_folder))

    # ---- Pass 2: one file per (category, year, month), then read it ----
    def _sort_key(k):
        category, yr, month_abbr = k
        return (str(yr), MONTH_ORDER.index(month_abbr) if month_abbr in MONTH_ORDER else 999, category)

    circle_data, resource_data = [], []
    for key in sorted(candidates, key=_sort_key):
        category, yr, month_abbr = key
        label = f"{month_abbr} {yr}"
        entries = sorted(candidates[key], reverse=True)          # newest first
        chosen = entries[0][1]
        for _, extra_path, extra_folder in entries[1:]:
            log.append(f"{label}: skipped older duplicate '{extra_folder}/{os.path.basename(extra_path)}' "
                       f"({category}) - using '{entries[0][2]}/{os.path.basename(chosen)}'")

        try:
            # Read by sheet name "Sheet1" (the source of truth), not by index.
            # This ensures we always get the base data, never pivot tables or other sheets.
            try:
                df = pd.read_excel(chosen, sheet_name='Sheet1', header=0)
            except ValueError:
                log.append(f"{label}: sheet 'Sheet1' not found in '{os.path.basename(chosen)}', trying first sheet...")
                df = pd.read_excel(chosen, sheet_name=0, header=0)
        except Exception as e:
            log.append(f"{label}: could not read '{os.path.basename(chosen)}': {e}")
            continue

        # Standardize column names (strip whitespace)
        df.columns = [str(c).strip() for c in df.columns]

        # Period columns. Month comes from the DATA (a monthly file also holds earlier months);
        # the file/folder month is only the file's REPORT month and the fallback.
        assign_months(df, month_abbr, label, log)
        df[SOURCE_COL] = os.path.basename(chosen)
        df[YEAR_COL] = int(yr) if str(yr).isdigit() else str(yr)

        # Apply circle filter if specified (same helper the upload path uses)
        if circle_column and circle_value and circle_column in df.columns:
            df = apply_circle_filter(df, circle_column, circle_value,
                                     f"{label} '{os.path.basename(chosen)}'", log)
            if len(df) == 0:
                log.append(f"{label}: '{os.path.basename(chosen)}' had no rows matching "
                           f"{circle_column}='{circle_value}'")
                continue
        else:
            log.append(f"{label}: loaded '{os.path.basename(chosen)}' ({len(df)} rows)")

        (circle_data if category == 'circle' else resource_data).append(df)

    circle_df = pd.concat(circle_data, ignore_index=True) if circle_data else pd.DataFrame()
    resource_df = pd.concat(resource_data, ignore_index=True) if resource_data else pd.DataFrame()

    return circle_df, resource_df