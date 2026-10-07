"""
Local Folder Auto-Discovery Data Loader
Scans folder structure and loads all months automatically.
No Streamlit calls - UI integration happens in the app.
"""

import os
import re
import pandas as pd

MONTHS = [
    ('january', 'Jan'), ('february', 'Feb'), ('march', 'Mar'), ('april', 'Apr'),
    ('may', 'May'), ('june', 'Jun'), ('july', 'Jul'), ('august', 'Aug'),
    ('september', 'Sep'), ('october', 'Oct'), ('november', 'Nov'), ('december', 'Dec'),
]
MONTH_ORDER = [abbr for _, abbr in MONTHS]

# Column added to every loaded frame so the same month in different years never collides.
YEAR_COL = 'Data_Year'


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
        (circle_wise_df, resource_df) - each has 'Month' ('Jul', 'Aug', ...) and
        YEAR_COL ('Data_Year', int) columns.
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

        # Period columns
        df['Month'] = month_abbr
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