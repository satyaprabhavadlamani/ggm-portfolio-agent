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
        base_path:      Root folder (e.g. C:\...\GGM Portfolio Analytics)
        year:           Year folder name (e.g. '2026')
        circle_name:    Optional subfolder inside each month folder. 
                        Leave '' when Excel files sit directly in month folder.
        circle_column:  Optional column name to filter by (e.g. 'Circle')
        circle_value:   Value to match in circle_column (e.g. 'Data and Insights')
        log:            Optional list. Messages (skipped files, errors, etc.) 
                        are appended for UI display.

    Returns:
        (circle_wise_df, resource_df) - each has a 'Month' column ('Jul', 'Aug', ...)
        Both are empty DataFrames if nothing is found (never None).

    Notes:
        - Skips Excel lock files (~$...)
        - If multiple files of same type per month, uses the most recent
        - Duplicates and skipped files are logged
    """
    log = log if log is not None else []
    circle_data, resource_data = [], []

    year_path = os.path.join(base_path, str(year))
    if not os.path.isdir(year_path):
        log.append(f"Year folder not found: {year_path}")
        return pd.DataFrame(), pd.DataFrame()

    for month_folder in sorted(os.listdir(year_path)):
        month_path = os.path.join(year_path, month_folder)
        if not os.path.isdir(month_path):
            continue

        scan_path = os.path.join(month_path, circle_name) if circle_name else month_path
        if not os.path.isdir(scan_path):
            log.append(f"{month_folder}: sub-folder '{circle_name}' not found, month skipped")
            continue

        month_abbr = get_month_from_folder_name(month_folder)

        candidates = {'circle': [], 'resource': []}
        for file in os.listdir(scan_path):
            # Skip Excel lock files (~$...) and non-Excel files
            if file.startswith('~$') or not file.lower().endswith(('.xlsx', '.xlsm')):
                continue
            
            category = categorize_excel_file(file)
            if category is None:
                log.append(f"{month_abbr}: skipped '{file}' (name has no circle/financial/resource/headcount)")
                continue
            
            candidates[category].append(os.path.join(scan_path, file))

        for category, paths in candidates.items():
            if not paths:
                continue
            
            # If multiple files of same type, use the most recent
            paths.sort(key=os.path.getmtime, reverse=True)
            for extra in paths[1:]:
                log.append(f"{month_abbr}: skipped older duplicate '{os.path.basename(extra)}' ({category})")

            chosen = paths[0]
            try:
                df = pd.read_excel(chosen, sheet_name=0, header=0)
            except Exception as e:
                log.append(f"{month_abbr}: could not read '{os.path.basename(chosen)}': {e}")
                continue

            # Standardize column names (strip whitespace)
            df.columns = [str(c).strip() for c in df.columns]
            
            # Add month column
            df['Month'] = month_abbr

            # Apply circle filter if specified
            if circle_column and circle_value and circle_column in df.columns:
                before = len(df)
                df = df[df[circle_column].astype(str).str.strip() == circle_value].copy()
                if len(df) == 0:
                    log.append(f"{month_abbr}: '{os.path.basename(chosen)}' had no rows matching {circle_column}='{circle_value}'")
                    continue
                log.append(f"{month_abbr}: filtered {os.path.basename(chosen)} {before} → {len(df)} rows")
            else:
                log.append(f"{month_abbr}: loaded '{os.path.basename(chosen)}' ({len(df)} rows)")

            if category == 'circle':
                circle_data.append(df)
            else:
                resource_data.append(df)

    circle_df = pd.concat(circle_data, ignore_index=True) if circle_data else pd.DataFrame()
    resource_df = pd.concat(resource_data, ignore_index=True) if resource_data else pd.DataFrame()
    
    return circle_df, resource_df