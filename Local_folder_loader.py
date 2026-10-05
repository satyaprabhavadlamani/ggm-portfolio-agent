"""
Local Folder Auto-Discovery Data Loader
Scans folder structure and loads all months automatically
Path: C:\Users\satyaprabha.v\OneDrive - ascendion\Documents\GGM Data\GGM Portfolio Analytics
"""

import os
import pandas as pd
import streamlit as st
from pathlib import Path

# Map folder names to month abbreviations
MONTH_MAP = {
    'january': 'Jan', 'february': 'Feb', 'march': 'Mar', 'april': 'Apr',
    'may': 'May', 'june': 'Jun', 'july': 'Jul', 'august': 'Aug',
    'september': 'Sep', 'october': 'Oct', 'november': 'Nov', 'december': 'Dec',
}

def get_month_from_folder_name(folder_name):
    """Extract month from folder name: 'August' → 'Aug', 'July' → 'Jul'"""
    folder_lower = folder_name.lower()
    for key, value in MONTH_MAP.items():
        if key in folder_lower:
            return value
    return None

def find_excel_files_in_folder(folder_path):
    """Find all Excel files in a folder"""
    excel_files = []
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(('.xlsx', '.xls')):
                excel_files.append(os.path.join(folder_path, file))
    return excel_files

def categorize_excel_file(filename):
    """Categorize file by name pattern: circle, resource, or unknown"""
    filename_lower = filename.lower()
    
    if 'circle' in filename_lower or 'financial' in filename_lower:
        return 'circle'
    elif 'resource' in filename_lower or 'headcount' in filename_lower:
        return 'resource'
    else:
        return 'unknown'

def load_monthly_data_from_folder(base_path, year='2026', circle_name='Data and Insights'):
    """
    Scan folder structure and load all months
    
    Path structure:
    C:\...\GGM Portfolio Analytics\
        ├── 2026\
        │   ├── August\
        │   │   ├── Circle_Wise_*.xlsx
        │   │   ├── Resource_*.xlsx
        │   ├── July\
        │   │   ├── Circle_Wise_*.xlsx
        │   │   ├── Resource_*.xlsx
        │   └── ...
    """
    
    year_path = os.path.join(base_path, year)
    
    if not os.path.exists(year_path):
        st.error(f"❌ Path not found: {year_path}")
        return None, None
    
    # Find all month folders
    month_folders = []
    for item in os.listdir(year_path):
        item_path = os.path.join(year_path, item)
        if os.path.isdir(item_path):
            month_folders.append((item, item_path))
    
    if not month_folders:
        st.error(f"❌ No month folders found in {year_path}")
        return None, None
    
    st.write(f"📁 Found {len(month_folders)} month folders")
    
    # Load circle wise data from all months
    circle_data = []
    resource_data = []
    
    for month_folder, month_path in sorted(month_folders):
        month_abbr = get_month_from_folder_name(month_folder)
        
        if not month_abbr:
            st.write(f"⏭️ Skipping {month_folder} (couldn't extract month)")
            continue
        
        st.write(f"📂 Loading {month_folder} ({month_abbr})...")
        
        # Find Excel files in this month's folder
        excel_files = find_excel_files_in_folder(month_path)
        
        if not excel_files:
            st.write(f"  ⚠️ No Excel files in {month_folder}")
            continue
        
        # Load files by category
        for file_path in excel_files:
            filename = os.path.basename(file_path)
            file_category = categorize_excel_file(filename)
            
            if file_category == 'circle':
                try:
                    df = pd.read_excel(file_path, sheet_name=0, header=0)
                    
                    # Filter to circle
                    if 'Circle' in df.columns:
                        df = df[df['Circle'] == circle_name]
                    
                    # Add month
                    if 'Month' not in df.columns:
                        df['Month'] = month_abbr
                    
                    if len(df) > 0:
                        circle_data.append(df)
                        st.write(f"  ✓ {filename}: {len(df)} records")
                
                except Exception as e:
                    st.write(f"  ❌ Error reading {filename}: {str(e)}")
            
            elif file_category == 'resource':
                try:
                    df = pd.read_excel(file_path, sheet_name=0, header=0)
                    
                    # Filter to circle
                    if 'Practices' in df.columns:
                        df = df[df['Practices'] == circle_name]
                    
                    # Add month
                    if 'Month' not in df.columns:
                        df['Month'] = month_abbr
                    
                    if len(df) > 0:
                        resource_data.append(df)
                        st.write(f"  ✓ {filename}: {len(df)} records")
                
                except Exception as e:
                    st.write(f"  ❌ Error reading {filename}: {str(e)}")
    
    # Consolidate all data
    df_circle_consolidated = None
    df_resource_consolidated = None
    
    if circle_data:
        df_circle_consolidated = pd.concat(circle_data, ignore_index=True)
        st.success(f"✅ Circle Wise: {len(df_circle_consolidated)} total records from {len(set(df_circle_consolidated['Month']))} months")
    else:
        st.warning("⚠️ No Circle Wise data found")
    
    if resource_data:
        df_resource_consolidated = pd.concat(resource_data, ignore_index=True)
        st.success(f"✅ Resource: {len(df_resource_consolidated)} total records from {len(set(df_resource_consolidated['Month']))} months")
    else:
        st.warning("⚠️ No Resource data found")
    
    return df_circle_consolidated, df_resource_consolidated

def get_available_months(df):
    """Get sorted list of available months"""
    if df is None or 'Month' not in df.columns:
        return []
    
    month_order = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    
    months = sorted(df['Month'].unique(), key=lambda x: month_order.get(x, 99))
    return months

# Example usage in Streamlit
if __name__ == "__main__":
    # Test
    base_path = r"C:\Users\satyaprabha.v\OneDrive - ascendion\Documents\GGM Data\GGM Portfolio Analytics"
    df_circle, df_resource = load_monthly_data_from_folder(base_path)
    
    if df_circle is not None:
        print(f"Circle data: {len(df_circle)} records")
        print(f"Months: {get_available_months(df_circle)}")
    
    if df_resource is not None:
        print(f"Resource data: {len(df_resource)} records")