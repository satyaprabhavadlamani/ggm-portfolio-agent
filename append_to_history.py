"""
Monthly script to append current month data to historical_consolidated.xlsx
Run this after generating the current month's Resource_Data_GGM & Circle_Wise_data files

Usage:
    python append_to_history.py --month 9 --year 2026 --resource Resource_Data_GGM_2026_09.xlsx --circle Circle_Wise_data_2026_09.xlsx
"""

import pandas as pd
from datetime import datetime
import argparse
import os

def append_to_history(resource_file, circle_file, month, year):
    """Append current month data to historical file"""
    
    # Load current month data
    print(f"Loading {month}/{year} data...")
    df_res = pd.read_excel(resource_file, sheet_name=0, header=0)
    df_fin = pd.read_excel(circle_file, sheet_name='Sheet1', header=0)
    
    # Filters
    ggm_regions = ['India', 'Philippines', 'UK', 'Singapore']
    df_res = df_res[(df_res['Region'].isin(ggm_regions)) & (df_res['Practices'] == 'Data and Insights')]
    df_fin = df_fin[df_fin['Region'].isin(ggm_regions)]
    
    # Aggregate financial data by account
    df_summary = df_fin[['Client', 'Region', 'Revenue USD m', 'GP USD m', 'Cost USD m']].copy()
    df_summary.columns = ['account_name', 'region', 'revenue_amount', 'gp_amount', 'cost_amount']
    
    # Calculate GPM
    df_summary['gpm_pct'] = (df_summary['gp_amount'] / df_summary['revenue_amount'] * 100).round(2)
    
    # Add headcount per account
    hc_by_account = df_res.groupby('Client Name').size().reset_index(name='headcount')
    hc_by_account.columns = ['account_name', 'headcount']
    df_summary = df_summary.merge(hc_by_account, on='account_name', how='left')
    df_summary['headcount'] = df_summary['headcount'].fillna(0).astype(int)
    
    # Add date columns
    # Use last day of month
    if month == 12:
        end_date = pd.Timestamp(year + 1, 1, 1) - pd.Timedelta(days=1)
    else:
        end_date = pd.Timestamp(year, month + 1, 1) - pd.Timedelta(days=1)
    
    df_summary['date'] = end_date
    df_summary['month'] = f"{month:02d}"
    df_summary['year'] = year
    
    print(f"✅ Loaded {len(df_summary)} accounts for {month}/{year}")
    
    # Load historical file (or create if doesn't exist)
    hist_file = 'historical_consolidated.xlsx'
    if os.path.exists(hist_file):
        print(f"Loading existing {hist_file}...")
        df_hist = pd.read_excel(hist_file)
        df_hist['date'] = pd.to_datetime(df_hist['date'])
    else:
        print(f"Creating new {hist_file}...")
        df_hist = pd.DataFrame()
    
    # Append new data
    df_hist = pd.concat([df_hist, df_summary], ignore_index=True)
    
    # Remove duplicates (keep last entry per month/account)
    df_hist = df_hist.sort_values('date')
    df_hist = df_hist.drop_duplicates(subset=['date', 'account_name', 'region'], keep='last')
    
    # Save
    df_hist.to_excel(hist_file, index=False)
    print(f"✅ Updated {hist_file}")
    print(f"   Total months: {df_hist['month'].nunique()}")
    print(f"   Total records: {len(df_hist)}")
    print(f"   Date range: {df_hist['date'].min().date()} to {df_hist['date'].max().date()}")
    
    return df_hist

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Append monthly data to historical file')
    parser.add_argument('--month', type=int, required=True, help='Month (1-12)')
    parser.add_argument('--year', type=int, required=True, help='Year (e.g., 2026)')
    parser.add_argument('--resource', type=str, required=True, help='Resource data file path')
    parser.add_argument('--circle', type=str, required=True, help='Circle wise data file path')
    
    args = parser.parse_args()
    
    # Validate
    if not (1 <= args.month <= 12):
        print("Error: Month must be 1-12")
        exit(1)
    
    if not os.path.exists(args.resource):
        print(f"Error: {args.resource} not found")
        exit(1)
    
    if not os.path.exists(args.circle):
        print(f"Error: {args.circle} not found")
        exit(1)
    
    append_to_history(args.resource, args.circle, args.month, args.year)