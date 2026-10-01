import streamlit as st
import pandas as pd
import requests
from datetime import datetime

st.set_page_config(page_title="GGM D&I Portfolio Agent", layout="wide")
st.title("📊 GGM D&I Portfolio Agent")

# ===== LOAD FROM ONEDRIVE =====
@st.cache_data
def load_from_onedrive():
    """
    Uses Microsoft 365 API to fetch files from OneDrive
    Assumes: /GGM Portfolio Analytics/2026/[Month]/files
    """
    try:
        # Files from current month folder (you upload manually)
        resource_file = st.secrets["onedrive"]["resource_file_url"]  # OneDrive share link
        circle_file = st.secrets["onedrive"]["circle_file_url"]
        
        df_res = pd.read_excel(resource_file, sheet_name=0, header=2)
        df_cir = pd.read_excel(circle_file, sheet_name='Sheet1', header=0)
        
        regions = ['India', 'Philippines', 'UK', 'Singapore']
        df_res = df_res[(df_res['Region'].isin(regions)) & (df_res['Practices'] == 'Data and Insights')]
        df_cir = df_cir[df_cir['Region'].isin(regions)]
        
        return df_res, df_cir, True
    except Exception as e:
        st.sidebar.error(f"Error loading: {str(e)}")
        return None, None, False

@st.cache_data
def load_historical():
    """Load historical consolidated data from OneDrive"""
    try:
        hist_file = st.secrets["onedrive"]["historical_file_url"]
        df = pd.read_excel(hist_file)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except:
        return pd.DataFrame()

# ===== SIDEBAR: FILE UPLOAD ALTERNATIVE =====
st.sidebar.header("📁 Or Upload Files")
use_upload = st.sidebar.checkbox("Upload manually instead of OneDrive")

if use_upload:
    resource_file = st.sidebar.file_uploader("Resource Data", type=['xlsx'])
    circle_file = st.sidebar.file_uploader("Circle Wise Data", type=['xlsx'])
    
    if resource_file and circle_file:
        df_res = pd.read_excel(resource_file, sheet_name=0, header=2)
        df_cir = pd.read_excel(circle_file, sheet_name='Sheet1', header=0)
        
        regions = ['India', 'Philippines', 'UK', 'Singapore']
        df_res = df_res[(df_res['Region'].isin(regions)) & (df_res['Practices'] == 'Data and Insights')]
        df_cir = df_cir[df_cir['Region'].isin(regions)]
        data_loaded = True
else:
    df_res, df_cir, data_loaded = load_from_onedrive()

if data_loaded:
    # ===== CURRENT MONTH METRICS =====
    col1, col2, col3, col4 = st.columns(4)
    
    total_rev = df_cir['Revenue USD m'].sum()
    total_gp = df_cir['GP USD m'].sum()
    gpm = (total_gp / total_rev * 100) if total_rev > 0 else 0
    hc = len(df_res)
    
    col1.metric("Current Revenue", f"${total_rev:.3f}M")
    col2.metric("Profit Margin", f"{gpm:.2f}%")
    col3.metric("Headcount", hc)
    col4.metric("Accounts", df_cir['Client'].nunique())
    
    # ===== LOAD HISTORICAL DATA =====
    df_hist = load_historical()
    
    if not df_hist.empty:
        st.divider()
        st.subheader("📈 MTD | QTD | YTD")
        
        today = datetime.now()
        current_month = today.month
        current_year = today.year
        current_quarter = (current_month - 1) // 3 + 1
        
        # MTD
        mtd = df_hist[(df_hist['date'].dt.month == current_month) & (df_hist['date'].dt.year == current_year)]
        mtd_rev = mtd['revenue_amount'].sum()
        mtd_gpm = (mtd['gp_amount'].sum() / mtd_rev * 100) if mtd_rev > 0 else 0
        
        # QTD
        qtd = df_hist[(df_hist['date'].dt.quarter == current_quarter) & (df_hist['date'].dt.year == current_year)]
        qtd_rev = qtd['revenue_amount'].sum()
        qtd_gpm = (qtd['gp_amount'].sum() / qtd_rev * 100) if qtd_rev > 0 else 0
        
        # YTD
        ytd = df_hist[df_hist['date'].dt.year == current_year]
        ytd_rev = ytd['revenue_amount'].sum()
        ytd_gpm = (ytd['gp_amount'].sum() / ytd_rev * 100) if ytd_rev > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MTD Revenue", f"${mtd_rev:.3f}M")
            st.metric("MTD GPM", f"{mtd_gpm:.2f}%")
        with col2:
            st.metric("QTD Revenue", f"${qtd_rev:.3f}M")
            st.metric("QTD GPM", f"{qtd_gpm:.2f}%")
        with col3:
            st.metric("YTD Revenue", f"${ytd_rev:.3f}M")
            st.metric("YTD GPM", f"{ytd_gpm:.2f}%")
        
        # Trend chart
        import plotly.express as px
        monthly_trend = df_hist.groupby('date').agg({
            'revenue_amount': 'sum',
            'gp_amount': 'sum'
        }).reset_index()
        monthly_trend['gpm'] = (monthly_trend['gp_amount'] / monthly_trend['revenue_amount'] * 100)
        
        fig = px.line(monthly_trend, x='date', y='revenue_amount', 
                     title='Revenue Trend', markers=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # ===== AGENT CHAT =====
    st.divider()
    st.subheader("💬 Ask Questions")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Ask about revenue, accounts, regions..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    by_acct = df_cir.groupby('Client')['Revenue USD m'].sum().sort_values(ascending=False).head(5)
                    by_reg = df_cir.groupby('Region')['Revenue USD m'].sum()
                    
                    context = f"""
Current Month Data:
- Total Revenue: ${total_rev:.3f}M
- Profit Margin: {gpm:.2f}%
- Headcount: {hc}

Top 5 Accounts:
{by_acct.round(3).to_string()}

By Region:
{by_reg.round(3).to_string()}

Historical YTD: ${ytd_rev:.3f}M (GPM: {ytd_gpm:.2f}%)
"""
                    
                    payload = {
                        "model": "mistral",
                        "prompt": f"{context}\n\nQuestion: {prompt}\n\nAnswer:",
                        "stream": False
                    }
                    resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
                    answer = resp.json().get('response', 'No response') if resp.status_code == 200 else "Error"
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Ensure ollama serve is running")
else:
    st.info("📁 Upload Excel files or configure OneDrive URLs in secrets.toml")