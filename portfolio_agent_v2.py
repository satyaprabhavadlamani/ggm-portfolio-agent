import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="GGM D&I Portfolio Agent", layout="wide")
st.title("📊 GGM D&I Portfolio Agent")

# ===== CONFIGURATION =====
GGM_REGIONS = ['India', 'Philippines', 'UK', 'Singapore']
GGM_PRACTICE = 'Data and Insights'

# ===== LOAD CURRENT MONTH DATA =====
st.sidebar.header("📁 Data Source")
data_source = st.sidebar.radio("Choose source:", ["Upload Files", "OneDrive"])

df_res = None
df_cir = None
data_loaded = False

if data_source == "Upload Files":
    resource_file = st.sidebar.file_uploader("Resource Data ", type=['xlsx'])
    circle_file = st.sidebar.file_uploader("Circle Wise Data", type=['xlsx'])
    
    if resource_file and circle_file:
        try:
            df_res = pd.read_excel(resource_file, sheet_name=0, header=0)
            df_cir = pd.read_excel(circle_file, sheet_name='Sheet1', header=0)
            
            # Filter
            df_res = df_res[(df_res['Region'].isin(GGM_REGIONS)) & (df_res['Practices'] == GGM_PRACTICE)]
            df_cir = df_cir[df_cir['Region'].isin(GGM_REGIONS)]
            
            data_loaded = True
            st.sidebar.success("✅ Files loaded")
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")

else:  # OneDrive
    try:
        resource_url = st.secrets["onedrive"]["resource_file_url"]
        circle_url = st.secrets["onedrive"]["circle_file_url"]
        
        df_res = pd.read_excel(resource_url, sheet_name=0, header=0)
        df_cir = pd.read_excel(circle_url, sheet_name='Sheet1', header=0)
        
        df_res = df_res[(df_res['Region'].isin(GGM_REGIONS)) & (df_res['Practices'] == GGM_PRACTICE)]
        df_cir = df_cir[df_cir['Region'].isin(GGM_REGIONS)]
        
        data_loaded = True
        st.sidebar.success("✅ OneDrive loaded")
    except Exception as e:
        st.sidebar.error(f"OneDrive error: {str(e)}\n\nAdd URLs to ~/.streamlit/secrets.toml")

# ===== LOAD HISTORICAL DATA =====
@st.cache_data
def load_historical():
    try:
        hist_url = st.secrets["onedrive"]["historical_file_url"]
        df = pd.read_excel(hist_url)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except:
        return pd.DataFrame()

df_hist = load_historical()

# ===== MAIN DASHBOARD =====
if data_loaded:
    # Current month metrics
    total_rev = df_cir['Revenue USD m'].sum()
    total_gp = df_cir['GP USD m'].sum()
    gpm = (total_gp / total_rev * 100) if total_rev > 0 else 0
    hc = len(df_res)
    num_accounts = df_cir['Client'].nunique()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Revenue", f"${total_rev:.3f}M")
    col2.metric("Profit Margin", f"{gpm:.2f}%")
    col3.metric("Headcount", hc)
    col4.metric("Active Accounts", num_accounts)
    
    # ===== MTD / QTD / YTD SECTION =====
    if not df_hist.empty:
        st.divider()
        st.subheader("📈 Period Comparisons")
        
        today = datetime.now()
        current_month = today.month
        current_year = today.year
        current_quarter = (current_month - 1) // 3 + 1
        
        # MTD
        mtd_data = df_hist[(df_hist['date'].dt.month == current_month) & (df_hist['date'].dt.year == current_year)]
        mtd_rev = mtd_data['revenue_amount'].sum()
        mtd_gp = mtd_data['gp_amount'].sum()
        mtd_gpm = (mtd_gp / mtd_rev * 100) if mtd_rev > 0 else 0
        
        # QTD
        qtd_data = df_hist[(df_hist['date'].dt.quarter == current_quarter) & (df_hist['date'].dt.year == current_year)]
        qtd_rev = qtd_data['revenue_amount'].sum()
        qtd_gp = qtd_data['gp_amount'].sum()
        qtd_gpm = (qtd_gp / qtd_rev * 100) if qtd_rev > 0 else 0
        
        # YTD
        ytd_data = df_hist[df_hist['date'].dt.year == current_year]
        ytd_rev = ytd_data['revenue_amount'].sum()
        ytd_gp = ytd_data['gp_amount'].sum()
        ytd_gpm = (ytd_gp / ytd_rev * 100) if ytd_rev > 0 else 0
        
        # MoM comparison
        prev_month_data = df_hist[(df_hist['date'].dt.month == current_month - 1) & (df_hist['date'].dt.year == current_year)]
        prev_rev = prev_month_data['revenue_amount'].sum()
        mom_growth = ((mtd_rev - prev_rev) / prev_rev * 100) if prev_rev > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("MTD Revenue", f"${mtd_rev:.3f}M", f"{mom_growth:+.1f}% MoM")
            st.metric("MTD GPM", f"{mtd_gpm:.2f}%")
        
        with col2:
            st.metric("QTD Revenue", f"${qtd_rev:.3f}M")
            st.metric("QTD GPM", f"{qtd_gpm:.2f}%")
        
        with col3:
            st.metric("YTD Revenue", f"${ytd_rev:.3f}M")
            st.metric("YTD GPM", f"{ytd_gpm:.2f}%")
        
        # Trend chart
        monthly_trend = df_hist.groupby('date').agg({
            'revenue_amount': 'sum',
            'gp_amount': 'sum'
        }).reset_index()
        monthly_trend['gpm'] = (monthly_trend['gp_amount'] / monthly_trend['revenue_amount'] * 100).round(2)
        monthly_trend = monthly_trend.sort_values('date')
        
        fig = px.line(monthly_trend, x='date', y='revenue_amount', 
                     title='Revenue Trend (All Months)', markers=True,
                     labels={'revenue_amount': 'Revenue ($M)', 'date': 'Month'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Regional breakdown
        by_region = df_hist.groupby('region').agg({
            'revenue_amount': 'sum',
            'gp_amount': 'sum'
        }).reset_index()
        by_region['gpm_pct'] = (by_region['gp_amount'] / by_region['revenue_amount'] * 100).round(2)
        by_region = by_region.sort_values('revenue_amount', ascending=False)
        
        st.subheader("By Region (YTD)")
        st.dataframe(by_region, use_container_width=True, hide_index=True)
    
    # ===== CURRENT MONTH BREAKDOWN =====
    st.divider()
    st.subheader("💼 Current Month Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Revenue by Region**")
        by_reg = df_cir.groupby('Region')['Revenue USD m'].sum().sort_values(ascending=False)
        fig1 = px.bar(by_reg, title='Revenue by Region', labels={'value': 'Revenue ($M)'})
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.write("**Top 10 Accounts**")
        by_acct = df_cir.groupby('Client')['Revenue USD m'].sum().sort_values(ascending=False).head(10)
        fig2 = px.bar(by_acct, title='Top 10 Accounts', labels={'value': 'Revenue ($M)'})
        st.plotly_chart(fig2, use_container_width=True)
    
    # ===== NLP AGENT =====
    st.divider()
    st.subheader("💬 Ask Questions")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Ask about revenue, accounts, regions, MTD/QTD/YTD..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Build context
                    by_acct_top = df_cir.groupby('Client')['Revenue USD m'].sum().sort_values(ascending=False).head(5)
                    by_reg = df_cir.groupby('Region')['Revenue USD m'].sum()
                    
                    context = f"""
CURRENT MONTH:
- Total Revenue: ${total_rev:.3f}M
- Profit Margin: {gpm:.2f}%
- Headcount: {hc}
- Active Accounts: {num_accounts}

Top 5 Accounts:
{by_acct_top.round(3).to_string()}

By Region:
{by_reg.round(3).to_string()}

YTD SUMMARY:
- YTD Revenue: ${ytd_rev:.3f}M (GPM: {ytd_gpm:.2f}%)
- QTD Revenue: ${qtd_rev:.3f}M (GPM: {qtd_gpm:.2f}%)
- MTD Revenue: ${mtd_rev:.3f}M (GPM: {mtd_gpm:.2f}%, Growth: {mom_growth:+.1f}% MoM)
"""
                    
                    payload = {
                        "model": "mistral",
                        "prompt": f"{context}\n\nQuestion: {prompt}\n\nProvide a concise, data-driven answer.\n\nAnswer:",
                        "stream": False
                    }
                    resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
                    
                    if resp.status_code == 200:
                        answer = resp.json().get('response', 'No response')
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.markdown(answer)
                    else:
                        st.error("Ollama error")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Ensure `ollama serve` is running on localhost:11434")

else:
    st.info("📁 Upload Excel files or configure OneDrive in secrets.toml")