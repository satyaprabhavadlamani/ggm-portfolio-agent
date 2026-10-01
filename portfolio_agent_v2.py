import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="GGM D&I Portfolio Agent", layout="wide")
st.title("📊 GGM D&I Portfolio Agent")

# ===== CONFIGURATION =====
GGM_REGIONS = ['India', 'Philippines', 'UK', 'Singapore']
GGM_PRACTICE = 'Data and Insights'

# ===== COLUMN SYNONYMS FOR FLEXIBLE QUERIES =====
COLUMN_SYNONYMS = {
    "employee": ["Emp No", "Employee ID", "EmpNo"],
    "name": ["Candidate Name", "Employee Name"],
    "client": ["Client Name", "Client"],
    "region": ["Region"],
    "practice": ["Practices"],
    "job_role": ["Job category"],
    "sow": ["SowCategory"],
    "revenue": ["Revenue USD m", "Revenue"],
    "cost": ["Cost USD m", "Cost"],
    "profit": ["GP USD m", "Profit"],
    "gpm": ["GPM", "Margin %"],
    "month": ["Month", "Date"],
    "headcount": ["HC", "Headcount", "Employees"],
    "accounts": ["Client", "Clients", "Accounts"],
}

# ===== UTILITY FUNCTIONS =====
def get_available_metrics(df_cir, df_res):
    """Dynamically identify available metrics"""
    metrics = {}
    
    if 'Revenue USD m' in df_cir.columns:
        metrics['revenue'] = df_cir['Revenue USD m'].sum()
    
    if 'Cost USD m' in df_cir.columns:
        metrics['cost'] = df_cir['Cost USD m'].sum()
    
    if 'GP USD m' in df_cir.columns:
        metrics['profit'] = df_cir['GP USD m'].sum()
    
    if 'GPM' in df_cir.columns:
        metrics['gpm'] = df_cir['GPM'].mean()
    elif metrics.get('revenue') and metrics.get('profit'):
        metrics['gpm'] = (metrics['profit'] / metrics['revenue'] * 100)
    
    metrics['headcount'] = len(df_res)
    metrics['accounts'] = df_cir['Client'].nunique() if 'Client' in df_cir.columns else 0
    
    return metrics

def generate_dynamic_context(df_cir, df_res, metrics):
    """Generate context dynamically based on available data"""
    context_lines = ["PORTFOLIO ANALYSIS DATA:\n"]
    
    # Current metrics
    if 'revenue' in metrics:
        context_lines.append(f"- Total Revenue: ${metrics['revenue']:.3f}M")
    if 'cost' in metrics:
        context_lines.append(f"- Total Cost: ${metrics['cost']:.3f}M")
    if 'profit' in metrics:
        context_lines.append(f"- Gross Profit: ${metrics['profit']:.3f}M")
    if 'gpm' in metrics:
        context_lines.append(f"- Profit Margin: {metrics['gpm']:.2f}%")
    if metrics.get('headcount'):
        context_lines.append(f"- Headcount: {metrics['headcount']}")
    if metrics.get('accounts'):
        context_lines.append(f"- Active Accounts: {metrics['accounts']}")
    
    context_lines.append("")
    
    # Top accounts by revenue
    if 'Client' in df_cir.columns and 'Revenue USD m' in df_cir.columns:
        top_acct = df_cir.groupby('Client')['Revenue USD m'].sum().sort_values(ascending=False).head(5)
        context_lines.append("Top 5 Accounts by Revenue:")
        context_lines.append(top_acct.round(3).to_string())
        context_lines.append("")
    
    # Revenue by region
    if 'Region' in df_cir.columns and 'Revenue USD m' in df_cir.columns:
        by_reg = df_cir.groupby('Region')['Revenue USD m'].sum()
        context_lines.append("Revenue by Region:")
        context_lines.append(by_reg.round(3).to_string())
        context_lines.append("")
    
    # Headcount by client
    if 'Client Name' in df_res.columns:
        hc_by_client = df_res.groupby('Client Name').size().sort_values(ascending=False).head(5)
        context_lines.append("Top 5 Accounts by Headcount:")
        context_lines.append(hc_by_client.to_string())
        context_lines.append("")
    
    return "\n".join(context_lines)

def suggest_questions(df_cir, df_res):
    """Suggest questions based on available data"""
    suggestions = [
        "What is total revenue?",
        "Show top accounts by revenue",
        "Revenue breakdown by region",
        "How many resources per account?",
    ]
    
    if 'GPM' in df_cir.columns or 'GP USD m' in df_cir.columns:
        suggestions.append("What is profit margin by account?")
    
    if 'Month' in df_cir.columns:
        suggestions.append("Show month-over-month growth")
    
    return suggestions

# ===== LOAD CURRENT MONTH DATA =====
st.sidebar.header("📁 Data Source")
data_source = st.sidebar.radio("Choose source:", ["Upload Files", "OneDrive"])

df_res = None
df_cir = None
data_loaded = False

if data_source == "Upload Files":
    resource_file = st.sidebar.file_uploader("Resource Data", type=['xlsx'])
    circle_file = st.sidebar.file_uploader("Circle Wise Data", type=['xlsx'])
    
    if resource_file and circle_file:
        try:
            df_res = pd.read_excel(resource_file, sheet_name=0, header=0)
            df_cir = pd.read_excel(circle_file, sheet_name='Sheet1', header=0)
            
            st.sidebar.write(f"📊 Loaded {len(df_res)} resources, {len(df_cir)} records")
            
            # Filter
            df_res_filtered = df_res[(df_res['Region'].isin(GGM_REGIONS)) & (df_res['Practices'] == GGM_PRACTICE)]
            df_cir_filtered = df_cir[df_cir['Region'].isin(GGM_REGIONS)]
            
            df_res = df_res_filtered
            df_cir = df_cir_filtered
            
            data_loaded = len(df_res) > 0 and len(df_cir) > 0
            
            if data_loaded:
                st.sidebar.success(f"✅ Filtered: {len(df_res)} resources, {len(df_cir)} records")
            else:
                st.sidebar.warning("⚠️ No data after filtering")
                
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
        st.sidebar.error(f"OneDrive error: {str(e)}")

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
    # Get metrics dynamically
    metrics = get_available_metrics(df_cir, df_res)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Revenue", f"${metrics.get('revenue', 0):.3f}M")
    col2.metric("Profit Margin", f"{metrics.get('gpm', 0):.2f}%")
    col3.metric("Headcount", metrics.get('headcount', 0))
    col4.metric("Active Accounts", metrics.get('accounts', 0))
    
    # ===== MTD / QTD / YTD SECTION =====
    ytd_rev = ytd_gp = ytd_gpm = 0
    qtd_rev = qtd_gp = qtd_gpm = 0
    mtd_rev = mtd_gp = mtd_gpm = 0
    mom_growth = 0
    
    if not df_hist.empty:
        st.divider()
        st.subheader("📈 Period Comparisons")
        
        # Determine current month from data if available
        if 'Month' in df_cir.columns:
            try:
                df_cir['Month'] = pd.to_datetime(df_cir['Month'], errors='coerce')
                current_month = df_cir['Month'].dt.month.max()
                current_year = df_cir['Month'].dt.year.max()
            except:
                today = datetime.now()
                current_month = today.month
                current_year = today.year
        else:
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
        if 'Region' in df_cir.columns and 'Revenue USD m' in df_cir.columns:
            st.write("**Revenue by Region**")
            by_reg = df_cir.groupby('Region')['Revenue USD m'].sum().sort_values(ascending=False)
            fig1 = px.bar(by_reg, title='Revenue by Region', labels={'value': 'Revenue ($M)'})
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        if 'Client' in df_cir.columns and 'Revenue USD m' in df_cir.columns:
            st.write("**Top 10 Accounts**")
            by_acct = df_cir.groupby('Client')['Revenue USD m'].sum().sort_values(ascending=False).head(10)
            fig2 = px.bar(by_acct, title='Top 10 Accounts', labels={'value': 'Revenue ($M)'})
            st.plotly_chart(fig2, use_container_width=True)
    
    # ===== NLP AGENT (Groq) =====
    st.divider()
    st.subheader("💬 Ask Questions")
    
    # Show suggestions
    with st.expander("💡 Suggested questions"):
        suggestions = suggest_questions(df_cir, df_res)
        for suggestion in suggestions:
            st.write(f"• {suggestion}")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Ask about revenue, accounts, regions, headcount, margins..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    # Generate dynamic context
                    dynamic_context = generate_dynamic_context(df_cir, df_res, metrics)
                    
                    # Add historical context if available
                    if not df_hist.empty:
                        dynamic_context += f"\n\nHISTORICAL SUMMARY:\n"
                        dynamic_context += f"- YTD Revenue: ${ytd_rev:.3f}M (GPM: {ytd_gpm:.2f}%)\n"
                        dynamic_context += f"- QTD Revenue: ${qtd_rev:.3f}M (GPM: {qtd_gpm:.2f}%)\n"
                        dynamic_context += f"- MTD Revenue: ${mtd_rev:.3f}M (GPM: {mtd_gpm:.2f}%, Growth: {mom_growth:+.1f}% MoM)"
                    
                    groq_api_key = st.secrets.get("groq", {}).get("api_key")
                    
                    if not groq_api_key:
                        st.error("❌ Groq API key not configured. Add to Streamlit Cloud Secrets.")
                    else:
                        from groq import Groq
                        
                        client = Groq(api_key=groq_api_key)
                        
                        response = client.chat.completions.create(
                            messages=[
                                {
                                    "role": "system",
                                    "content": f"You are a portfolio analysis expert. Answer questions based on this data concisely and numerically:\n\n{dynamic_context}"
                                },
                                {"role": "user", "content": prompt}
                            ],
                            model="llama-3-70b-versatile",
                            max_tokens=500,
                        )
                        
                        answer = response.choices[0].message.content
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.markdown(answer)
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")

else:
    st.info("📁 Upload Excel files or configure OneDrive")