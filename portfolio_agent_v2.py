import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="GGM D&I Portfolio Agent", layout="wide")
st.title("📊 GGM D&I Portfolio Agent")

# ===== CONFIGURATION =====
GGM_CIRCLE = 'Data and Insights'

# ===== GROQ MODEL FALLBACK =====
GROQ_MODELS = [
    "llama-3.3-70b-versatile",    # Best reasoning (current preferred)
    "openai/gpt-oss-120b",         # High capability fallback
    "gemma2-9b-it",                # Lightweight fallback
]

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
    
    # Headcount from resource file - total employee count
    if df_res is not None and len(df_res) > 0:
        metrics['headcount'] = len(df_res)
        
        # Active employees - excluding benched and terminated
        active = df_res.copy()
        # Exclude benched
        if 'Bench Resource' in active.columns:
            active = active[active['Bench Resource'] != 'Y']
        # Exclude terminated (has attrition date)
        if 'AttritionDate' in active.columns:
            active = active[active['AttritionDate'].isna()]
        metrics['active'] = len(active)
    else:
        metrics['headcount'] = 0
        metrics['active'] = 0
    
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
        context_lines.append(f"- Total Headcount: {metrics['headcount']}")
    if metrics.get('active'):
        context_lines.append(f"- Active Employees: {metrics['active']} (excl. benched & terminated)")
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
    if df_res is not None and 'Client Name' in df_res.columns:
        hc_by_client = df_res.groupby('Client Name').size().sort_values(ascending=False).head(5)
        context_lines.append("Top 5 Accounts by Headcount:")
        context_lines.append(hc_by_client.to_string())
        context_lines.append("")
    
    return "\n".join(context_lines)

def suggest_questions(df_cir):
    """Suggest questions based on available data"""
    suggestions = [
        "What is total revenue?",
        "Show top accounts by revenue",
        "Revenue breakdown by region",
    ]
    
    if 'GPM' in df_cir.columns or 'GP USD m' in df_cir.columns:
        suggestions.append("What is profit margin by account?")
    
    if 'Month' in df_cir.columns:
        suggestions.append("Show revenue by month")
    
    return suggestions

def get_groq_response(client, messages, max_attempts=3):
    """Try to get Groq response with model fallback"""
    last_error = None
    
    for attempt, model in enumerate(GROQ_MODELS[:max_attempts]):
        try:
            st.write(f"🤔 Using model: {model}")
            response = client.chat.completions.create(
                messages=messages,
                model=model,
                max_tokens=500,
            )
            st.success(f"✅ Response from {model}")
            return response.choices[0].message.content
        except Exception as e:
            last_error = str(e)
            st.warning(f"⚠️ {model} failed: {last_error[:100]}")
            continue
    
    # All models failed
    raise Exception(f"All Groq models failed. Last error: {last_error}")

# ===== LOAD CURRENT MONTH DATA =====
st.sidebar.header("📁 Data Source")
data_source = st.sidebar.radio("Choose source:", ["Upload Files", "OneDrive"])

df_res = None
df_cir = None
data_loaded = False

if data_source == "Upload Files":
    resource_file = st.sidebar.file_uploader("Resource Data (Optional)", type=['xlsx'])
    circle_file = st.sidebar.file_uploader("Circle Wise Data (Required)", type=['xlsx'])
    
    if circle_file:
        try:
            df_cir = pd.read_excel(circle_file, sheet_name='Sheet1', header=0)
            
            st.sidebar.write(f"📊 Loaded {len(df_cir)} Circle Wise records")
            
            # Filter by Circle if column exists
            if 'Circle' in df_cir.columns:
                df_cir = df_cir[df_cir['Circle'] == GGM_CIRCLE]
                st.sidebar.write(f"✅ Filtered: {len(df_cir)} {GGM_CIRCLE} records")
            
            # Load resource data if provided
            if resource_file:
                df_res = pd.read_excel(resource_file, sheet_name=0, header=0)
                # Filter if needed
                if 'Practices' in df_res.columns and 'Region' in df_res.columns:
                    df_res = df_res[df_res['Practices'] == 'Data and Insights']
                    # Get regions from Circle data
                    valid_regions = df_cir['Region'].unique() if 'Region' in df_cir.columns else []
                    df_res = df_res[df_res['Region'].isin(valid_regions)]
                st.sidebar.write(f"📊 Loaded {len(df_res)} resources")
            
            data_loaded = len(df_cir) > 0
            
            if data_loaded:
                st.sidebar.success("✅ Data loaded")
            else:
                st.sidebar.warning("⚠️ No data found")
                
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")

else:  # OneDrive
    try:
        circle_url = st.secrets["onedrive"]["circle_file_url"]
        df_cir = pd.read_excel(circle_url, sheet_name='Sheet1', header=0)
        
        if 'Circle' in df_cir.columns:
            df_cir = df_cir[df_cir['Circle'] == GGM_CIRCLE]
        
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
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Current Revenue", f"${metrics.get('revenue', 0):.3f}M")
    col2.metric("Gross Profit (GP)", f"${metrics.get('profit', 0):.3f}M", f"GPM: {metrics.get('gpm', 0):.2f}%")
    col3.metric("Headcount", metrics.get('headcount', 0), "Total employees")
    col4.metric("Active", metrics.get('active', 0), "Excl. benched/term.")
    col5.metric("Accounts", metrics.get('accounts', 0))
    
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
                df_cir['date'] = pd.to_datetime(df_cir['Month'], errors='coerce')
                current_month = df_cir['date'].dt.month.max()
                current_year = df_cir['date'].dt.year.max()
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
        suggestions = suggest_questions(df_cir)
        for suggestion in suggestions:
            st.write(f"• {suggestion}")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Ask about revenue, accounts, regions, margins..."):
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
                        st.error("❌ Groq API key not configured.")
                    else:
                        from groq import Groq
                        
                        client = Groq(api_key=groq_api_key)
                        
                        try:
                            messages = [
                                {
                                    "role": "system",
                                    "content": f"You are a portfolio analysis expert. Answer questions based on this data:\n\n{dynamic_context}"
                                },
                                {"role": "user", "content": prompt}
                            ]
                            
                            answer = get_groq_response(client, messages)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            st.markdown(answer)
                            
                        except Exception as e:
                            st.error(f"❌ Model error: {str(e)}")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")

else:
    st.info("📁 Upload Circle Wise data to start")