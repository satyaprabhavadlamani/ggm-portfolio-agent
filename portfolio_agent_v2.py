import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
from difflib import SequenceMatcher
from local_folder_loader import load_monthly_data_from_folder, get_available_months

st.set_page_config(page_title="GGM D&I Portfolio Agent", layout="wide")
st.title("📊 GGM D&I Portfolio Agent")

# ===== CONFIGURATION =====
GGM_CIRCLE = 'Data and Insights'

# ===== MONTH ORDERING (Calendar order: Jan=1, Dec=12) =====
MONTH_ORDER = {
    'Jan': 1, 'January': 1,
    'Feb': 2, 'February': 2,
    'Mar': 3, 'March': 3,
    'Apr': 4, 'April': 4,
    'May': 5,
    'Jun': 6, 'June': 6,
    'Jul': 7, 'July': 7,
    'Aug': 8, 'August': 8,
    'Sep': 9, 'September': 9,
    'Oct': 10, 'October': 10,
    'Nov': 11, 'November': 11,
    'Dec': 12, 'December': 12,
}

# ===== GROQ MODEL FALLBACK =====
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "gemma2-9b-it",
]

# ===== EXECUTIVE SYNONYM MAPPING =====
SYNONYM_MAP = {
    'revenue': ['sales', 'top line', 'topline', 'income', 'earnings', 'throughput', 'business', 'volume', 'billing', 'invoiced', 'billed'],
    'revenue_amount': ['revenue', 'sales', 'top line', 'generated', 'brought in', 'total sales'],
    'profit': ['gp', 'gross profit', 'margin', 'bottomline', 'bottom line', 'earnings', 'returns', 'benefit', 'gain', 'net'],
    'gp_amount': ['profit', 'gp', 'gross profit', 'gains', 'profit amount'],
    'gpm': ['margin', 'profitability', 'margin %', 'percentage', 'efficiency', 'returns', 'yield', 'margin percentage'],
    'headcount': ['staff', 'team', 'people', 'employees', 'strength', 'bench', 'resources', 'workforce', 'fte', 'team size', 'personnel', 'manpower'],
    'headcount_amount': ['headcount', 'staff', 'people', 'team size', 'strength', 'number of people'],
    'account': ['client', 'customer', 'partner', 'engagement', 'project', 'contract', 'business', 'deal', 'opportunity'],
    'client_name': ['account', 'client', 'customer', 'project', 'engagement', 'company', 'vendor'],
    'region': ['geography', 'location', 'zone', 'area', 'market', 'country', 'place', 'territory'],
    'cost': ['expense', 'cost of sales', 'cogs', 'operational cost', 'spending'],
    'mtd': ['month', 'current month', 'this month', 'monthly', 'month to date'],
    'qtd': ['quarter', 'quarterly', 'this quarter', 'current quarter', 'q1', 'q2', 'q3', 'q4'],
    'ytd': ['year', 'annual', 'yearly', 'this year', 'current year', 'year to date'],
    'growth': ['increase', 'improvement', 'trend', 'change', 'momentum', 'uptick', 'growth rate', 'expansion'],
    'decline': ['decrease', 'drop', 'fall', 'downturn', 'reduction', 'dip', 'decline rate'],
    'top': ['best', 'largest', 'biggest', 'highest', 'leading', 'major', 'top performing'],
    'bottom': ['worst', 'smallest', 'lowest', 'trailing', 'weakest', 'bottom performing'],
    'performance': ['how is', 'status', 'doing', 'outlook', 'trajectory', 'progress'],
    'comparison': ['vs', 'versus', 'against', 'compared to', 'relative to'],
}

# ===== UTILITY FUNCTIONS =====

def sort_months_calendar_order(months):
    """Sort months in calendar order (Jan -> Dec), NOT alphabetically"""
    if isinstance(months, np.ndarray):
        months = list(months)
    return sorted(months, key=lambda m: MONTH_ORDER.get(m, 99))

def detect_column_mapping(df_cir, df_res):
    """Intelligently detect and map column names from actual data"""
    column_map = {
        'revenue': None,
        'cost': None,
        'profit': None,
        'gpm': None,
        'client': None,
        'region': None,
        'circle': None,
        'month': None,
    }
    
    # Revenue column detection (multiple patterns)
    for col in df_cir.columns:
        col_lower = col.lower()
        if any(term in col_lower for term in ['revenue', 'sales', 'topline', 'top_line', 'billing', 'invoiced']):
            column_map['revenue'] = col
            break
    
    # Cost/COGS detection
    for col in df_cir.columns:
        col_lower = col.lower()
        if any(term in col_lower for term in ['cost', 'cogs', 'expense', 'operational cost']):
            column_map['cost'] = col
            break
    
    # Profit/GP detection
    for col in df_cir.columns:
        col_lower = col.lower()
        if any(term in col_lower for term in ['gross profit', 'gp ', 'profit ', 'net profit', 'gain']):
            column_map['profit'] = col
            break
    
    # GPM/Margin detection
    for col in df_cir.columns:
        col_lower = col.lower()
        if any(term in col_lower for term in ['margin', 'gpm', 'profitability', 'margin %']):
            column_map['gpm'] = col
            break
    
    # Client/Customer detection
    for col in df_cir.columns:
        col_lower = col.lower()
        if any(term in col_lower for term in ['client', 'customer', 'account', 'company', 'vendor']):
            column_map['client'] = col
            break
    
    # Region detection
    for col in df_cir.columns:
        col_lower = col.lower()
        if any(term in col_lower for term in ['region', 'geography', 'location', 'territory', 'zone']):
            column_map['region'] = col
            break
    
    # Circle/Practice detection
    for col in df_cir.columns:
        col_lower = col.lower()
        if any(term in col_lower for term in ['circle', 'practice', 'business unit', 'bu']):
            column_map['circle'] = col
            break
    
    # Month detection
    for col in df_cir.columns:
        col_lower = col.lower()
        if any(term in col_lower for term in ['month', 'period', 'date', 'time_period']):
            column_map['month'] = col
            break
    
    # Fallback to exact matches if not found
    for key in ['revenue', 'cost', 'profit', 'gpm', 'client', 'region', 'circle', 'month']:
        if column_map[key] is None:
            if f'Revenue USD m' in df_cir.columns and key == 'revenue':
                column_map['revenue'] = 'Revenue USD m'
            elif f'Cost USD m' in df_cir.columns and key == 'cost':
                column_map['cost'] = 'Cost USD m'
            elif f'GP USD m' in df_cir.columns and key == 'profit':
                column_map['profit'] = 'GP USD m'
            elif f'Circle' in df_cir.columns and key == 'circle':
                column_map['circle'] = 'Circle'
    
    return column_map

def expand_synonyms(text):
    """Expand user query with synonym explanations for AI"""
    expanded_context = "\n[SYNONYM CONTEXT]: "
    for main_term, synonyms in SYNONYM_MAP.items():
        for syn in synonyms:
            if syn.lower() in text.lower():
                expanded_context += f"'{syn}' = {main_term}; "
    return expanded_context if expanded_context != "\n[SYNONYM CONTEXT]: " else ""

def get_column_explanations(column_map):
    """Generate explanations for detected columns for AI context"""
    explanations = []
    if column_map.get('revenue'):
        explanations.append(f"Revenue column: '{column_map['revenue']}'")
    if column_map.get('cost'):
        explanations.append(f"Cost column: '{column_map['cost']}'")
    if column_map.get('profit'):
        explanations.append(f"Profit column: '{column_map['profit']}'")
    if column_map.get('gpm'):
        explanations.append(f"Margin column: '{column_map['gpm']}'")
    if column_map.get('client'):
        explanations.append(f"Client/Account column: '{column_map['client']}'")
    if column_map.get('region'):
        explanations.append(f"Region column: '{column_map['region']}'")
    return explanations

def fuzzy_match_client(user_text, df_res):
    """Find matching client names from user query using fuzzy matching"""
    if df_res is None or 'Client Name' not in df_res.columns:
        return user_text
    
    actual_clients = df_res['Client Name'].unique()
    words = user_text.lower().split()
    updated_text = user_text
    
    for word in words:
        if len(word) < 3:
            continue
        
        best_match = None
        best_score = 0
        
        for client in actual_clients:
            ratio = SequenceMatcher(None, word.lower(), client.lower()).ratio()
            if ratio > best_score and ratio > 0.6:
                best_score = ratio
                best_match = client
        
        if best_match:
            updated_text = updated_text.replace(word, best_match, 1)
    
    return updated_text

def get_available_metrics(df_cir, df_res, column_map=None):
    """Dynamically identify available metrics using detected column names"""
    metrics = {}
    
    # Try to use column_map if provided, otherwise use standard column names
    revenue_col = None
    cost_col = None
    profit_col = None
    
    if column_map:
        revenue_col = column_map.get('revenue')
        cost_col = column_map.get('cost')
        profit_col = column_map.get('profit')
    
    # Fallback to standard names if not in column_map
    if not revenue_col and 'Revenue USD m' in df_cir.columns:
        revenue_col = 'Revenue USD m'
    if not cost_col and 'Cost USD m' in df_cir.columns:
        cost_col = 'Cost USD m'
    if not profit_col and 'GP USD m' in df_cir.columns:
        profit_col = 'GP USD m'
    
    # Calculate metrics
    if revenue_col and revenue_col in df_cir.columns:
        metrics['revenue'] = df_cir[revenue_col].sum()
    else:
        metrics['revenue'] = 0
    
    if cost_col and cost_col in df_cir.columns:
        metrics['cost'] = df_cir[cost_col].sum()
    else:
        metrics['cost'] = 0
    
    if profit_col and profit_col in df_cir.columns:
        metrics['profit'] = df_cir[profit_col].sum()
    elif metrics.get('revenue') and metrics.get('cost'):
        metrics['profit'] = metrics['revenue'] - metrics['cost']
    else:
        metrics['profit'] = 0
    
    # Calculate GPM
    if metrics.get('revenue') and metrics.get('revenue') > 0:
        metrics['gpm'] = (metrics['profit'] / metrics['revenue'] * 100)
    else:
        metrics['gpm'] = 0
    
    # Headcount
    if df_res is not None and len(df_res) > 0:
        metrics['headcount'] = len(df_res)
        metrics['active'] = len(df_res)
    else:
        metrics['headcount'] = 0
        metrics['active'] = 0
    
    # Accounts (try to detect client column)
    client_col = column_map.get('client') if column_map else None
    if not client_col and 'Client Name' in df_res.columns:
        client_col = 'Client Name'
    
    if df_res is not None and client_col and client_col in df_res.columns:
        metrics['accounts'] = df_res[client_col].nunique()
    elif df_res is not None and 'Client Name' in df_res.columns:
        metrics['accounts'] = df_res['Client Name'].nunique()
    else:
        metrics['accounts'] = 0
    
    return metrics

def generate_dynamic_context(df_cir, df_res, metrics, df_cir_raw=None, selected_month=None, column_map=None):
    """Generate comprehensive executive-level context for AI"""
    context_lines = ["="*70, "PORTFOLIO INTELLIGENCE DASHBOARD", "="*70, ""]
    
    # Explain data file structure upfront
    context_lines.append("📂 DATA FILE STRUCTURE & AVAILABILITY:")
    context_lines.append("   Source 1 - Circle Wise Financial Data (FINANCIAL METRICS):")
    context_lines.append("   • CONTAINS: All months Jan 2026 through Aug 2026 in single consolidated Excel sheet")
    context_lines.append("   • METRICS: Available financial columns detected below")
    context_lines.append("   • STRUCTURE: One row per month-region-client combination")
    context_lines.append("")
    
    # Show detected columns
    if column_map:
        context_lines.append("📋 DETECTED COLUMNS IN DATA (VP may use different names):")
        if column_map.get('revenue'):
            context_lines.append(f"   • Revenue: '{column_map['revenue']}' (VP may say: sales, topline, income)")
        if column_map.get('cost'):
            context_lines.append(f"   • Cost: '{column_map['cost']}' (VP may say: expense, COGS, operational cost)")
        if column_map.get('profit'):
            context_lines.append(f"   • Profit: '{column_map['profit']}' (VP may say: GP, gains, net profit)")
        if column_map.get('gpm'):
            context_lines.append(f"   • Margin: '{column_map['gpm']}' (VP may say: profitability, margin %)")
        if column_map.get('client'):
            context_lines.append(f"   • Client: '{column_map['client']}' (VP may say: account, customer, company)")
        if column_map.get('region'):
            context_lines.append(f"   • Region: '{column_map['region']}' (VP may say: geography, territory, location)")
        context_lines.append("")
    
    context_lines.append("   Source 2 - Resource Headcount Data (STAFFING SNAPSHOT):")
    context_lines.append("   • REPRESENTS: Latest active employee data as of August 2026")
    context_lines.append("   • METRICS: Active headcount, client assignments, practices, regions")
    context_lines.append("   • STRUCTURE: One row per active employee (no historical versions)")
    context_lines.append("")
    
    if df_cir_raw is not None and 'Month' in df_cir_raw.columns:
        available_months = sort_months_calendar_order(df_cir_raw['Month'].unique())
        context_lines.append(f"✅ FINANCIAL DATA AVAILABLE FOR ALL MONTHS:")
        context_lines.append(f"   Months available: {', '.join(map(str, available_months))} (complete set)")
        context_lines.append(f"   Currently selected for dashboard: {selected_month if selected_month else 'August (latest)'}")
        context_lines.append(f"   Full historical data: Available for month-over-month analysis, trends, seasonal patterns")
        context_lines.append("")
        
        if selected_month and len(available_months) > 1:
            available_months_list = list(available_months)
            if selected_month in available_months_list:
                idx = available_months_list.index(selected_month)
                if idx > 0:
                    prev_month = available_months_list[idx - 1]
                    context_lines.append(f"📊 COMPARISON CAPABILITY:")
                    context_lines.append(f"   {selected_month} vs {prev_month}: Month-over-month metrics available")
                    context_lines.append("")
    
    context_lines.append("📊 PORTFOLIO SNAPSHOT:")
    if 'revenue' in metrics:
        context_lines.append(f"  Revenue:        ${metrics['revenue']:.3f}M USD")
    if 'profit' in metrics:
        context_lines.append(f"  Gross Profit:   ${metrics['profit']:.3f}M USD")
    if 'gpm' in metrics:
        context_lines.append(f"  Profit Margin:  {metrics['gpm']:.2f}%")
    if metrics.get('headcount'):
        context_lines.append(f"  Headcount:      {metrics['headcount']} people")
    if metrics.get('accounts'):
        context_lines.append(f"  Active Clients: {metrics['accounts']} accounts")
    context_lines.append("")
    
    if df_res is not None and 'Client Name' in df_res.columns:
        top_clients = df_res['Client Name'].value_counts().head(5)
        context_lines.append("📋 TOP 5 ACCOUNTS BY HEADCOUNT:")
        for i, (client, count) in enumerate(top_clients.items(), 1):
            context_lines.append(f"   {i}. {client}: {count} employees")
        context_lines.append("")
    
    context_lines.append("⚠️ CRITICAL RULES:")
    context_lines.append("1. The Circle Wise file contains ALL months (Jan-Aug) ALREADY CONSOLIDATED")
    context_lines.append("2. DO NOT say 'we need July data' - it IS in the context above")
    context_lines.append("3. DO NOT ask for missing months - all available months are listed above")
    context_lines.append("4. Headcount is CURRENT SNAPSHOT only (no historical versions)")
    context_lines.append("5. ALWAYS use the available data - never claim data is missing")
    context_lines.append("")
    
    return "\n".join(context_lines)

def suggest_questions(df_cir):
    """Generate contextual follow-up questions based on data"""
    questions = []
    
    if 'Client' in df_cir.columns:
        top_client = df_cir.groupby('Client')['Revenue USD m'].sum().idxmax()
        questions.append(f"How is {top_client} performing this month?")
    
    if 'Region' in df_cir.columns:
        regions = df_cir['Region'].unique()
        if len(regions) > 1:
            questions.append(f"Compare revenue across {', '.join(regions)}")
    
    questions.extend([
        "What are our top 5 accounts by revenue?",
        "Show me margin trends",
        "Which accounts are growing?"
    ])
    
    return questions[:3]

def get_groq_response(client, messages):
    """Get response from Groq API with fallback models"""
    for model in GROQ_MODELS:
        try:
            if hasattr(client, 'messages'):
                response = client.messages.create(
                    model=model,
                    messages=messages,
                    max_tokens=1024
                )
                return response.content[0].text
            elif hasattr(client, 'chat') and hasattr(client.chat, 'completions'):
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=1024
                )
                return response.choices[0].message.content
        except Exception as e:
            continue
    
    return None

# ===== SESSION STATE INITIALIZATION =====
if "messages" not in st.session_state:
    st.session_state.messages = []
if "groq_requests" not in st.session_state:
    st.session_state.groq_requests = []

# ===== SIDEBAR CONFIGURATION =====
st.sidebar.title("⚙️ Settings")

df_cir = None
df_cir_raw = None
df_res = None
data_loaded = False
selected_month = None

# ===== DATA SOURCE: LOCAL FOLDER AUTO-DISCOVERY =====
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Data Source")

data_source = st.sidebar.radio("Choose source:", ["Local Folder (Auto-Load)", "Manual Upload"])

if data_source == "Local Folder (Auto-Load)":
    st.sidebar.write("📁 Auto-loading from GGM Portfolio Analytics folder...")
    
    # Your folder path
    BASE_PATH = r"C:\Users\satyaprabha.v\OneDrive - ascendion\Documents\GGM Data\GGM Portfolio Analytics"
    YEAR = "2026"
    
    if st.sidebar.button("🔄 Load Historical Data"):
        with st.spinner("Scanning folders and loading data..."):
            try:
                df_cir_raw, df_res = load_monthly_data_from_folder(
                    base_path=BASE_PATH,
                    year=YEAR,
                    circle_name='Data and Insights'
                )
                
                if df_cir_raw is not None and len(df_cir_raw) > 0:
                    data_loaded = True
                    
                    # Show available months
                    available_months = get_available_months(df_cir_raw)
                    st.sidebar.success(f"✅ Data loaded!")
                    st.sidebar.write(f"📅 Available months: {', '.join(available_months)}")
                    
                    # Month selector
                    selected_month = st.sidebar.selectbox(
                        "Select month for dashboard:",
                        options=available_months,
                        index=len(available_months) - 1  # Default to latest month
                    )
                    
                    # Filter to selected month for current dashboard
                    if selected_month:
                        df_cir = df_cir_raw[df_cir_raw['Month'] == selected_month].copy()
                        st.sidebar.write(f"📊 Dashboard month: {selected_month}")
                else:
                    st.sidebar.error("❌ No data loaded. Check folder structure.")
            
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")
                st.sidebar.write("💡 Check that:")
                st.sidebar.write("- Folder path is correct")
                st.sidebar.write("- Month folders exist (August, July, etc.)")
                st.sidebar.write("- Excel files are in month folders")

elif data_source == "Manual Upload":
    st.sidebar.write("📁 Manual File Upload")
    
    resource_file = st.sidebar.file_uploader("Resource Data (Optional)", type=['xlsx'])
    circle_file = st.sidebar.file_uploader("Circle Wise Data (Required)", type=['xlsx'])
    
    if circle_file:
        try:
            # Single sheet load for manual upload
            df_cir_raw = pd.read_excel(circle_file, sheet_name=0, header=0)
            
            if 'Circle' in df_cir_raw.columns:
                df_cir_raw = df_cir_raw[df_cir_raw['Circle'] == 'Data and Insights']
            
            st.sidebar.write(f"✅ Loaded {len(df_cir_raw)} records")
            
            if resource_file:
                df_res = pd.read_excel(resource_file, sheet_name=0, header=0)
                if 'Practices' in df_res.columns:
                    df_res = df_res[df_res['Practices'] == 'Data and Insights']
                st.sidebar.write(f"✅ Loaded {len(df_res)} employees")
            
            if df_cir_raw is not None and len(df_cir_raw) > 0:
                data_loaded = True
                
                # Check for Month column for historical data
                if 'Month' in df_cir_raw.columns:
                    available_months = sort_months_calendar_order(df_cir_raw['Month'].unique())
                    st.sidebar.write(f"📅 Months (calendar order): {', '.join(map(str, available_months))}")
                    
                    selected_month = st.sidebar.selectbox(
                        "Select month:",
                        options=available_months,
                        index=len(available_months) - 1
                    )
                    st.sidebar.write(f"✅ Selected: {selected_month}")
                    
                    df_cir = df_cir_raw[df_cir_raw['Month'] == selected_month].copy()
                else:
                    df_cir = df_cir_raw.copy()
                
                st.sidebar.success("✅ Data loaded!")
        
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")

if data_loaded and df_cir is not None:
    # Check Groq API key
    groq_api_key = st.secrets.get("groq", {}).get("api_key")
    if not groq_api_key:
        st.warning(
            "⚠️ **Groq API key not configured**\n\n"
            "To enable the AI agent:\n"
            "1. Go to Streamlit Cloud → Settings → Secrets\n"
            "2. Add:\n"
            "```\n"
            "[groq]\n"
            "api_key = \"gsk_your_key_here\"\n"
            "```\n"
            "3. Get your key: https://console.groq.com/keys\n"
            "4. Redeploy the app"
        )
    
    # Detect columns once for the entire session
    column_map = detect_column_mapping(df_cir, df_res)
    metrics = get_available_metrics(df_cir, df_res, column_map)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Revenue", f"${metrics.get('revenue', 0):.3f}M")
    col2.metric("Gross Profit", f"${metrics.get('profit', 0):.3f}M", f"GPM: {metrics.get('gpm', 0):.2f}%")
    col3.metric("Headcount", metrics.get('headcount', 0), "Active")
    col4.metric("Accounts", metrics.get('accounts', 0))
    
    st.divider()
    st.subheader("💼 Account Breakdown")
    
    if 'Client' in df_cir.columns and 'Revenue USD m' in df_cir.columns:
        by_acct = df_cir.groupby('Client')['Revenue USD m'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(by_acct, title='Top 10 Accounts', labels={'value': 'Revenue ($M)'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    st.subheader("💬 Ask Questions")
    
    with st.expander("💡 Suggested questions"):
        suggestions = suggest_questions(df_cir)
        for suggestion in suggestions:
            st.write(f"• {suggestion}")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Get prompt from either chat input OR detect if button was clicked
    prompt = st.chat_input("Ask about revenue, accounts, regions...")
    
    # If no chat input but last message is user (unresponded), it's from button click
    if not prompt and len(st.session_state.messages) > 0:
        if st.session_state.messages[-1].get("role") == "user":
            prompt = st.session_state.messages[-1].get("content")
    
    if prompt:
        matched_prompt = fuzzy_match_client(prompt, df_res)
        
        # Only append message if it's not already the last message (from button click)
        if len(st.session_state.messages) == 0 or st.session_state.messages[-1].get("content") != prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            if matched_prompt != prompt:
                st.caption(f"🔍 Recognized: {matched_prompt}")
        
        with st.chat_message("assistant"):
            answer = None  # Initialize so it's available outside try block
            with st.spinner("Analyzing..."):
                try:
                    # Use already-detected column mapping for consistent handling
                    dynamic_context = generate_dynamic_context(df_cir, df_res, metrics, df_cir_raw, selected_month, column_map)
                    groq_api_key = st.secrets.get("groq", {}).get("api_key")
                    
                    if not groq_api_key:
                        st.error("❌ Groq API key not configured")
                    else:
                        from groq import Groq
                        client = Groq(api_key=groq_api_key)
                        
                        synonym_context = expand_synonyms(matched_prompt)
                        system_prompt = f"""You are a senior portfolio analyst briefing C-level executives on GGM D&I portfolio performance.

CRITICAL DATA STRUCTURE - UNDERSTAND THIS FIRST:
{dynamic_context}

⚠️ UNDERSTANDING EXECUTIVE TERMINOLOGY:
1. VPs may use business language instead of exact column names
   • "revenue" = "sales", "topline", "income", "billing"
   • "profit" = "GP", "gains", "earnings", "returns"
   • "margin" = "profitability", "margin %", "efficiency"
   • "headcount" = "staff", "team", "people", "resources", "FTE"
   • "account" = "client", "customer", "engagement", "company"
   • "region" = "geography", "territory", "location", "zone"

2. Translate VP questions to actual data columns automatically
   • When VP asks "How are we doing on sales?", use revenue data
   • When VP asks "Show me our staff breakdown", use headcount data
   • When VP asks "Which are our top customers?", use client/account data

⚠️ CRITICAL DATA RULES:
1. ALL financial data (revenue, cost, profit, margin) for Jan-Aug 2026 is ALREADY IN your context
2. DO NOT ask for missing months - they are provided in the column mapping above
3. DO NOT ask for July data when doing Aug vs Jul comparison - use the data provided
4. Headcount is a CURRENT SNAPSHOT (August 2026) - it doesn't have historical versions
5. When comparing months, use months available (Jan-Aug)
6. When you say "we need data", check the context above first - it's probably there

WHAT YOU CAN ANSWER:
✅ Any financial metric (any terminology) for Jan-Aug
✅ Month-over-month comparisons with growth rates
✅ Trends across multiple months
✅ Current headcount and staffing by account
✅ Account/client rankings, concentration, efficiency metrics
✅ Regional performance and breakdown

WHAT YOU CANNOT DO:
❌ Provide September or later data (doesn't exist)
❌ Provide historical headcount versions (only current snapshot available)
❌ Make assumptions about data - use only what's in the context

RESPONSE GUIDELINES:
1. Start with DIRECT answer (VPs value brevity)
2. Translate business language to data automatically
3. Use AVAILABLE data from the context (don't ask for it)
4. Include month-to-month comparisons when relevant
5. Provide business implications, not just numbers
6. Suggest 2-3 follow-up questions phrased in business language

{synonym_context}"""
                        
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": matched_prompt}
                        ]
                        
                        answer = get_groq_response(client, messages)
                        
                        if answer:
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            st.markdown(answer)
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            
            # Buttons OUTSIDE spinner so they persist (not hidden after loading)
            if answer:
                st.divider()
                st.caption("💡 **Suggested follow-ups:**")
                col1, col2, col3 = st.columns(3)
                followups = ["Show me the trend", "Deep dive into top account", "Compare to last month"]
                
                # Define callback to add message and trigger rerun
                def on_followup_click(text):
                    st.session_state.messages.append({"role": "user", "content": text})
                
                for i, followup in enumerate(followups):
                    with [col1, col2, col3][i]:
                        st.button(
                            followup,
                            key=f"btn_{i}_{hash(str(st.session_state.messages))}",
                            on_click=on_followup_click,
                            args=(followup,)
                        )

else:
    st.info("📁 Upload Circle Wise data or click 'Load Historical Data' to start")#   F o r c e   r e d e p l o y  
 