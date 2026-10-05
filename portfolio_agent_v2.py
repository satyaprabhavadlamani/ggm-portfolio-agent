import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
from difflib import SequenceMatcher

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

# ===== EXECUTIVE SYNONYM MAPPING =====
SYNONYM_MAP = {
    # Revenue synonyms
    'revenue': ['sales', 'top line', 'topline', 'income', 'earnings', 'throughput', 'business', 'volume'],
    'revenue_amount': ['revenue', 'sales', 'top line', 'generated', 'brought in'],
    
    # Profit synonyms
    'profit': ['gp', 'gross profit', 'margin', 'bottomline', 'bottom line', 'earnings', 'returns', 'benefit'],
    'gp_amount': ['profit', 'gp', 'gross profit', 'gains'],
    
    # Margin synonyms
    'gpm': ['margin', 'profitability', 'margin %', 'percentage', 'efficiency', 'returns', 'yield'],
    
    # Headcount synonyms
    'headcount': ['staff', 'team', 'people', 'employees', 'strength', 'bench', 'resources', 'workforce', 'fte'],
    'headcount_amount': ['headcount', 'staff', 'people', 'team size', 'strength'],
    
    # Account/Client synonyms
    'account': ['client', 'customer', 'partner', 'engagement', 'project', 'contract', 'business'],
    'client_name': ['account', 'client', 'customer', 'project', 'engagement'],
    
    # Region synonyms
    'region': ['geography', 'location', 'zone', 'area', 'market', 'country', 'place'],
    
    # Time synonyms
    'mtd': ['month', 'current month', 'this month', 'monthly'],
    'qtd': ['quarter', 'quarterly', 'this quarter', 'current quarter'],
    'ytd': ['year', 'annual', 'yearly', 'this year', 'current year'],
    
    # Comparison synonyms
    'growth': ['increase', 'improvement', 'trend', 'change', 'momentum', 'uptick'],
    'decline': ['decrease', 'drop', 'fall', 'downturn', 'reduction', 'dip'],
    
    # Analysis synonyms
    'top': ['best', 'largest', 'biggest', 'highest', 'leading', 'major'],
    'bottom': ['worst', 'smallest', 'lowest', 'trailing', 'weakest'],
}

def expand_synonyms(text):
    """Expand user query with synonym explanations for AI"""
    expanded_context = "\n[SYNONYM CONTEXT]: "
    for main_term, synonyms in SYNONYM_MAP.items():
        for syn in synonyms:
            if syn.lower() in text.lower():
                expanded_context += f"'{syn}' = {main_term}; "
    return expanded_context if expanded_context != "\n[SYNONYM CONTEXT]: " else ""

# ===== UTILITY FUNCTIONS =====

def fuzzy_match_client(user_text, df_res):
    """Find matching client names from user query using fuzzy matching"""
    if df_res is None or 'Client Name' not in df_res.columns:
        return user_text
    
    actual_clients = df_res['Client Name'].unique()
    words = user_text.lower().split()
    updated_text = user_text
    
    for word in words:
        if len(word) < 3:  # Skip very short words
            continue
        
        # Find best match
        best_match = None
        best_score = 0
        
        for client in actual_clients:
            # Similarity score between word and client name
            ratio = SequenceMatcher(None, word.lower(), client.lower()).ratio()
            if ratio > best_score and ratio > 0.6:  # Threshold of 60% similarity
                best_score = ratio
                best_match = client
        
        # Replace word with full client name if good match found
        if best_match:
            updated_text = updated_text.replace(word, best_match, 1)
    
    return updated_text
def get_available_metrics(df_cir, df_res):
    """Dynamically identify available metrics"""
    metrics = {}
    
    if 'Revenue USD m' in df_cir.columns:
        metrics['revenue'] = df_cir['Revenue USD m'].sum()
    
    if 'Cost USD m' in df_cir.columns:
        metrics['cost'] = df_cir['Cost USD m'].sum()
    
    if 'GP USD m' in df_cir.columns:
        metrics['profit'] = df_cir['GP USD m'].sum()
    
    # Calculate GPM = (Gross Profit / Revenue) * 100
    if metrics.get('revenue') and metrics.get('revenue') > 0:
        metrics['gpm'] = (metrics['profit'] / metrics['revenue'] * 100)
    else:
        metrics['gpm'] = 0
    
    # Headcount & Active from resource file
    # File contains ONLY active resources, so both counts are the same
    if df_res is not None and len(df_res) > 0:
        metrics['headcount'] = len(df_res)
        metrics['active'] = len(df_res)  # Same as headcount (file pre-filtered to active only)
    else:
        metrics['headcount'] = 0
        metrics['active'] = 0
    
    # Accounts = distinct clients from Resource file
    if df_res is not None and 'Client Name' in df_res.columns:
        metrics['accounts'] = df_res['Client Name'].nunique()
    else:
        metrics['accounts'] = 0
    
    return metrics

def generate_dynamic_context(df_cir, df_res, metrics):
    """Generate comprehensive executive-level context for AI"""
    context_lines = ["="*70, "PORTFOLIO INTELLIGENCE DASHBOARD", "="*70, ""]
    
    # EXECUTIVE SUMMARY
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
    
    # KEY INSIGHTS
    context_lines.append("💡 KEY INSIGHTS:")
    if metrics.get('headcount') and metrics.get('revenue'):
        rev_per_emp = metrics['revenue'] / (metrics['headcount'] / 1000) if metrics['headcount'] > 0 else 0
        context_lines.append(f"  Revenue per Employee: ${rev_per_emp:.2f}K")
    if metrics.get('gpm') and metrics['gpm'] < 5:
        context_lines.append(f"  ⚠️  ALERT: Low margin ({metrics['gpm']:.2f}%) - review cost structure")
    elif metrics.get('gpm') and metrics['gpm'] > 20:
        context_lines.append(f"  ✅ Strong margin ({metrics['gpm']:.2f}%) - healthy profitability")
    context_lines.append("")
    
    # TOP ACCOUNTS ANALYSIS
    if 'Client' in df_cir.columns and 'Revenue USD m' in df_cir.columns:
        top_acct = df_cir.groupby('Client')['Revenue USD m'].sum().sort_values(ascending=False)
        total_rev = top_acct.sum()
        top_5_rev = top_acct.head(5).sum()
        concentration = (top_5_rev / total_rev * 100) if total_rev > 0 else 0
        
        context_lines.append("📈 REVENUE CONCENTRATION:")
        context_lines.append(f"  Top 5 accounts generate {concentration:.1f}% of total revenue")
        context_lines.append("")
        context_lines.append("🏆 TOP 5 ACCOUNTS BY REVENUE:")
        for i, (client, rev) in enumerate(top_acct.head(5).items(), 1):
            pct = (rev / total_rev * 100) if total_rev > 0 else 0
            context_lines.append(f"  {i}. {client}: ${rev:.3f}M ({pct:.1f}%)")
        context_lines.append("")
    
    # REGIONAL BREAKDOWN
    if 'Region' in df_cir.columns and 'Revenue USD m' in df_cir.columns:
        by_reg = df_cir.groupby('Region')['Revenue USD m'].sum().sort_values(ascending=False)
        context_lines.append("🌍 REGIONAL DISTRIBUTION:")
        for region, rev in by_reg.items():
            pct = (rev / by_reg.sum() * 100)
            context_lines.append(f"  {region}: ${rev:.3f}M ({pct:.1f}%)")
        context_lines.append("")
    
    # STAFFING BY ACCOUNT
    if df_res is not None and 'Client Name' in df_res.columns:
        hc_by_client = df_res.groupby('Client Name').size().sort_values(ascending=False)
        context_lines.append(f"👥 STAFFING ACROSS {len(hc_by_client)} ACCOUNTS:")
        for i, (client, hc) in enumerate(hc_by_client.head(10).items(), 1):
            context_lines.append(f"  {i}. {client}: {hc} employees")
        if len(hc_by_client) > 10:
            context_lines.append(f"  ... and {len(hc_by_client) - 10} more accounts")
        context_lines.append("")
        context_lines.append("📋 COMPLETE CLIENT LIST:")
        for client, hc in hc_by_client.items():
            context_lines.append(f"  • {client}: {hc}")
        context_lines.append("")
    
    context_lines.append("="*70)
    return "\n".join(context_lines)

def suggest_questions(df_cir):
    """Smart suggestions based on available data - VP/SVP focused"""
    suggestions = [
        # Strategic overview
        "How are we doing overall? (Portfolio health check)",
        
        # Revenue questions
        "What's driving our revenue? (Account breakdown)",
        
        # Risk/Concentration
        "Are we over-concentrated with any client? (Risk analysis)",
        
        # Headcount/Efficiency
        "Revenue per employee by account? (Efficiency metrics)",
        
        # Account specific
        "Tell me about ZS Associates (Deep dive on largest account)",
        
        # Comparisons
        "Which accounts are underperforming? (Comparative analysis)",
    ]
    
    if 'GPM' in df_cir.columns or 'GP USD m' in df_cir.columns:
        suggestions.append("Profit margins by account - who's most profitable? (Margin analysis)")
    
    if 'Month' in df_cir.columns:
        suggestions.append("Month-over-month trend - where's the momentum? (Growth analysis)")
    
    return suggestions

def detect_question_type(prompt):
    """Detect type of question for better handling"""
    prompt_lower = prompt.lower()
    
    question_types = {
        'strategic': ['how are we', 'overall', 'portfolio', 'status', 'summary', 'snapshot'],
        'specific_account': ['about', 'for ', 'at ', 'regarding', 'tell me'],
        'comparison': ['vs ', 'compare', 'better', 'worse', 'vs.', 'against', 'difference'],
        'trend': ['trend', 'change', 'momentum', 'growth', 'declining', 'improving'],
        'deep_dive': ['deep dive', 'detail', 'breakdown', 'analyze', 'analysis'],
        'risk': ['risk', 'concentration', 'exposure', 'vulnerable', 'depend on'],
    }
    
    detected = []
    for qtype, keywords in question_types.items():
        if any(kw in prompt_lower for kw in keywords):
            detected.append(qtype)
    
    return detected if detected else ['general']

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
                # Filter by Practices (all should be 'Data and Insights' already)
                if 'Practices' in df_res.columns:
                    df_res = df_res[df_res['Practices'] == 'Data and Insights']
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
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Revenue", f"${metrics.get('revenue', 0):.3f}M")
    col2.metric("Gross Profit (GP)", f"${metrics.get('profit', 0):.3f}M", f"GPM: {metrics.get('gpm', 0):.2f}%")
    col3.metric("Headcount", metrics.get('headcount', 0), "Active employees")
    col4.metric("Accounts", metrics.get('accounts', 0))
    
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
        st.info("📊 Complete headcount data for all accounts is available. Ask about any client!")
        suggestions = suggest_questions(df_cir)
        for suggestion in suggestions:
            st.write(f"• {suggestion}")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Ask about revenue, accounts, regions, margins..."):
        # Apply fuzzy matching to client names in prompt
        matched_prompt = fuzzy_match_client(prompt, df_res)
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            if matched_prompt != prompt:
                st.caption(f"🔍 Recognized: {matched_prompt}")
        
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
                            # Expand synonyms for better understanding
                            synonym_context = expand_synonyms(matched_prompt)
                            
                            system_prompt = f"""You are a senior portfolio analysis expert briefing VPs and SVPs on account intelligence.

YOUR ROLE:
- Analyze portfolio data and provide executive-level insights
- Answer both broad strategic questions AND detailed deep-dives
- When given vague questions, provide the most relevant insights AND suggest specific follow-ups
- Use business language, avoid technical jargon
- Highlight risks, opportunities, and metrics that matter to leadership

DATA AVAILABLE:
{dynamic_context}

RESPONSE GUIDELINES:
1. START with the direct answer to their question
2. PROVIDE context: why this matters, what it means
3. INCLUDE comparisons: vs targets, competitors, or previous periods if relevant
4. HIGHLIGHT outliers: unusually high/low values, concentration risks
5. END with 2-3 suggested follow-up questions they might want to ask

COMMON QUESTIONS YOU'LL RECEIVE:
- "How are we doing?" (open-ended - show key metrics + insights)
- "Tell me about [client]" (specific account deep-dive)
- "What's our revenue?" (direct metric questions)
- "Where should we focus?" (strategic recommendations)
- "Which client is struggling?" (comparative analysis)
- "What happened?" (period-over-period analysis)

{synonym_context}

Remember: Executive users value INSIGHTS over raw data. Interpret findings and explain implications."""
                            
                            messages = [
                                {
                                    "role": "system",
                                    "content": system_prompt
                                },
                                {"role": "user", "content": matched_prompt}  # Use matched_prompt for accurate client lookup
                            ]
                            
                            answer = get_groq_response(client, messages)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            st.markdown(answer)
                            
                            # Add follow-up suggestions
                            st.divider()
                            st.caption("💡 **Suggested follow-ups:**")
                            col1, col2, col3 = st.columns(3)
                            suggested_followups = [
                                "Show me the trend",
                                "Deep dive into top account",
                                "Compare to last month"
                            ]
                            for i, followup in enumerate(suggested_followups):
                                with [col1, col2, col3][i]:
                                    if st.button(followup, key=f"followup_{i}"):
                                        st.session_state.messages.append({"role": "user", "content": followup})
                                        st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Model error: {str(e)}")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")

else:
    st.info("📁 Upload Circle Wise data to start")