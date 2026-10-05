import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
from difflib import SequenceMatcher

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
    'revenue': ['sales', 'top line', 'topline', 'income', 'earnings', 'throughput', 'business', 'volume'],
    'revenue_amount': ['revenue', 'sales', 'top line', 'generated', 'brought in'],
    'profit': ['gp', 'gross profit', 'margin', 'bottomline', 'bottom line', 'earnings', 'returns', 'benefit'],
    'gp_amount': ['profit', 'gp', 'gross profit', 'gains'],
    'gpm': ['margin', 'profitability', 'margin %', 'percentage', 'efficiency', 'returns', 'yield'],
    'headcount': ['staff', 'team', 'people', 'employees', 'strength', 'bench', 'resources', 'workforce', 'fte'],
    'headcount_amount': ['headcount', 'staff', 'people', 'team size', 'strength'],
    'account': ['client', 'customer', 'partner', 'engagement', 'project', 'contract', 'business'],
    'client_name': ['account', 'client', 'customer', 'project', 'engagement'],
    'region': ['geography', 'location', 'zone', 'area', 'market', 'country', 'place'],
    'mtd': ['month', 'current month', 'this month', 'monthly'],
    'qtd': ['quarter', 'quarterly', 'this quarter', 'current quarter'],
    'ytd': ['year', 'annual', 'yearly', 'this year', 'current year'],
    'growth': ['increase', 'improvement', 'trend', 'change', 'momentum', 'uptick'],
    'decline': ['decrease', 'drop', 'fall', 'downturn', 'reduction', 'dip'],
    'top': ['best', 'largest', 'biggest', 'highest', 'leading', 'major'],
    'bottom': ['worst', 'smallest', 'lowest', 'trailing', 'weakest'],
}

# ===== UTILITY FUNCTIONS =====

def sort_months_calendar_order(months):
    """Sort months in calendar order (Jan -> Dec), NOT alphabetically"""
    if isinstance(months, np.ndarray):
        months = list(months)
    return sorted(months, key=lambda m: MONTH_ORDER.get(m, 99))

def expand_synonyms(text):
    """Expand user query with synonym explanations for AI"""
    expanded_context = "\n[SYNONYM CONTEXT]: "
    for main_term, synonyms in SYNONYM_MAP.items():
        for syn in synonyms:
            if syn.lower() in text.lower():
                expanded_context += f"'{syn}' = {main_term}; "
    return expanded_context if expanded_context != "\n[SYNONYM CONTEXT]: " else ""

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

def get_available_metrics(df_cir, df_res):
    """Dynamically identify available metrics"""
    metrics = {}
    
    if 'Revenue USD m' in df_cir.columns:
        metrics['revenue'] = df_cir['Revenue USD m'].sum()
    
    if 'Cost USD m' in df_cir.columns:
        metrics['cost'] = df_cir['Cost USD m'].sum()
    
    if 'GP USD m' in df_cir.columns:
        metrics['profit'] = df_cir['GP USD m'].sum()
    
    if metrics.get('revenue') and metrics.get('revenue') > 0:
        metrics['gpm'] = (metrics['profit'] / metrics['revenue'] * 100)
    else:
        metrics['gpm'] = 0
    
    if df_res is not None and len(df_res) > 0:
        metrics['headcount'] = len(df_res)
        metrics['active'] = len(df_res)
    else:
        metrics['headcount'] = 0
        metrics['active'] = 0
    
    if df_res is not None and 'Client Name' in df_res.columns:
        metrics['accounts'] = df_res['Client Name'].nunique()
    else:
        metrics['accounts'] = 0
    
    return metrics

def generate_dynamic_context(df_cir, df_res, metrics, df_cir_raw=None, selected_month=None):
    """Generate comprehensive executive-level context for AI"""
    context_lines = ["="*70, "PORTFOLIO INTELLIGENCE DASHBOARD", "="*70, ""]
    
    if df_cir_raw is not None and 'Month' in df_cir_raw.columns:
        available_months = sort_months_calendar_order(df_cir_raw['Month'].unique())
        context_lines.append(f"📅 DATASET CONTAINS {len(available_months)} MONTHS OF HISTORICAL DATA:")
        context_lines.append(f"   Available months (calendar order): {', '.join(map(str, available_months))}")
        context_lines.append(f"   Currently analyzing: {selected_month if selected_month else 'Latest month'}")
        
        if selected_month and len(available_months) > 1:
            available_months_list = list(available_months)
            if selected_month in available_months_list:
                idx = available_months_list.index(selected_month)
                if idx > 0:
                    prev_month = available_months_list[idx - 1]
                    context_lines.append(f"   Can compare to: {prev_month} (previous month in calendar order)")
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
    
    context_lines.append("💡 KEY INSIGHTS:")
    if metrics.get('headcount') and metrics.get('revenue'):
        rev_per_emp = metrics['revenue'] / (metrics['headcount'] / 1000) if metrics['headcount'] > 0 else 0
        context_lines.append(f"  Revenue per Employee: ${rev_per_emp:.2f}K")
    if metrics.get('gpm') and metrics['gpm'] < 5:
        context_lines.append(f"  ⚠️  ALERT: Low margin ({metrics['gpm']:.2f}%) - review cost structure")
    elif metrics.get('gpm') and metrics['gpm'] > 20:
        context_lines.append(f"  ✅ Strong margin ({metrics['gpm']:.2f}%) - healthy profitability")
    
    if 'Revenue USD m' in df_cir.columns and 'Client' in df_cir.columns:
        by_acct = df_cir.groupby('Client')['Revenue USD m'].sum().sort_values(ascending=False)
        if len(by_acct) > 0:
            top_client = by_acct.index[0]
            top_rev = by_acct.iloc[0]
            concentration = (top_rev / by_acct.sum() * 100) if by_acct.sum() > 0 else 0
            context_lines.append(f"  📌 Top Account: {top_client} (${top_rev:.3f}M, {concentration:.1f}% of revenue)")
            if concentration > 40:
                context_lines.append(f"  ⚠️  HIGH CONCENTRATION - Over-reliance on {top_client}")
    
    context_lines.append("")
    context_lines.append("="*70)
    return "\n".join(context_lines)

def resolve_month_reference(user_query, available_months):
    """Convert month references to actual month values (calendar order)"""
    if not available_months:
        return None
    
    available_months_sorted = sort_months_calendar_order(available_months)
    
    month_mapping = {
        'jan': 'Jan', 'january': 'Jan', 'feb': 'Feb', 'february': 'Feb',
        'mar': 'Mar', 'march': 'Mar', 'apr': 'Apr', 'april': 'Apr', 'may': 'May',
        'jun': 'Jun', 'june': 'Jun', 'jul': 'Jul', 'july': 'Jul',
        'aug': 'Aug', 'august': 'Aug', 'sep': 'Sep', 'september': 'Sep',
        'oct': 'Oct', 'october': 'Oct', 'nov': 'Nov', 'november': 'Nov',
        'dec': 'Dec', 'december': 'Dec',
    }
    
    query_lower = user_query.lower()
    
    for key, month_val in month_mapping.items():
        if key in query_lower and month_val in available_months_sorted:
            return month_val
    
    if 'last month' in query_lower and len(available_months_sorted) > 1:
        return available_months_sorted[-2]
    
    if 'previous month' in query_lower and len(available_months_sorted) > 1:
        return available_months_sorted[-2]
    
    if 'this month' in query_lower or 'current month' in query_lower:
        return available_months_sorted[-1]
    
    return None

def get_month_comparison(df_cir_raw, current_month):
    """Calculate month-over-month comparison (calendar order)"""
    if df_cir_raw is None or 'Month' not in df_cir_raw.columns:
        return None
    
    available_months = sort_months_calendar_order(df_cir_raw['Month'].unique())
    available_months_list = list(available_months)
    
    try:
        current_idx = available_months_list.index(current_month)
    except ValueError:
        return None
    
    if current_idx <= 0:
        return None
    
    prev_month = available_months_list[current_idx - 1]
    
    if 'Circle' in df_cir_raw.columns:
        df_curr = df_cir_raw[(df_cir_raw['Month'] == current_month) & (df_cir_raw['Circle'] == GGM_CIRCLE)]
        df_prev = df_cir_raw[(df_cir_raw['Month'] == prev_month) & (df_cir_raw['Circle'] == GGM_CIRCLE)]
    else:
        df_curr = df_cir_raw[df_cir_raw['Month'] == current_month]
        df_prev = df_cir_raw[df_cir_raw['Month'] == prev_month]
    
    if 'Revenue USD m' in df_curr.columns:
        curr_rev = df_curr['Revenue USD m'].sum()
        prev_rev = df_prev['Revenue USD m'].sum()
        growth = ((curr_rev - prev_rev) / prev_rev * 100) if prev_rev > 0 else 0
        
        return {
            'prev_month': prev_month,
            'prev_revenue': prev_rev,
            'curr_revenue': curr_rev,
            'growth_pct': growth
        }
    
    return None

def suggest_questions(df_cir):
    """Smart suggestions based on available data"""
    suggestions = [
        "How are we doing overall? (Portfolio health check)",
        "What's driving our revenue? (Account breakdown)",
        "Which accounts are growing fastest?",
        "How has [client] been performing?",
        "Show me regional breakdown",
        "What's our margin trend?",
        "Where should we focus resources?",
    ]
    return suggestions

def get_groq_response(client, messages):
    """Get response from Groq with fallback logic and detailed error logging"""
    from datetime import datetime, timedelta
    
    # Track request for rate limiting
    now = datetime.now()
    st.session_state.groq_requests.append(now)
    
    # Clean up old requests (older than 1 minute)
    one_minute_ago = now - timedelta(minutes=1)
    st.session_state.groq_requests = [t for t in st.session_state.groq_requests if t > one_minute_ago]
    
    requests_in_last_minute = len(st.session_state.groq_requests)
    
    # Show rate limit warning if approaching limit
    if requests_in_last_minute > 25:
        st.warning(f"⚠️ Rate limit approaching: {requests_in_last_minute}/30 requests in last minute")
    
    if requests_in_last_minute >= 30:
        st.error(f"🚫 Rate limit exceeded: {requests_in_last_minute}/30 requests in last minute\n\nPlease wait 1 minute before asking more questions")
        return None
    
    errors = []
    
    for model in GROQ_MODELS:
        try:
            st.write(f"🔄 Trying model: {model}...")
            
            # Try modern API first (groq >= 0.4.0)
            if hasattr(client, 'messages'):
                response = client.messages.create(
                    model=model,
                    messages=messages,
                    max_tokens=1024,
                )
            # Fallback for older groq versions
            elif hasattr(client, 'chat') and hasattr(client.chat, 'completions'):
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=1024,
                )
            else:
                raise AttributeError("Groq client missing both 'messages' and 'chat.completions' APIs")
            
            if hasattr(response, 'content'):
                # Modern API response
                if response.content and len(response.content) > 0:
                    st.write(f"✅ Success with {model}")
                    return response.content[0].text
            elif hasattr(response, 'choices'):
                # Legacy API response
                if response.choices and len(response.choices) > 0:
                    st.write(f"✅ Success with {model}")
                    return response.choices[0].message.content
            else:
                error_msg = "No content in response"
                errors.append(f"  • {model}: {error_msg}")
                st.write(f"⚠️ {model}: {error_msg}")
                
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            errors.append(f"  • {model}: [{error_type}] {error_msg}")
            st.write(f"❌ {model} failed: {error_type}")
            st.write(f"   Details: {error_msg}")
    
    # All models failed - provide diagnostic info
    error_summary = "\n".join(errors) if errors else "Unknown error"
    
    st.error(f"""⚠️ **All Groq models failed**

**Error Details:**
{error_summary}

**Likely cause:** Groq library version mismatch

**Quick fix:**
1. Open PowerShell
2. Run: `pip install --upgrade groq`
3. Refresh the Streamlit app (F5)
4. Click "Test Groq Connection" again

**If still failing:**
1. Check Groq API key is valid (start with gsk_)
2. Check Groq service status: https://status.groq.com/
3. Verify internet connectivity
4. Try waiting 1-2 minutes if rate limited
""")
    
    return None

# ===== MAIN APP =====

df_res = None
df_cir = None
df_cir_raw = None
selected_month = None
data_loaded = False

if "messages" not in st.session_state:
    st.session_state.messages = []
if "process_followup" not in st.session_state:
    st.session_state.process_followup = None
if "groq_requests" not in st.session_state:
    st.session_state.groq_requests = []  # Track timestamps of requests

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Diagnostics")

# Show rate limit status
if "groq_requests" in st.session_state:
    from datetime import datetime, timedelta
    
    now = datetime.now()
    one_minute_ago = now - timedelta(minutes=1)
    requests_in_last_minute = len([t for t in st.session_state.groq_requests if t > one_minute_ago])
    
    if requests_in_last_minute == 0:
        st.sidebar.success(f"📊 Requests (1min): {requests_in_last_minute}/30 ✅")
    elif requests_in_last_minute < 25:
        st.sidebar.info(f"📊 Requests (1min): {requests_in_last_minute}/30")
    elif requests_in_last_minute < 30:
        st.sidebar.warning(f"📊 Requests (1min): {requests_in_last_minute}/30 ⚠️")
    else:
        st.sidebar.error(f"📊 Requests (1min): {requests_in_last_minute}/30 🚫 LIMIT HIT")

# Test Groq Connection
if st.sidebar.button("🧪 Test Groq Connection"):
    groq_api_key = st.secrets.get("groq", {}).get("api_key")
    
    if not groq_api_key:
        st.sidebar.error("❌ No Groq API key in secrets")
    else:
        try:
            from groq import Groq
            client = Groq(api_key=groq_api_key)
            
            st.sidebar.info("Testing connection to Groq...")
            
            # Try modern API first
            try:
                response = client.messages.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": "Say 'OK' in one word"}],
                    max_tokens=50,
                )
                result = response.content[0].text if response.content else "No response"
            except AttributeError:
                # Fallback for older groq library versions
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": "Say 'OK' in one word"}],
                    max_tokens=50,
                )
                result = response.choices[0].message.content if response.choices else "No response"
            
            st.sidebar.success(f"✅ Groq is reachable!\n\nResponse: {result}")
                
        except AttributeError as e:
            st.sidebar.error(f"❌ Library version issue:\n\n{type(e).__name__}\n\nFix: `pip install --upgrade groq`\n\nError: {str(e)}")
        except Exception as e:
            st.sidebar.error(f"❌ Connection failed:\n\n{type(e).__name__}\n\n{str(e)}")

st.sidebar.markdown("---")
data_source = st.sidebar.radio("Data Source", ["Upload Files", "OneDrive"])

if data_source == "Upload Files":
    resource_file = st.sidebar.file_uploader("Resource Data (Optional)", type=['xlsx'])
    circle_file = st.sidebar.file_uploader("Circle Wise Data (Required)", type=['xlsx'])
    
    if circle_file:
        try:
            df_cir_raw = pd.read_excel(circle_file, sheet_name='Sheet1', header=0)
            st.sidebar.write(f"📊 Loaded {len(df_cir_raw)} records")
            
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
            
            if 'Circle' in df_cir.columns:
                df_cir = df_cir[df_cir['Circle'] == GGM_CIRCLE]
                st.sidebar.write(f"✅ Filtered: {len(df_cir)} {GGM_CIRCLE} records")
            
            if resource_file:
                df_res = pd.read_excel(resource_file, sheet_name=0, header=0)
                if 'Practices' in df_res.columns:
                    df_res = df_res[df_res['Practices'] == GGM_CIRCLE]
                st.sidebar.write(f"👥 Loaded {len(df_res)} resources")
            
            data_loaded = True
            
        except Exception as e:
            st.sidebar.error(f"❌ Error loading data: {str(e)}")

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
    
    metrics = get_available_metrics(df_cir, df_res)
    
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
    
    # Check if a follow-up button was clicked (will have set process_followup in session state)
    prompt = None
    if st.session_state.process_followup:
        prompt = st.session_state.process_followup
        st.session_state.process_followup = None
    else:
        prompt = st.chat_input("Ask about revenue, accounts, regions...")
    
    if prompt:
        matched_prompt = fuzzy_match_client(prompt, df_res)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            if matched_prompt != prompt:
                st.caption(f"🔍 Recognized: {matched_prompt}")
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    dynamic_context = generate_dynamic_context(df_cir, df_res, metrics, df_cir_raw, selected_month)
                    groq_api_key = st.secrets.get("groq", {}).get("api_key")
                    
                    if not groq_api_key:
                        st.error("❌ Groq API key not configured")
                    else:
                        from groq import Groq
                        client = Groq(api_key=groq_api_key)
                        
                        synonym_context = expand_synonyms(matched_prompt)
                        system_prompt = f"""You are a senior portfolio analyst. Provide executive-level insights on the portfolio data below.

DATA:
{dynamic_context}

GUIDELINES:
1. Start with direct answer
2. Provide context and implications
3. Include comparisons and insights
4. End with 2-3 follow-up suggestions

{synonym_context}"""
                        
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": matched_prompt}
                        ]
                        
                        answer = get_groq_response(client, messages)
                        
                        if answer:
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            st.markdown(answer)
                        
                        st.divider()
                        st.caption("💡 **Suggested follow-ups:**")
                        col1, col2, col3 = st.columns(3)
                        followups = ["Show me the trend", "Deep dive into top account", "Compare to last month"]
                        
                        for i, followup in enumerate(followups):
                            with [col1, col2, col3][i]:
                                if st.button(followup, key=f"followup_{i}"):
                                    # Set flag to process this message on rerun
                                    st.session_state.process_followup = followup
                                    st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

else:
    st.info("📁 Upload Circle Wise data to start")