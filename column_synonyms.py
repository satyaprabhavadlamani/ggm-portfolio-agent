# Column synonyms for NLP understanding
COLUMN_SYNONYMS = {
    # Resource columns
    "employee": ["Emp No", "Employee ID", "resource_id"],
    "name": ["Candidate Name", "Employee Name"],
    "client": ["Client Name", "Account", "Customer"],
    "region": ["Region", "Geography", "Location"],
    "practice": ["Practices", "Practice Area", "Domain"],
    "job_role": ["Job category", "Role", "Position"],
    "sow": ["SowCategory", "SOW", "Engagement Type"],

    
    # Circle Wise columns (FILL THESE IN)
    "revenue": ["Revenue USD m", "Revenue", "Sales", "Income"],
    "cost": ["Cost USD m", "Expense", "Cost"],
    "profit": ["GP USD m", "Gross Profit", "Margin"],
    "gpm": ["GPM", "Gross Profit Margin %", "Margin %"],
    "month": ["Month", "Date", "Period"],
    
    # Metrics
    "headcount": ["HC", "Headcount", "Employees", "FTE", "Resource Count"],
    "accounts": ["Clients", "Customers", "Accounts", "Partners"],
}

def get_column_for_query(user_query: str, df_columns):
    """Match user query to actual column name"""
    query_lower = user_query.lower()
    
    for synonym_key, synonyms in COLUMN_SYNONYMS.items():
        for syn in synonyms:
            if syn.lower() in query_lower:
                # Find matching column in dataframe
                for col in df_columns:
                    if col.lower() in [s.lower() for s in synonyms]:
                        return col
    return None