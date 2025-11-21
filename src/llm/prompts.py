"""Prompt templates for LLM interactions."""

SYSTEM_PROMPT_ANALYST = """You are an expert business analyst specializing in automotive sales analysis.
You have access to data introspection tools that provide ground truth about the dataset.
Analyze the provided data profile and extract key business insights.
Be precise, data-driven, and focus on actionable findings.
IMPORTANT: Use ONLY the actual column names, values, and statistics provided by the tools."""

SYSTEM_PROMPT_VISUALIZATION = """You are a data visualization expert. Design effective visualizations 
to communicate insights clearly and professionally.
You have access to actual column names and data types from introspection tools.
CRITICAL: Use ONLY the column names provided in the data summary. Do not assume or guess column names."""

SYSTEM_PROMPT_WRITER = """You are a business report writer. Write clear, concise, and professional 
reports that are understandable to C-level executives."""

ANALYSIS_PROMPT_TEMPLATE = """Analyze the dataset using the tool-validated information below:

{data_profile}

The above information comes from data introspection tools that have inspected the actual dataset.
Use ONLY the column names, values, and statistics shown above.

Identify:
1. Key trends over time, specifically by year or region (use actual date column detected).
2. Regional performance patterns, highlighting top and underperforming markets (use actual categorical values found).
3. Top and bottom performing models by sales (based on actual values from tools).
4. Key drivers of sales (e.g., price, market segment, or model type)
5. Electrical vehicle (EV, which consists of Electric and Hybrid Fuel Type) market share and trends over the years.
6. Regional market preferences, an example is below: 
Regional Model Preferences
Different markets favor different models:
North America: i8 leads (sporty EV appeal)
Middle East: X6 and M5 (performance/luxury SUVs)
Europe: i8 and M3 (performance + electric)
Africa: 7 Series and 5 Series (traditional luxury sedans)
Asia: 7 Series and M3 (status + performance)
Insight: Emerging markets prefer traditional sedans, while developed markets show stronger appetite for electric and performance variants.

Provide your analysis as a JSON array of insights, each with:
- insight: The key finding (reference actual column names and values)
- evidence: Supporting data/metrics from tool outputs
- impact: Business implication
- confidence: Score from 0.0 to 1.0

Return ONLY valid JSON, no additional text."""

VISUALIZATION_PLANNING_PROMPT_TEMPLATE = """Based on these insights:

{insights}

And this data summary (includes ACTUAL column names from data introspection tools):
{statistical_summary}

Plan {min_plots}-{max_plots} informative visualizations.

CRITICAL: The data summary above contains the ACTUAL column names available in the dataset.
Use ONLY these column names. Do not assume columns like 'date', 'sales', 'region' exist unless they are listed above.

Suggested Visualization Types:

1. **Temporal Analysis** (if date column exists):
   - Line chart showing trends over time
   - Use the actual date column name from the tools
   - Aggregate by appropriate time period

2. **Comparative Analysis** (for categorical columns):
   - Bar charts showing top/bottom performers
   - Use actual categorical column names and values from tools
   - Apply aggregation='sum' or 'mean' as appropriate

3. **Relationship Analysis** (for numeric columns):
   - Scatter plots showing relationships between numeric variables
   - Use actual numeric column names from statistical summary
   - ALWAYS aggregate to avoid overplotting (use groupby with a categorical column)
   - Correlation heatmap: Use plot_type='heatmap' (auto-generates from numeric columns)

4. **Distribution Analysis**:
   - Box plots or histograms for numeric columns
   - Use actual column names from the data summary

CRITICAL GUIDELINES:
- Use ONLY column names listed in the data summary above
- ALWAYS use aggregations for scatter plots (aggregation='mean' or 'sum')
- For time series: use actual date column name with aggregation='sum'
- For comparisons: use bar charts with aggregation='sum' and appropriate sort/limit
- For scatter plots: aggregate by a categorical variable to avoid overplotting
- Use descriptive titles that explain the business insight
- Avoid plotting raw data points - always aggregate first

For each plot, specify:
{{
    "plot_type": "line|bar|scatter|heatmap|box",
    "title": "Clear, descriptive title explaining the insight",
    "x_axis": "ACTUAL column name from data summary above",
    "y_axis": "ACTUAL column name from data summary above",
    "groupby": "optional grouping column (must be from actual columns)",
    "color": "optional color column (must be from actual columns)",
    "aggregation": "sum|mean|count (REQUIRED for meaningful insights)",
    "sort": "desc|asc|none (for bar charts)",
    "limit": "number (e.g., 10 for top 10, null for all)",
    "description": "what business insight this plot reveals"
}}

Return as JSON array. Output ONLY valid JSON."""

EXECUTIVE_SUMMARY_PROMPT_TEMPLATE = """Write an executive summary (2-3 paragraphs) for this BMW sales analysis.

Key Findings:
{key_findings}

Insights:
{insights}

The summary should:
- Be business-focused and actionable
- Highlight the most critical findings
- Be understandable to C-level executives
- Be 200-300 words

Write in markdown format."""

RECOMMENDATIONS_PROMPT_TEMPLATE = """Based on this analysis:

Insights: {insights}
Key Findings: {key_findings}

Provide 5-7 actionable business recommendations.

Each recommendation should include:
- The recommendation
- Rationale based on data
- Expected impact
- Priority (High/Medium/Low)

Format as markdown bullet points."""
