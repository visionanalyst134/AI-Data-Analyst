import json
from langchain_core.prompts import ChatPromptTemplate


def detect_domain_and_kpis(llm, df):

    columns = df.columns.tolist()
    sample = df.head(5).to_string()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a business data analyst.

Understand the dataset using the column names and sample rows.

Identify:
1. Industry
2. Business function
3. Analytics use case
4. 5 useful business KPIs

Return ONLY valid JSON.

Use this format:

{{
    "industry": "industry name",
    "business_function": "business function",
    "use_case": "analytics use case",

    "kpis": [
        {{
            "name": "KPI name",
            "type": "simple",
            "column": "column_name",
            "aggregation": "sum",
            "percentage": false
        }},

        {{
            "name": "KPI name",
            "type": "ratio",

            "numerator": {{
                "column": "column_name",
                "aggregation": "sum"
            }},

            "denominator": {{
                "column": "column_name",
                "aggregation": "nunique"
            }},

            "percentage": false
        }}
    ]
}}

Allowed KPI types:
simple
ratio

Allowed aggregations:
sum
mean
median
count
nunique
min
max

Rules:

- Use only columns that actually exist in the dataset.
- Do not invent columns.
- Use simple type when one aggregation is enough.
- Use ratio type when one calculated metric must be divided by another.
- For a 0/1 rate, use mean with percentage=true.
- Do not calculate KPI values yourself.
- Only define how Python should calculate them.
- Industry means company/business sector.
- Business function means Sales, HR, Finance, Marketing, Operations, etc.
- Use case means the analytics problem being studied.
- If industry cannot be identified, use "Unknown".
- Return exactly 5 useful KPIs.
- Do not write explanations outside JSON.
"""
        ),
        (
            "human",
            """
Columns:
{columns}

Sample rows:
{sample}
"""
        )
    ])

    final_prompt = prompt.format_messages(
        columns=columns,
        sample=sample
    )

    response = llm.invoke(final_prompt)
    content = response.content.strip()
    content = content.replace("```json", "")
    content = content.replace("```", "")
    content = content.strip()

    print("RAW LLM RESPONSE:")
    print(content)
    
    try:
        result = json.loads(content)

    except json.JSONDecodeError as e:
        print("JSON ERROR:", e)

        result = {
            "industry": "Unknown",
            "business_function": "Unknown",
            "use_case": "Unknown",
            "kpis": []
        }

    return result
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
