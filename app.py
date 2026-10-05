import streamlit as st
import pandas as pd
import duckdb

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent

from domain_engine import detect_domain_and_kpis
from kpi_engine import calculate_kpis


load_dotenv()


# ---------------- PAGE ----------------

st.set_page_config(
    page_title="AI Data Analyst",
    layout="wide"
)

st.title("AI Data Analyst")


# ---------------- DATA ----------------

uploaded_file = st.file_uploader(
    "Upload cleaned CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file.")
    st.stop()


df = pd.read_csv(uploaded_file)

# ---------------- LLM ----------------

@st.cache_resource
def get_llm():

    return ChatGroq(
        model="qwen/qwen3.8-27b",
        temperature=0,
        max_tokens=700
    )


llm = get_llm()


# ---------------- DOMAIN ----------------

domain_result = detect_domain_and_kpis(
    llm,
    df
)


# ---------------- KPI ----------------

kpi_results = calculate_kpis(
    df,
    domain_result["kpis"]
)


# ---------------- AGENT ----------------

agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    allow_dangerous_code=True
)


# ---------------- DOMAIN INFO ----------------

st.subheader("Dataset Understanding")

st.write("Industry:", domain_result["industry"])
st.write("Business Function:", domain_result["business_function"])
st.write("Use Case:", domain_result["use_case"])


# ---------------- KPI CARDS ----------------

st.subheader("Key Business KPIs")

cols = st.columns(len(kpi_results))

if kpi_results:

    cols = st.columns(len(kpi_results))

    for col, kpi in zip(cols, kpi_results):

        value = kpi["value"]

        if value is None:
            display_value = "N/A"

        elif kpi["percentage"]:
            display_value = f"{value:.2f}%"

        else:
            display_value = round(value, 2)

        col.metric(
            label=kpi["name"],
            value=display_value
        )

else:
    st.warning("No KPIs generated.")


# ---------------- TABS ----------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Data Overview",
    "Ask AI",
    "SQL",
    "Summary"
])


# ---------------- DATA OVERVIEW ----------------

with tab1:

    st.subheader("Data Overview")

    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Column Names")
    st.write(list(df.columns))

    st.subheader("Data Types")
    st.write(df.dtypes)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Duplicate Rows")
    st.write(df.duplicated().sum())


# ---------------- ASK AI ----------------

with tab2:

    user_query = st.text_input(
        "Ask a question about the dataset"
    )

    if user_query:

        response = agent.invoke({
            "input": user_query
        })

        st.subheader("Answer")
        st.write(response["output"])


# ---------------- SQL ----------------

with tab3:

    st.subheader("SQL Analysis")

    st.write("Table name: data")

    sql_query = st.text_area(
        "Write SQL Query",
        "SELECT * FROM data LIMIT 5"
    )

    if st.button("Run SQL"):

        con = duckdb.connect()

        con.register(
            "data",
            df
        )

        result = con.execute(
            sql_query
        ).df()

        st.dataframe(result)


# ---------------- AI SUMMARY ----------------

with tab4:

    st.subheader("AI Summary")

    if st.button("Generate Summary"):

        summary_prompt = f"""
You are a business data analyst.

Industry:
{domain_result["industry"]}

Business Function:
{domain_result["business_function"]}

Use Case:
{domain_result["use_case"]}

Calculated KPIs:
{kpi_results}

Write a short business summary based only on these KPI values.

Include:

1. Key Metrics
2. Important Observations
3. Areas to Investigate

Rules:

- Use only the provided KPI values.
- Do not calculate new metrics.
- Do not invent values, benchmarks, currency symbols, or units.
- Do not classify values as high, low, good, bad, significant, weak, strong, substantial, large, small, or poor.
- Do not infer business impact unless directly supported by the provided KPIs.
- Do not claim correlation unless it was explicitly calculated.
- Do not claim causation or use phrases such as "driven by", "caused by", or "led to".
- Clearly separate factual observations from areas that require further investigation.
- Keep the summary concise.
"""

        response = llm.invoke(summary_prompt)

        st.write(response.content)