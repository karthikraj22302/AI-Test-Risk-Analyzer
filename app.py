import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.sparse import hstack
import plotly.express as px

import google.generativeai as genai

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="TestRisk AI ", layout="wide")

# ----------------------------
# DARK THEME + KPI COLORS
# ----------------------------
st.markdown("""
<style>
body {background-color: #0e1117; color: white;}

[data-testid="stMetric"] {
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    color: white;
}

div[data-testid="stMetric"]:nth-of-type(1) {
    background: linear-gradient(135deg, #ff4b4b, #b30000);
}
div[data-testid="stMetric"]:nth-of-type(2) {
    background: linear-gradient(135deg, #ff9f43, #ff6f00);
}
div[data-testid="stMetric"]:nth-of-type(3) {
    background: linear-gradient(135deg, #00c9a7, #00796b);
}

thead tr th, tbody tr td {
    text-align: center !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 BugSense AI — Test Risk Analyzer")
st.subheader("TestRisk AI  Smart QA Console")
# ----------------------------
# GEMINI API
# ----------------------------
st.sidebar.header("🔑 Gemini API")
api_key = st.sidebar.text_input("Enter API Key", type="password")

if api_key:
    genai.configure(api_key=api_key)

# ----------------------------
# FILE UPLOAD
# ----------------------------
file = st.file_uploader("📂 Upload Bug CSV", type=["csv"])

if file:

    try:
        df = pd.read_csv(file, encoding='utf-8')
    except:
        df = pd.read_csv(file, encoding='latin-1')

    df.columns = df.columns.str.strip().str.lower()

    if not all(col in df.columns for col in ['priority','component','summary']):
        st.error("CSV must contain: priority, component, summary")
        st.stop()

    df['priority'] = df['priority'].astype(str).str.lower()
    df['summary'] = df['summary'].fillna("")

    # ----------------------------
    # FILTERS
    # ----------------------------
    st.sidebar.header("🔎 Filters")

    comp_filter = st.sidebar.multiselect(
        "Component", df['component'].unique(), default=df['component'].unique()
    )

    prio_filter = st.sidebar.multiselect(
        "Priority", df['priority'].unique(), default=df['priority'].unique()
    )

    df = df[(df['component'].isin(comp_filter)) & (df['priority'].isin(prio_filter))]

    # ----------------------------
    # RISK CALCULATION
    # ----------------------------
    weights = {'blocker':3,'high':2,'medium':1,'low':0}
    df['RiskWeight'] = df['priority'].map(weights).fillna(0)

    grouped = df.groupby('component').agg(
        Total_Bugs=('RiskWeight','count'),
        Blocker_Bugs=('priority', lambda x: (x == 'blocker').sum()),
        High_Bugs=('priority', lambda x: (x == 'high').sum()),
        Risk_Score=('RiskWeight','sum')
    ).reset_index()

    # ----------------------------
    # KPI
    # ----------------------------
    c1,c2,c3 = st.columns(3)
    c1.metric("🐞 Total Bugs", len(df))
    c2.metric("⚠️ High Risk Bugs", len(df[df['priority'].isin(['blocker','high'])]))
    c3.metric("🧩 Components", df['component'].nunique())

    st.markdown("---")

    # ----------------------------
    # 🤖 AI FUNCTION (BOLD TEST TYPE)
    # ----------------------------
    def get_ai_strategy(component, bugs, model_ai):
        prompt = f"""
        You are a senior QA expert.

        Analyze bugs for component: {component}

        Bugs:
        {bugs}

        Give output in this format:

        **Testing Type**: (Regression / Smoke / Sanity / Integration / Security)

        Focus Areas:
        - point 1
        - point 2

        Reason:
        short explanation
        """

        res = model_ai.generate_content(prompt)
        return res.text.strip()

    # ----------------------------
    # AI GENERATION
    # ----------------------------
    grouped['AI_Testing_Strategy'] = "⚠️ Not generated"
    if api_key:

        try:
            model_ai = genai.GenerativeModel("gemini-2.5-flash")
            st.success("✅ Gemini Connected")
        except Exception as e:
            st.error(f"❌ Model Error: {e}")

        strategies = []

        for comp in grouped['component']:
            bug_text = " ".join(df[df['component']==comp]['summary'].head(10))[:1500]
            try:
                out = get_ai_strategy(comp, bug_text, model_ai)
                print("AI OUTPUT:", out)  # 👈 DEBUG
            except Exception as e:
                out = f"❌ AI Error: {str(e)}"

            strategies.append(out)

        grouped['AI_Testing_Strategy'] = strategies
    else:
        grouped['AI_Testing_Strategy'] = "Enter API key"

    # ----------------------------
    # TABLE
    # ----------------------------
    st.subheader("📊 Risk-Based Test Plan (AI)")
    st.dataframe(grouped, use_container_width=True)

    # ----------------------------
    # EXPAND VIEW (BEST UI)
    # ----------------------------
    st.subheader("🧠 AI Testing Strategy Details")

    for i,row in grouped.iterrows():
        with st.expander(f"🧩 {row['component']}"):
            st.markdown(row['AI_Testing_Strategy'])

    # ----------------------------
    # ML MODEL
    # ----------------------------
    df['Is_High_Risk'] = np.where(df['priority'].isin(['blocker','high']),1,0)

    le = LabelEncoder()
    df['ComponentEncoded'] = le.fit_transform(df['component'])

    vec = TfidfVectorizer(max_features=100)
    txt = vec.fit_transform(df['summary'])

    X = hstack([txt, df[['ComponentEncoded']]])
    y = df['Is_High_Risk']

    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3)

    model = RandomForestClassifier()
    model.fit(Xtr,ytr)

    pred = model.predict(Xte)

    st.subheader("🧠 ML Prediction")
    st.text(classification_report(yte,pred))

    # ----------------------------
    # 📈 GRAPH (AUTO FIXED)
    # ----------------------------

    st.subheader("📊 Risk Distribution")

    fig = px.pie(
        grouped,
        names='component',
        values='Risk_Score',
        title="Risk Score by Component"
    )

    fig.update_layout(
        template="plotly_dark",
         height=600,
         width=900,
         
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # DATA PREVIEW
    # ----------------------------

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())
