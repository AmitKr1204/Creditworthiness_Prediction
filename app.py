# import streamlit as st
# import pandas as pd
# import joblib

# model = joblib.load("extra_trees_credit_model.pkl")
# encoders = {col : joblib.load(f"{col}_encoder.pkl") for col in ["Sex", "Housing", "Saving accounts", "Checking account"]}

# st.title("Credit Worthiness")
# st.write("Enter applicant information to predict if the risk is good or bad")

# age = st.number_input("Age", min_value = 18, max_value =60, value = 25)
# sex = st.selectbox("Sex", ["male", "female"])
# job = st.number_input("Job (0-3)", min_value = 0, max_value = 3, value = 1)
# housing = st.selectbox("Housing",["own","rent","free"])
# saving_accounts = st.selectbox("Saving Accounts",["little","moderate","rich","quite rich"])
# checking_account = st.selectbox("Checking Account",["little","moderate","rich"])
# credit_amount = st.number_input("Credit amount", min_value = 0, value = 100)
# duration = st.number_input("Duration (months)",min_value= 1, value = 12)


# input_df = pd.DataFrame({
#     "Age":[age],
#     "Sex":[encoders["Sex"].transform([sex])[0]],
#     "Job":[job],
#     "Housing" : [encoders["Housing"].transform([housing])[0]],
#     "Saving accounts" : [encoders["Saving accounts"].transform([saving_accounts])[0]],
#     "Checking account" : [encoders["Checking account"].transform([checking_account])[0]],
#     "Credit amount" : [credit_amount],
#     "Duration" : [duration]

# })

# if st.button("Predict Risk"):
#     pred = model.predict(input_df)[0]
    
#     if pred == 1:
#         st.success("The predicted credit Risk is : **GOOD**")
#     else:
#         st.error("The predicted credit Risk is : **BAD**")


import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import numpy as np

# ==========================
# 🎯 Page Configuration
# ==========================
st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="💳",
    layout="centered"
)

# ==========================
# 🎨 Enhanced CSS Styling
# ==========================
st.markdown("""
    <style>
    /* Background gradient */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f0f4ff 0%, #f9fafb 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e0e7ff, #f1f5f9);
    }

    /* Header styling */
    .main-title {
        text-align: center;
        color: #1e3a8a;
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }
    .sub-title {
        text-align: center;
        color: #475569;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    /* Button design */
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
        height: 2.8rem;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #2563eb, #1e40af);
        transform: scale(1.02);
    }

    /* Card styling */
    .card {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        padding: 1.2rem;
        margin-top: 1rem;
        text-align: center;
    }

    /* Result text */
    .good-result {
        color: #166534;
        font-size: 1.4rem;
        font-weight: 700;
    }
    .bad-result {
        color: #991b1b;
        font-size: 1.4rem;
        font-weight: 700;
    }

    /* Gauge container */
    .gauge-box {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        padding: 1rem;
        margin-top: 1rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# 🧠 Load Model and Encoders
# ==========================
model = joblib.load("extra_trees_credit_model.pkl")
encoders = {
    col: joblib.load(f"{col}_encoder.pkl")
    for col in ["Sex", "Housing", "Saving accounts", "Checking account"]
}

# ==========================
# 🏦 Header
# ==========================
st.markdown("<h1 class='main-title'>💳 Credit Risk Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Analyze an applicant’s financial profile and predict their creditworthiness with confidence.</p>", unsafe_allow_html=True)

# ==========================
# 🧾 Sidebar Input
# ==========================
with st.sidebar:
    st.header("⚙️ Input Parameters")

    age = st.slider("Age", 18, 60, 30)
    sex = st.radio("Sex", ["male", "female"])
    job = st.slider("Job (0-3)", 0, 3, 1)
    housing = st.selectbox("Housing", ["own", "rent", "free"])
    saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich"])
    checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich"])
    credit_amount = st.number_input("Credit Amount", min_value=0, value=1200)
    duration = st.slider("Duration (months)", 1, 72, 12)

# ==========================
# 📊 Prepare Input Data
# ==========================
input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [job],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration]
})

# ==========================
# 🔮 Prediction
# ==========================
if st.button("🔍 Predict Credit Risk"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else (0.8 if pred == 1 else 0.2)
    score = int(prob * 100)


# Gauge with Needle
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Credit Score", 'font': {'size': 18, 'color': '#1e3a8a'}},
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "rgba(0,0,0,0)"},  # hide bar (we’ll add needle)
            'bgcolor': "white",
            'borderwidth': 1.5,
            'bordercolor': "#ccc",
            'steps': [
                {'range': [0, 40], 'color': '#fee2e2'},
                {'range': [40, 70], 'color': '#fef3c7'},
                {'range': [70, 100], 'color': '#dcfce7'}
            ],
        }
    ))

    # 🧭 Add Needle (hand)
    angle = (score / 100) * 180  # convert score to degree (half circle)
    radians = np.deg2rad(180 - angle)
    x_head = 0.5 + 0.4 * np.cos(radians)
    y_head = 0.5 + 0.4 * np.sin(radians)

    fig.add_shape(
        type='line',
        x0=0.5, y0=0.5,
        x1=x_head, y1=y_head,
        line=dict(color="#1e3a8a", width=4)
    )

    # Needle center dot
    fig.add_shape(
        type="circle",
        xref="paper", yref="paper",
        x0=0.48, y0=0.48, x1=0.52, y1=0.52,
        fillcolor="#1e3a8a", line_color="#1e3a8a"
    )

    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=40, b=0),
        paper_bgcolor="white",
        font={'color': "#1e3a8a", 'family': "sans-serif"}
)

    # ==========================
    # 🎨 Layout Columns
    # ==========================
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='gauge-box'>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if pred == 1:
            st.markdown(f"""
            <div class='card'>
                <div class='good-result'>✅ Credit Risk: GOOD</div>
                <p>This applicant is financially reliable.<br>Creditworthiness score: <b>{score}%</b></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='card'>
                <div class='bad-result'>❌ Credit Risk: BAD</div>
                <p>This applicant shows higher financial risk.<br>Creditworthiness score: <b>{score}%</b></p>
            </div>
            """, unsafe_allow_html=True)

    # ==========================
    # 📋 Summary Table
    # ==========================
    st.markdown("### 🧾 Applicant Summary")
    st.dataframe(input_df.style.highlight_max(axis=0, color="#dbeafe"))
