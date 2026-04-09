import streamlit as st
import joblib
import pandas as pd

# -------------------------------
# LOAD MODEL
# -------------------------------
model = joblib.load("model.pkl")

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Churn Predictor", layout="wide")

# -------------------------------
# PREMIUM CSS 🔥
# -------------------------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

/* Title */
.title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Glass Card */
.card {
    background: rgba(255,255,255,0.05);
    padding: 30px;
    border-radius: 20px;
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    margin-bottom: 20px;
}

/* Section Title */
.section-title {
    font-size: 20px;
    margin-bottom: 10px;
    color: #38bdf8;
}

/* Button */
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 12px;
    height: 55px;
    font-size: 18px;
}

/* Result Box */
.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# TITLE
# -------------------------------
st.markdown('<div class="title">🤖 AI Customer Churn Predictor</div>', unsafe_allow_html=True)

# -------------------------------
# MAIN CARD
# -------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

# -------------------------------
# BASIC INFO
# -------------------------------
with col1:
    st.markdown('<div class="section-title">👤 Customer Info</div>', unsafe_allow_html=True)

    gender = st.radio("Gender", ["Male", "Female"])
    senior = st.radio("Age Group", ["Below 60", "Above 60"])
    partner = st.radio("Has Partner?", ["No", "Yes"])
    dependents = st.radio("Has Family Dependents?", ["No", "Yes"])
    tenure = st.slider("How long with company (months)", 0, 72)

# -------------------------------
# SERVICES
# -------------------------------
with col2:
    st.markdown('<div class="section-title">📡 Services</div>', unsafe_allow_html=True)

    phone = st.radio("Uses Phone Service?", ["Yes", "No"])
    multi = st.radio("Multiple Phone Lines?", ["No", "Yes"])
    internet = st.selectbox("Internet Type", ["DSL", "Fiber", "No Internet"])

    security = st.checkbox("Online Security")
    backup = st.checkbox("Online Backup")

# -------------------------------
# FEATURES
# -------------------------------
with col3:
    st.markdown('<div class="section-title">⚙️ Features</div>', unsafe_allow_html=True)

    device = st.checkbox("Device Protection")
    support = st.checkbox("Tech Support")
    tv = st.checkbox("Streaming TV")
    movies = st.checkbox("Streaming Movies")

    contract = st.selectbox("Contract Type", ["Monthly", "1 Year", "2 Years"])

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# BILLING CARD 💰
# -------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown('<div class="section-title">💰 Billing (Simple)</div>', unsafe_allow_html=True)

monthly = st.slider(
    "💵 Monthly bill amount (₹)",
    0, 10000, 1000
)

# AUTO CALCULATE
total = monthly * tenure

st.success(f"📊 Total paid so far: ₹ {total}")

paper = st.radio("Paperless Billing?", ["Yes", "No"])
payment = st.selectbox("Payment Method", ["Online", "Card", "Bank", "Cash"])

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# PREPROCESS
# -------------------------------
def preprocess():
    return pd.DataFrame([{
        "gender": 1 if gender=="Male" else 0,
        "SeniorCitizen": 1 if senior=="Above 60" else 0,
        "Partner": 1 if partner=="Yes" else 0,
        "Dependents": 1 if dependents=="Yes" else 0,
        "tenure": tenure,
        "PhoneService": 1 if phone=="Yes" else 0,
        "MultipleLines": 1 if multi=="Yes" else 0,
        "InternetService": 0 if internet=="DSL" else (1 if internet=="Fiber" else 2),
        "OnlineSecurity": 1 if security else 0,
        "OnlineBackup": 1 if backup else 0,
        "DeviceProtection": 1 if device else 0,
        "TechSupport": 1 if support else 0,
        "StreamingTV": 1 if tv else 0,
        "StreamingMovies": 1 if movies else 0,
        "Contract": 0 if contract=="Monthly" else (1 if contract=="1 Year" else 2),
        "PaperlessBilling": 1 if paper=="Yes" else 0,
        "PaymentMethod": 0,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

# -------------------------------
# PREDICT BUTTON
# -------------------------------
if st.button("🚀 Analyze Customer"):

    df = preprocess()
    pred = model.predict(df)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(df)[0][1]
    else:
        prob = 0.5

    percentage = round(prob * 100, 2)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    if pred == 1:
        st.markdown(
            f'<div class="result-box" style="background:#7f1d1d;">⚠️ High Churn Risk ({percentage}%)</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-box" style="background:#14532d;">✅ Customer Will Stay ({100 - percentage}%)</div>',
            unsafe_allow_html=True
        )

    # PROGRESS BAR
    st.progress(prob)

    # RISK LEVEL
    if percentage < 30:
        st.success("🟢 Low Risk")
    elif percentage < 70:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")

    st.markdown('</div>', unsafe_allow_html=True)
