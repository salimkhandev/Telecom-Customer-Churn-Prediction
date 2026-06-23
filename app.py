import streamlit as st
import plotly.graph_objects as go
from src.predict import predict_single_customer

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Removed background gradient to use clean native theme */

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.02);
    border-right: 1px solid rgba(0,0,0,0.1);
}

/* Section headers */
.section-header {
    background: rgba(0,0,0,0.03);
    border-left: 4px solid #000000;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin: 20px 0 12px 0;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.5px;
}

/* Result cards */
.result-card {
    background: transparent;
    border-radius: 16px;
    padding: 28px;
    border: 1px solid rgba(128,128,128,0.2);
    text-align: center;
}
.result-card-high  { border-top: 4px solid #ef4444; }
.result-card-medium{ border-top: 4px solid #f59e0b; }
.result-card-low   { border-top: 4px solid #10b981; }

/* Metric chips */
.metric-chip {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 4px;
}
.chip-high   { background: rgba(239,68,68,0.2);  color: #f87171; border: 1px solid #ef4444; }
.chip-medium { background: rgba(245,158,11,0.2); color: #fbbf24; border: 1px solid #f59e0b; }
.chip-low    { background: rgba(16,185,129,0.2); color: #34d399; border: 1px solid #10b981; }

/* Predict button */
div.stButton > button {
    width: 100%;
    background: #000000;
    color: #ffffff;
    border: 1px solid #000000;
    border-radius: 6px;
    padding: 14px;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    transition: all 0.2s ease;
}
div.stButton > button:hover {
    background: #333333;
    border-color: #333333;
}

/* Input fields */
/* Removed overrides to use native Streamlit styling */

/* Divider */
hr { border-color: rgba(128,128,128,0.2); }

/* Title */
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 4px;
}
.hero-sub {
    color: #666666;
    font-size: 0.95rem;
    margin-bottom: 28px;
}
</style>
""", unsafe_allow_html=True)


# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Churn Predictor")
    st.markdown("---")
    st.markdown("""
    <div style='color:#666666; font-size:0.85rem; line-height:1.7'>
    This tool uses an <b style='color:#a78bfa'>XGBoost</b> model trained on the IBM Telco Customer Churn dataset to predict whether a customer is likely to cancel their service.<br><br>
    The model was trained on <b style='color:#a78bfa'>7,043 customers</b> and achieves:<br>
    • Accuracy: <b style='color:#34d399'>77.1%</b><br>
    • F1 Score: <b style='color:#34d399'>0.777</b><br>
    • AUC-ROC: <b style='color:#34d399'>0.821</b>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='color:#888888; font-size:0.75rem'>
    FYP — Islamia College University Peshawar<br>
    Dept. of Computer Science
    </div>
    """, unsafe_allow_html=True)


# ─── MAIN HEADER ────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">📡 Telecom Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Fill in the customer profile below and click Predict to see their churn risk.</div>', unsafe_allow_html=True)

# ─── FORM ───────────────────────────────────────────────────────────────────
with st.form("prediction_form"):

    # Section 1: Demographics
    st.markdown('<div class="section-header">👤 Customer Demographics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with c2:
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with c3:
        partner = st.selectbox("Partner", ["Yes", "No"])
    with c4:
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    # Section 2: Account
    st.markdown('<div class="section-header">📋 Account Information</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    with c2:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
    with c3:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=780.0, step=1.0)

    # Section 3: Services
    st.markdown('<div class="section-header">🌐 Services Subscribed</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    with c2:
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    with c3:
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    with c4:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])

    c1, c2 = st.columns([1, 3])
    with c1:
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    # Section 4: Contract & Payment
    st.markdown('<div class="section-header">💳 Contract & Billing</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    with c2:
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    with c3:
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔮 Predict Churn Risk")


# ─── RESULT ─────────────────────────────────────────────────────────────────
if submitted:
    customer = {
        'gender': gender, 'SeniorCitizen': senior,
        'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone_service,
        'MultipleLines': multiple_lines, 'InternetService': internet_service,
        'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
        'DeviceProtection': device_protection, 'TechSupport': tech_support,
        'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
        'Contract': contract, 'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }

    with st.spinner("Analyzing customer profile..."):
        result = predict_single_customer(customer)

    prob = result['churn_probability']
    risk = result['risk_level']
    prediction = result['churn_prediction']

    # Color theme based on risk
    color_map = {'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'}
    card_class = {'High': 'result-card-high', 'Medium': 'result-card-medium', 'Low': 'result-card-low'}
    chip_class = {'High': 'chip-high', 'Medium': 'chip-medium', 'Low': 'chip-low'}
    color = color_map[risk]

    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    # Left: gauge chart
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            number={'suffix': '%', 'font': {'size': 44, 'color': color}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': 'gray'},
                'bar': {'color': color, 'thickness': 0.25},
                'bgcolor': 'rgba(0,0,0,0)',
                'bordercolor': 'gray',
                'steps': [
                    {'range': [0, 30],  'color': 'rgba(16,185,129,0.15)'},
                    {'range': [30, 60], 'color': 'rgba(245,158,11,0.15)'},
                    {'range': [60, 100],'color': 'rgba(239,68,68,0.15)'},
                ],
                'threshold': {
                    'line': {'color': color, 'width': 3},
                    'thickness': 0.75,
                    'value': prob * 100
                }
            },
            title={'text': "Churn Probability", 'font': {'size': 16}}
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300,
            margin=dict(t=60, b=20, l=30, r=30)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Right: verdict card
    with col2:
        verdict = "⚠️ Likely to Churn" if prediction == 1 else "✅ Likely to Stay"
        advice_map = {
            'High':   "Immediate retention action recommended. Consider a personalized discount or contract upgrade offer.",
            'Medium': "Monitor this customer closely. A proactive check-in call or loyalty reward may prevent churn.",
            'Low':    "This customer appears stable. Continue providing quality service to maintain loyalty."
        }
        st.markdown(f"""
        <div class="result-card {card_class[risk]}">
            <div style="font-size:2.8rem; margin-bottom:8px">{'🔴' if risk=='High' else '🟡' if risk=='Medium' else '🟢'}</div>
            <div style="font-size:1.5rem; font-weight:700; color:{color}; margin-bottom:8px">{risk} Risk</div>
            <div style="font-size:1.1rem; color:inherit; margin-bottom:16px">{verdict}</div>
            <span class="metric-chip {chip_class[risk]}">Churn Probability: {prob*100:.1f}%</span>
            <hr style="border-color:rgba(128,128,128,0.3); margin:16px 0">
            <div style="color:inherit; opacity:0.75; font-size:0.85rem; line-height:1.6">{advice_map[risk]}</div>
        </div>
        """, unsafe_allow_html=True)
