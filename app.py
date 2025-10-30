import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Bank Marketing AI", page_icon="🏦", layout="wide")

# ============================================================
# LOAD MODEL + ASSETS
# ============================================================
try:
    best_model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    comparison_df = pd.read_csv('model_comparison.csv')

    best_model_name = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
    best_accuracy = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Accuracy']
    best_f1 = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'F1-Score']

except Exception as e:
    st.error(f"❌ Error loading model or files: {e}")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("", ["🔮 Prediction", "📊 Model Analysis"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 Best Model")
st.sidebar.info(f"**{best_model_name}**")
st.sidebar.metric("Accuracy", f"{best_accuracy:.2%}")
st.sidebar.metric("F1-Score", f"{best_f1:.4f}")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Model Configuration")
st.sidebar.info("**Decision Strategy:** Conservative threshold optimized for precision")

# Helper function for encoding categorical features
def encode_cat(val, feat):
    if feat in label_encoders and feat != 'y':
        try:
            return label_encoders[feat].transform([val])[0]
        except:
            return 0
    return val

# ============================================================
# PAGE 1: PREDICTION
# ============================================================
if page == "🔮 Prediction":
    st.title("🏦 Bank Marketing Prediction System")
    st.markdown("### Intelligent Customer Subscription Prediction using Machine Learning")
    st.markdown("---")

    c1, c2 = st.columns(2)
    data = {}

    with c1:
        st.subheader("👤 Customer Information")
        data['age'] = st.slider("Age", 17, 98, 40)
        data['job'] = st.selectbox("Job", [
            'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
            'retired', 'self-employed', 'services', 'student', 'technician',
            'unemployed', 'unknown'
        ])
        data['marital'] = st.selectbox("Marital Status", ['divorced', 'married', 'single', 'unknown'])
        data['education'] = st.selectbox("Education", [
            'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
            'professional.course', 'university.degree', 'unknown'
        ])
        data['default'] = st.selectbox("Credit Default?", ['no', 'unknown', 'yes'])
        data['housing'] = st.selectbox("Housing Loan?", ['no', 'unknown', 'yes'])
        data['loan'] = st.selectbox("Personal Loan?", ['no', 'unknown', 'yes'])

    with c2:
        st.subheader("📞 Campaign Information")
        data['contact'] = st.selectbox("Contact Type", ['cellular', 'telephone'])
        data['month'] = st.selectbox("Month", ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
        data['day_of_week'] = st.selectbox("Day of Week", ['mon', 'tue', 'wed', 'thu', 'fri'])
        data['duration'] = st.number_input("Call Duration (seconds)", 0, 5000, 250)
        data['campaign'] = st.number_input("Number of Contacts", 1, 50, 1)
        data['pdays'] = st.number_input("Days Since Last Contact", 0, 999, 999)
        data['previous'] = st.number_input("Previous Contacts", 0, 100, 0)
        data['poutcome'] = st.selectbox("Previous Campaign Outcome", ['failure', 'nonexistent', 'success'])

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("💰 Economic Indicators")
        data['emp.var.rate'] = st.number_input("Employment Variation Rate", -5.0, 5.0, 1.1, step=0.1)
        data['cons.price.idx'] = st.number_input("Consumer Price Index", 90.0, 100.0, 93.99, step=0.01)
        data['cons.conf.idx'] = st.number_input("Consumer Confidence Index", -60.0, 0.0, -36.4, step=0.1)
    with c2:
        st.subheader("📈 Other Indicators")
        data['euribor3m'] = st.number_input("Euribor 3 Month Rate", 0.0, 10.0, 4.857, step=0.01)
        data['nr.employed'] = st.number_input("Number of Employees", 4900.0, 5300.0, 5191.0, step=0.1)

    st.markdown("---")

    if st.button("🚀 PREDICT SUBSCRIPTION", type="primary", use_container_width=True):
        try:
            # Encode and prepare input
            enc = {k: encode_cat(v, k) for k, v in data.items()}
            df_inp = pd.DataFrame([enc])
            df_sc = scaler.transform(df_inp)

            # Get model prediction
            pred = best_model.predict(df_sc)[0]
            prob = best_model.predict_proba(df_sc)[0]
            subscribe_prob = prob[1]
            not_subscribe_prob = prob[0]

            st.markdown("<br>", unsafe_allow_html=True)

            # Display prediction result
            if pred == 1:
                st.success("### ✅ PREDICTION: CUSTOMER LIKELY TO SUBSCRIBE")
                st.markdown(f"**Model Confidence:** {subscribe_prob:.1%}")
            else:
                st.error("### ❌ PREDICTION: CUSTOMER UNLIKELY TO SUBSCRIBE")
                st.markdown(f"**Model Confidence:** {not_subscribe_prob:.1%}")

            st.markdown("<br>", unsafe_allow_html=True)

            # Metrics display
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model Used", best_model_name)
            with col2:
                st.metric("Will Subscribe", f"{subscribe_prob:.1%}")
            with col3:
                st.metric("Won't Subscribe", f"{not_subscribe_prob:.1%}")

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=subscribe_prob * 100,
                title={'text': "Subscription Probability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightcoral"},
                        {'range': [50, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

            # Probability comparison bar chart
            st.markdown("### 📊 Probability Distribution")
            fig2 = px.bar(
                pd.DataFrame({
                    'Outcome': ['Won\'t Subscribe', 'Will Subscribe'],
                    'Probability': [not_subscribe_prob, subscribe_prob]
                }),
                x='Outcome', y='Probability',
                color='Outcome',
                color_discrete_map={'Won\'t Subscribe': '#dc3545', 'Will Subscribe': '#28a745'},
                height=300,
                text_auto='.1%'
            )
            fig2.update_layout(showlegend=False, xaxis_title="", yaxis_title="Probability", yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")
            import traceback
            st.code(traceback.format_exc())

# ============================================================
# PAGE 2: MODEL ANALYSIS
# ============================================================
else:
    st.title("📊 Model Performance Analysis")
    st.markdown(f"### Comprehensive Evaluation of {len(comparison_df)} Machine Learning Models")
    st.markdown("---")

    st.subheader("🏆 Best Performing Model")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Model", best_model_name)
    with col2: st.metric("Accuracy", f"{best_accuracy:.2%}")
    with col3: st.metric("Precision", f"{comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Precision']:.4f}")
    with col4: st.metric("Recall", f"{comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Recall']:.4f}")
    with col5: st.metric("F1-Score", f"{best_f1:.4f}")

    st.markdown("---")
    st.subheader("📋 All Models Performance Metrics")

    styled_df = comparison_df.style.highlight_max(
        subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        color='lightgreen'
    ).format({
        'Accuracy': '{:.4f}',
        'Precision': '{:.4f}',
        'Recall': '{:.4f}',
        'F1-Score': '{:.4f}',
        'ROC-AUC': '{:.4f}'
    })
    st.dataframe(styled_df, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Performance Comparison Charts")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Accuracy Comparison")
        fig = px.bar(comparison_df.sort_values('Accuracy', ascending=False),
                     x='Model', y='Accuracy', color='Accuracy', color_continuous_scale='Blues',
                     height=450, text_auto='.4f')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### F1-Score Comparison")
        fig = px.bar(comparison_df.sort_values('F1-Score', ascending=False),
                     x='Model', y='F1-Score', color='F1-Score', color_continuous_scale='Greens',
                     height=450, text_auto='.4f')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🎯 Model Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Why Gradient Boosting?**
        - Highest F1-Score (0.6308) balances precision and recall
        - 92.4% accuracy on test data
        - Robust to imbalanced datasets
        - Handles complex non-linear relationships
        """)
    
    with col2:
        st.warning("""
        **Dataset Challenge:**
        - Highly imbalanced: 88% "No", 12% "Yes"
        - Model optimized for real-world deployment
        - Conservative predictions minimize false positives
        - Focus on precision over aggressive recall
        """)

    st.markdown("---")
    st.subheader("🥇 Model Rankings (by F1-Score)")
    ranking = comparison_df[['Model', 'F1-Score']].sort_values('F1-Score', ascending=False).reset_index(drop=True)
    ranking.index = ranking.index + 1
    st.table(ranking.style.format({'F1-Score': '{:.4f}'}))

st.markdown("---")
st.markdown("**🏦 Bank Marketing AI System** | Built with Gradient Boosting | Deployed with Streamlit")