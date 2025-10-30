import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Bank Marketing AI", page_icon="🏦", layout="wide")

# Load everything
try:
    best_model = joblib.load('best_model_balanced.pkl')

    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    comparison_df = pd.read_csv('model_comparison.csv')
    
    # Find best model name
    best_model_name = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
    best_accuracy = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Accuracy']
    best_f1 = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'F1-Score']
    
    st.sidebar.success(f"✅ Loaded: {best_model_name}")
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.stop()

# SIDEBAR
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("", ["🔮 Prediction", "📊 Model Analysis"])

st.sidebar.markdown("---")
st.sidebar.markdown(f"### 🏆 Best Model")
st.sidebar.info(f"**{best_model_name}**")
st.sidebar.metric("Accuracy", f"{best_accuracy:.2%}")
st.sidebar.metric("F1-Score", f"{best_f1:.4f}")

def encode_cat(val, feat):
    if feat in label_encoders and feat != 'y':
        try:
            return label_encoders[feat].transform([val])[0]
        except:
            return 0
    return val

# ============================================================================
# PAGE 1: PREDICTION WITH BEST MODEL
# ============================================================================
if page == "🔮 Prediction":
    st.title(f"🏦 Bank Marketing Prediction")
    st.markdown(f"### Using **{best_model_name}** (Best Model)")
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    data = {}
    
    with c1:
        st.subheader("👤 Customer Information")
        data['age'] = st.slider("Age", 17, 98, 40)
        data['job'] = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
        data['marital'] = st.selectbox("Marital Status", ['divorced', 'married', 'single', 'unknown'])
        data['education'] = st.selectbox("Education", ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown'])
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
            # Encode
            enc = {k: encode_cat(v, k) for k, v in data.items()}
            df_inp = pd.DataFrame([enc])
            df_sc = scaler.transform(df_inp)
            
            # Predict with probability
            pred_raw = best_model.predict(df_sc)[0]
            prob = best_model.predict_proba(df_sc)[0]
            
            # THRESHOLD TUNING - Adjust to 0.25 for better YES predictions
            THRESHOLD = 0.20
            if prob[1] > THRESHOLD:
                pred = 1
            else:
                pred = 0
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Result box
            if pred == 1:
                st.success(f"### ✅ CUSTOMER WILL SUBSCRIBE")
                st.markdown(f"**Model Used:** {best_model_name}")
                st.markdown(f"**Subscription Probability:** {prob[1]:.1%}")
            else:
                st.error(f"### ❌ CUSTOMER WON'T SUBSCRIBE")
                st.markdown(f"**Model Used:** {best_model_name}")
                st.markdown(f"**Subscription Probability:** {prob[1]:.1%}")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Metrics
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Model Used", best_model_name)
            with c2:
                st.metric("Won't Subscribe", f"{prob[0]:.1%}")
            with c3:
                st.metric("Will Subscribe", f"{prob[1]:.1%}")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob[1] * 100,
                title={'text': "Subscription Probability (%)"},
                delta={'reference': 25},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightcoral"},
                        {'range': [25, 60], 'color': "lightyellow"},
                        {'range': [60, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 25
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Bar chart
            fig2 = px.bar(
                pd.DataFrame({'Outcome': ['Won\'t Subscribe', 'Will Subscribe'], 'Probability': [prob[0], prob[1]]}),
                x='Outcome', y='Probability',
                color='Outcome',
                color_discrete_map={'Won\'t Subscribe': '#dc3545', 'Will Subscribe': '#28a745'},
                height=400,
                text_auto='.1%'
            )
            fig2.update_layout(showlegend=False, xaxis_title="", yaxis_title="Probability")
            st.plotly_chart(fig2, use_container_width=True)
            
            st.info(f"**Note:** Prediction threshold is set to {THRESHOLD*100:.0f}%. This means YES is predicted when subscription probability exceeds {THRESHOLD*100:.0f}%.")
            
        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")
            import traceback
            st.code(traceback.format_exc())

# ============================================================================
# PAGE 2: MODEL COMPARISON & ANALYSIS
# ============================================================================
else:
    st.title("📊 Model Performance Analysis")
    st.markdown(f"### Comparison of All {len(comparison_df)} Models")
    st.markdown("---")
    
    # Best model highlight
    st.subheader("🏆 Best Performing Model")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Model", best_model_name, help="Best model by F1-Score")
    with col2:
        st.metric("Accuracy", f"{comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Accuracy']:.2%}")
    with col3:
        st.metric("Precision", f"{comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Precision']:.4f}")
    with col4:
        st.metric("Recall", f"{comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Recall']:.4f}")
    with col5:
        st.metric("F1-Score", f"{best_f1:.4f}")
    
    st.markdown("---")
    
    # Performance table
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
    
    # Charts
    st.subheader("📈 Performance Comparison Charts")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### Accuracy Comparison")
        fig = px.bar(
            comparison_df.sort_values('Accuracy', ascending=False),
            x='Model', y='Accuracy',
            color='Accuracy',
            color_continuous_scale='Blues',
            height=450,
            text_auto='.4f'
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Accuracy Score")
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.markdown("#### F1-Score Comparison")
        fig = px.bar(
            comparison_df.sort_values('F1-Score', ascending=False),
            x='Model', y='F1-Score',
            color='F1-Score',
            color_continuous_scale='Greens',
            height=450,
            text_auto='.4f'
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="F1-Score")
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### Precision Comparison")
        fig = px.bar(
            comparison_df.sort_values('Precision', ascending=False),
            x='Model', y='Precision',
            color='Precision',
            color_continuous_scale='Oranges',
            height=450,
            text_auto='.4f'
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Precision Score")
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.markdown("#### Recall Comparison")
        fig = px.bar(
            comparison_df.sort_values('Recall', ascending=False),
            x='Model', y='Recall',
            color='Recall',
            color_continuous_scale='Reds',
            height=450,
            text_auto='.4f'
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Recall Score")
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### ROC-AUC Comparison")
        fig = px.bar(
            comparison_df.sort_values('ROC-AUC', ascending=False),
            x='Model', y='ROC-AUC',
            color='ROC-AUC',
            color_continuous_scale='Purples',
            height=450,
            text_auto='.4f'
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="ROC-AUC Score")
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.markdown("#### Precision vs Recall")
        fig = px.scatter(
            comparison_df,
            x='Recall', y='Precision',
            size='F1-Score',
            color='Model',
            hover_name='Model',
            hover_data={'Accuracy': ':.4f', 'F1-Score': ':.4f', 'ROC-AUC': ':.4f'},
            height=450,
            size_max=50
        )
        fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig, use_container_width=True)
    
    # Model ranking
    st.markdown("---")
    st.subheader("🥇 Model Rankings (by F1-Score)")
    
    ranking = comparison_df[['Model', 'F1-Score']].sort_values('F1-Score', ascending=False).reset_index(drop=True)
    ranking.index = ranking.index + 1
    ranking.index.name = 'Rank'
    
    st.table(ranking.style.format({'F1-Score': '{:.4f}'}))

# Footer
st.markdown("---")
st.markdown(f"**🏦 Bank Marketing AI System** | Powered by **{best_model_name}** | 8 Models Trained | Threshold: 20%")

