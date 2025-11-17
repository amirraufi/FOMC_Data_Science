"""
FOMC Market Reaction Predictor - Interactive Web App

A Streamlit web app for predicting Treasury market reactions to FOMC communications.

Features:
- Upload FOMC statement text
- Get instant prediction
- See which features drove the prediction (SHAP)
- Compare to historical statements
- Interactive visualizations

Usage:
    streamlit run app_streamlit.py

Requirements:
    pip install streamlit plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pickle

# Page config
st.set_page_config(
    page_title="FOMC Market Reaction Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 48px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 20px 0;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">📊 FOMC Market Reaction Predictor</p>', unsafe_allow_html=True)
st.markdown("### Predict Treasury yield reactions to Fed communications using AI")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    model_choice = st.selectbox(
        "Select Model",
        ["Random Forest", "Gradient Boosting", "XGBoost", "Neural Network"]
    )

    target_yield = st.selectbox(
        "Target Yield",
        ["2-Year Treasury", "5-Year Treasury", "10-Year Treasury"]
    )

    horizon = st.selectbox(
        "Prediction Horizon",
        ["1 day", "2 days", "5 days", "10 days"]
    )

    st.markdown("---")
    st.header("📚 About")
    st.markdown("""
    This app uses machine learning to predict how Treasury yields will
    react to FOMC communications.

    **Key Features:**
    - Multi-modal NLP (GPT-4, FinBERT, BART)
    - Novel change detection
    - SHAP interpretability
    - Real-time predictions
    """)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📊 Historical Data", "📈 Model Performance", "ℹ️ About"])

# ============================================================================
# TAB 1: PREDICTION
# ============================================================================
with tab1:
    st.header("Enter FOMC Statement")

    col1, col2 = st.columns([2, 1])

    with col1:
        statement_text = st.text_area(
            "Paste FOMC statement text here:",
            height=300,
            placeholder="The Federal Open Market Committee decided today to..."
        )

        predict_button = st.button("🔮 Predict Market Reaction", type="primary", use_container_width=True)

    with col2:
        st.markdown("### Quick Examples")

        if st.button("📝 Hawkish Example"):
            statement_text = """The Committee decided to raise the target range for the federal funds rate to 5.25 to 5.5 percent. The Committee will continue to assess additional information and its implications for monetary policy. In determining the extent of additional policy firming that may be appropriate, the Committee will take into account the cumulative tightening of monetary policy, the lags with which monetary policy affects economic activity and inflation, and economic and financial developments."""
            st.experimental_rerun()

        if st.button("🕊️ Dovish Example"):
            statement_text = """The Committee decided to lower the target range for the federal funds rate to 4.25 to 4.50 percent. Inflation has eased over the past year but remains somewhat elevated. In support of its goals, the Committee decided to lower the target range for the federal funds rate by 1/4 percentage point."""
            st.experimental_rerun()

    if predict_button and statement_text:
        with st.spinner("Analyzing statement and predicting market reaction..."):
            # This is a DEMO - replace with actual prediction logic
            # For now, show mock prediction

            st.markdown("---")
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)

            # Prediction result
            col_p1, col_p2, col_p3 = st.columns(3)

            predicted_change = np.random.normal(5, 3)  # Mock prediction
            confidence = np.random.uniform(0.6, 0.9)

            with col_p1:
                st.metric(
                    label=f"{target_yield} Predicted Change",
                    value=f"{predicted_change:+.2f} bp",
                    delta="Hawkish" if predicted_change > 0 else "Dovish"
                )

            with col_p2:
                st.metric(
                    label="Model Confidence",
                    value=f"{confidence:.1%}"
                )

            with col_p3:
                st.metric(
                    label="Prediction Horizon",
                    value=horizon
                )

            st.markdown('</div>', unsafe_allow_html=True)

            # Feature importance
            st.markdown("### 🔍 What Drove This Prediction?")

            features_data = {
                'Feature': ['GPT-4 Hawkishness', 'Inflation Language Change', 'Rate Increase Mentioned',
                           'BART Sentiment', 'Text Length Change', 'Labor Market Language'],
                'Importance': [0.35, 0.28, 0.15, 0.12, 0.06, 0.04],
                'Direction': ['+', '+', '+', '-', '+', '-']
            }
            features_df = pd.DataFrame(features_data)

            fig = px.bar(
                features_df,
                x='Importance',
                y='Feature',
                orientation='h',
                color='Direction',
                color_discrete_map={'+': 'red', '-': 'blue'},
                title="Top Features Contributing to Prediction"
            )
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

            # Comparison to historical
            st.markdown("### 📊 Historical Context")

            col_h1, col_h2 = st.columns(2)

            with col_h1:
                # Distribution plot
                hist_data = np.random.normal(0, 7, 200)  # Mock historical data
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=hist_data,
                    name='Historical Reactions',
                    nbinsx=30,
                    marker_color='lightblue'
                ))
                fig_hist.add_vline(
                    x=predicted_change,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Your Prediction"
                )
                fig_hist.update_layout(
                    title="Prediction vs Historical Distribution",
                    xaxis_title="Yield Change (bp)",
                    yaxis_title="Frequency",
                    height=350
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_h2:
                # Similar statements
                st.markdown("**Most Similar Historical Statements:**")

                similar_statements = [
                    {"Date": "2023-07-26", "Similarity": 0.89, "Actual Change": "+4.2 bp"},
                    {"Date": "2023-05-03", "Similarity": 0.85, "Actual Change": "+6.1 bp"},
                    {"Date": "2023-03-22", "Similarity": 0.82, "Actual Change": "+3.8 bp"},
                ]

                for stmt in similar_statements:
                    st.markdown(f"""
                    **{stmt['Date']}** (Similarity: {stmt['Similarity']:.0%})
                    - Actual market reaction: {stmt['Actual Change']}
                    """)

# ============================================================================
# TAB 2: HISTORICAL DATA
# ============================================================================
with tab2:
    st.header("Historical FOMC Reactions")

    # Load historical data (mock for demo)
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='60D')
    historical_data = pd.DataFrame({
        'Date': dates,
        'dy2_1d_bp': np.random.normal(0, 7, len(dates)),
        'gpt_hawk_score': np.random.randint(-2, 3, len(dates))
    })

    # Time series plot
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=historical_data['Date'],
        y=historical_data['dy2_1d_bp'],
        mode='markers+lines',
        name='2Y Treasury Reaction',
        marker=dict(
            size=8,
            color=historical_data['gpt_hawk_score'],
            colorscale='RdBu',
            showscale=True,
            colorbar=dict(title="Hawkishness")
        )
    ))
    fig_ts.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_ts.update_layout(
        title="Market Reactions Over Time",
        xaxis_title="Date",
        yaxis_title="2Y Yield Change (bp)",
        height=500
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # Statistics
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    with col_s1:
        st.metric("Average Reaction", f"{historical_data['dy2_1d_bp'].mean():.2f} bp")

    with col_s2:
        st.metric("Std Deviation", f"{historical_data['dy2_1d_bp'].std():.2f} bp")

    with col_s3:
        positive_pct = (historical_data['dy2_1d_bp'] > 0).mean()
        st.metric("% Positive Reactions", f"{positive_pct:.1%}")

    with col_s4:
        max_reaction = historical_data['dy2_1d_bp'].abs().max()
        st.metric("Max Absolute Move", f"{max_reaction:.2f} bp")

# ============================================================================
# TAB 3: MODEL PERFORMANCE
# ============================================================================
with tab3:
    st.header("Model Performance Metrics")

    # Mock model results
    model_results = pd.DataFrame({
        'Model': ['Ridge', 'Lasso', 'Random Forest', 'Gradient Boosting'],
        'RMSE': [9.14, 8.01, 7.46, 8.13],
        'MAE': [6.33, 5.74, 5.43, 6.13],
        'R²': [-1.02, -0.48, -0.28, -0.53],
        'Dir Accuracy': [0.52, 0.55, 0.58, 0.54]
    })

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        # RMSE comparison
        fig_rmse = px.bar(
            model_results,
            x='Model',
            y='RMSE',
            title="Model RMSE Comparison",
            color='RMSE',
            color_continuous_scale='Reds_r'
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

    with col_m2:
        # Directional Accuracy
        fig_acc = px.bar(
            model_results,
            x='Model',
            y='Dir Accuracy',
            title="Directional Accuracy",
            color='Dir Accuracy',
            color_continuous_scale='Greens'
        )
        fig_acc.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Random (50%)")
        st.plotly_chart(fig_acc, use_container_width=True)

    # Feature importance
    st.markdown("### Top 10 Most Important Features")

    feature_imp = pd.DataFrame({
        'Feature': ['BART Hawk Prob', 'Hawk Count', 'Change: Inflation Easing Added',
                   'GPT-4 Score', 'FinBERT Positive', 'Semantic Change',
                   'Text Length Change', 'Overall Similarity', 'Sentence Count Change', 'BART Score'],
        'Importance': [0.62, 0.44, 0.30, 0.29, 0.23, 0.23, 0.17, 0.13, 0.12, 0.11]
    })

    fig_feat = px.bar(
        feature_imp,
        x='Importance',
        y='Feature',
        orientation='h',
        title="SHAP Feature Importance"
    )
    fig_feat.update_layout(height=500)
    st.plotly_chart(fig_feat, use_container_width=True)

# ============================================================================
# TAB 4: ABOUT
# ============================================================================
with tab4:
    st.header("About This Project")

    st.markdown("""
    ## 🎯 Project Overview

    This application uses advanced machine learning and natural language processing
    to predict how Treasury markets will react to Federal Reserve (FOMC) communications.

    ### 🔬 Methodology

    **1. Multi-Modal NLP Analysis:**
    - **GPT-4**: Hawkishness scoring with detailed reasoning
    - **FinBERT**: Financial sentiment analysis
    - **BART**: Zero-shot classification
    - **Semantic Embeddings**: Capture meaning and context

    **2. Novel Change Detection** 🌟
    - Compares consecutive statements to detect linguistic shifts
    - Tracks 30+ features including:
      - Sentences added/removed
      - Key phrase changes (inflation, rates, labor)
      - Semantic similarity
      - Text structure changes

    **3. Time-Series Modeling:**
    - Walk-forward cross-validation (no lookahead bias)
    - Multiple model families (Linear, Tree-based, Neural Nets)
    - SHAP analysis for interpretability

    ### 📊 Performance

    - **Best Model**: Random Forest
    - **RMSE**: 7.46 basis points
    - **Directional Accuracy**: 58% (vs 50% random)
    - **Top Feature**: BART Hawk Probability (0.62 importance)

    ### 📚 Research Paper

    This work is being developed for submission to top finance journals:
    - Target: Journal of Finance, JFE, RFS
    - Novel contribution: Change detection approach
    - Timeline: Conference submission → Journal submission → Publication (2-3 years)

    ### 👥 Team

    [Your Name]
    [Your Institution]
    [Contact Information]

    ### 📖 Citation

    ```
    @article{fomc_nlp_2025,
      title={Predicting Market Reactions to FOMC Communications: A Change Detection Approach},
      author={[Your Names]},
      year={2025},
      note={Working paper}
    }
    ```

    ### 🔗 Resources

    - [GitHub Repository](#)
    - [Research Paper](#)
    - [Data Sources](DATA_SOURCES.md)

    ---

    **Built with**: Python, Streamlit, scikit-learn, SHAP, Plotly
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>FOMC Market Reaction Predictor v1.0 | Built with Streamlit</p>
    <p>⚠️ For research purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
