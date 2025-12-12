"""
FOMC Market Reaction Diagnostic Tool - Interactive Web App

A Streamlit web app for DIAGNOSTIC analysis of FOMC communications.

Features:
- Percentile scoring (how hawkish vs. history)
- Change highlighting (what changed linguistically)
- Nearest neighbor retrieval (similar past episodes)
- Probabilistic forecasts (not point predictions)

Usage:
    streamlit run app_streamlit_diagnostic.py

Requirements:
    pip install streamlit plotly scikit-learn
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pickle
import os

# Import our diagnostic tools
from fomc_analysis_utils import (
    DiagnosticAnalyzer,
    ProbabilisticPredictor,
    SubtleLinguisticAnalyzer,
    ChangeDetector,
    StatementMapPCA
)

# Page config
st.set_page_config(
    page_title="FOMC Diagnostic Tool",
    page_icon="🔬",
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
    .diagnostic-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 20px 0;
    }
    .percentile-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .change-highlight {
        background-color: #fff3cd;
        padding: 10px;
        border-left: 4px solid #ffc107;
        margin: 5px 0;
        color: #b30000;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA AND MODEL
# ============================================================================

@st.cache_data
def load_data():
    """Load enhanced dataset with all features"""
    try:
        df = pd.read_csv('data_enhanced_with_changes.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except:
        st.error("⚠️ Could not load data_enhanced_with_changes.csv. Please run run_analysis.py first.")
        return None

@st.cache_resource
def load_model():
    """Load trained Random Forest model (or train if needed)"""
    try:
        # Try to load saved model
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except:
        # If not available, train a quick model
        st.warning("Training model (first run only)...")
        from sklearn.ensemble import RandomForestRegressor

        df = load_data()
        if df is None:
            return None

        # Select features
        feature_cols = [col for col in df.columns if (
            col.startswith('change_') or
            col.startswith('subtle_') or
            col.startswith('gpt_') or
            col.startswith('bart_') or
            col.startswith('finbert_') or
            col in ['hawk_cnt', 'dove_cnt', 'hawk_minus_dove', 'cos_prev', 'delta_semantic']
        )]
        feature_cols = [col for col in feature_cols if col != 'gpt_reason' and col != 'bart_label']
        feature_cols = [col for col in feature_cols if col in df.columns]

        # Prepare data
        X = df[feature_cols].fillna(0)
        y = df['dy2_1d_bp']
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]

        # Train model
        model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)
        model.fit(X, y)

        # Save for next time
        with open('best_model.pkl', 'wb') as f:
            pickle.dump(model, f)

        return model

# Load data
df = load_data()
model = load_model()

if df is None:
    st.stop()

# Get feature columns
feature_cols = [col for col in df.columns if (
    col.startswith('change_') or
    col.startswith('subtle_') or
    col.startswith('gpt_') or
    col.startswith('bart_') or
    col.startswith('finbert_') or
    col in ['hawk_cnt', 'dove_cnt', 'hawk_minus_dove', 'cos_prev', 'delta_semantic']
)]
feature_cols = [col for col in feature_cols if col != 'gpt_reason' and col != 'bart_label']
feature_cols = [col for col in feature_cols if col in df.columns]

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<p class="main-header">🔬 FOMC Diagnostic Tool</p>', unsafe_allow_html=True)
st.markdown("### Sophisticated analysis of Fed communications (not point predictions)")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    target_yield = st.selectbox(
        "Target Yield",
        ["2-Year Treasury", "5-Year Treasury", "10-Year Treasury"]
    )
    target_map = {
        "2-Year Treasury": "dy2_1d_bp",
        "5-Year Treasury": "dy5_1d_bp",
        "10-Year Treasury": "dy10_1d_bp"
    }
    target_col = target_map[target_yield]

    k_neighbors = st.slider(
        "Number of Similar Statements",
        min_value=5,
        max_value=50,
        value=20,
        help="How many similar historical statements to use for probabilistic forecast"
    )

    # Highlight font color for linguistic changes
    highlight_color_choice = st.selectbox(
        "Linguistic change font color",
        ["Black", "Red"],
        index=0,
        help="Choose the font color used in the change highlighting boxes"
    )
    highlight_color = "black" if highlight_color_choice == "Black" else "red"

    st.markdown("---")
    st.header("📚 About")
    st.markdown("""
    This tool provides **diagnostic analysis** rather than point predictions:

    1. **Percentile Scoring**: How hawkish vs. history?
    2. **Change Highlighting**: What changed linguistically?
    3. **Nearest Neighbors**: Similar past episodes
    4. **Probabilistic Forecast**: Conditional distributions

    ⚠️ **Not**: "Yields will rise 4.3 bp"
    ✅ **Instead**: "Based on 20 similar statements, 80% of outcomes fell between -1 and +8 bp"
    """)

# ============================================================================
# MAIN TABS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs(["🔬 Diagnostic Analysis", "🗺️ Statement Map", "📊 Historical Data", "📈 Model Performance"])

# ============================================================================
# TAB 1: DIAGNOSTIC ANALYSIS
# ============================================================================
with tab1:
    st.header("Analyze FOMC Statement")

    # Option 1: Select from historical statements
    st.subheader("Option 1: Select Historical Statement")

    # Get statements with valid market data
    valid_statements = df[df[target_col].notna()].copy()
    valid_statements['display'] = valid_statements['Date'].dt.strftime('%Y-%m-%d') + ' - ' + valid_statements['Type']

    selected_statement = st.selectbox(
        "Choose a statement to analyze:",
        options=range(len(valid_statements)),
        format_func=lambda i: valid_statements.iloc[i]['display']
    )

    # Get selected statement
    current_idx = valid_statements.index[selected_statement]
    current_row = df.loc[current_idx]
    current_date = current_row['Date']

    # Option 2: Enter custom text (for future implementation)
    with st.expander("Option 2: Enter Custom Statement (Coming Soon)"):
        st.text_area(
            "Paste FOMC statement text here:",
            height=200,
            placeholder="The Federal Open Market Committee decided today to...",
            disabled=True,
            help="Feature coming soon: analyze custom text"
        )

    analyze_button = st.button("🔬 Run Diagnostic Analysis", type="primary", use_container_width=True)

    if analyze_button:
        with st.spinner("Running diagnostic analysis..."):

            st.markdown("---")

            # ================================================================
            # 1. PERCENTILE SCORING
            # ================================================================
            st.markdown("## 1️⃣ Hawkishness Percentile Score")
            st.markdown("*How does this statement compare to historical Fed language?*")

            # Create composite hawkishness
            composite_scores = DiagnosticAnalyzer.create_composite_hawkishness(df)
            if isinstance(composite_scores, np.ndarray):
                composite_scores = pd.Series(composite_scores, index=df.index)
            df['composite_hawk'] = composite_scores

            current_score = df.loc[current_idx, 'composite_hawk']
            historical_scores = df.loc[df.index != current_idx, 'composite_hawk']
            if isinstance(historical_scores, pd.Series):
                historical_scores = historical_scores.dropna()
            else:
                historical_scores = pd.Series(historical_scores).dropna()

            percentile = DiagnosticAnalyzer.compute_hawkishness_percentile(
                current_score, historical_scores
            )

            # Display percentile
            col_p1, col_p2, col_p3 = st.columns([1, 2, 1])

            with col_p2:
                st.markdown(f"""
                <div class="percentile-card">
                    <h1 style="font-size: 72px; margin: 10px 0;">{percentile:.0f}th</h1>
                    <h3 style="margin: 5px 0;">Percentile</h3>
                    <p style="font-size: 18px; margin-top: 15px;">
                    {"🔴 VERY HAWKISH" if percentile > 80 else
                     "🟠 Moderately Hawkish" if percentile > 60 else
                     "🟡 Neutral" if percentile > 40 else
                     "🔵 Moderately Dovish" if percentile > 20 else
                     "🟢 VERY DOVISH"}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            **Interpretation**: This statement is more hawkish than **{percentile:.0f}%** of all FOMC statements since 2000.
            """)

            # Show composite score components
            with st.expander("📊 See Score Components"):
                score_components = pd.DataFrame({
                    'Model': ['GPT-4 Score', 'BART Hawk Prob', 'FinBERT Score'],
                    'Value': [
                        df.loc[current_idx, 'gpt_hawk_score'] if 'gpt_hawk_score' in df.columns else np.nan,
                        df.loc[current_idx, 'bart_hawk_prob'] if 'bart_hawk_prob' in df.columns else np.nan,
                        df.loc[current_idx, 'finbert_pos'] - df.loc[current_idx, 'finbert_neg'] if 'finbert_pos' in df.columns else np.nan
                    ]
                })
                st.dataframe(score_components, use_container_width=True)

            # ================================================================
            # 2. CHANGE HIGHLIGHTING
            # ================================================================
            st.markdown("## 2️⃣ Linguistic Changes Detected")
            st.markdown("*What changed from the previous statement?*")

            if current_idx > 0:
                # Get change features for this statement
                change_features = {col: df.loc[current_idx, col]
                                 for col in df.columns
                                 if (col.startswith('change_') or col.startswith('subtle_'))
                                 and pd.notna(df.loc[current_idx, col])}

                # Highlight key changes
                highlights = DiagnosticAnalyzer.highlight_key_changes(change_features)

                if highlights:
                    for highlight in highlights:
                        st.markdown(f"""
                        <div class="change-highlight" style="color: {highlight_color};">
                            💡 {highlight}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No significant linguistic changes detected from previous statement.")

                # Show sentence-level changes
                with st.expander("📝 See Sentence-Level Changes"):
                    col_c1, col_c2, col_c3 = st.columns(3)
                    with col_c1:
                        st.metric("Sentences Added",
                                int(df.loc[current_idx, 'change_sentences_added']) if 'change_sentences_added' in df.columns else 0)
                    with col_c2:
                        st.metric("Sentences Removed",
                                int(df.loc[current_idx, 'change_sentences_removed']) if 'change_sentences_removed' in df.columns else 0)
                    with col_c3:
                        st.metric("Overall Similarity",
                                f"{df.loc[current_idx, 'change_overall_similarity']:.1%}" if 'change_overall_similarity' in df.columns else "N/A")
            else:
                st.info("This is the first statement in the dataset - no previous statement to compare to.")

            # ================================================================
            # 3. NEAREST NEIGHBORS
            # ================================================================
            st.markdown("## 3️⃣ Most Similar Historical Statements")
            st.markdown("*What happened in similar past episodes?*")

            # Get current features
            current_features = df.loc[current_idx, feature_cols].fillna(0)

            # Get historical data (exclude very recent)
            historical_df = df.loc[df.index != current_idx].copy()
            historical_df = historical_df[historical_df[target_col].notna()]

            if len(historical_df) > 0:
                historical_features = historical_df[feature_cols].fillna(0)

                # Find nearest neighbors using standardized Euclidean distance (new method)
                neighbor_indices = DiagnosticAnalyzer.find_nearest_neighbors(
                    current_features, historical_features, k=min(5, len(historical_df)),
                    metric='euclidean_standardized'  # Use standardized distance for better results
                )

                st.markdown("**Top 5 Most Similar Statements:**")
                st.caption("_Similarity uses standardized Euclidean distance (more discriminative than cosine)_")

                for rank, idx in enumerate(neighbor_indices, 1):
                    neighbor_row = historical_df.iloc[idx]
                    neighbor_date = neighbor_row['Date']
                    neighbor_reaction = neighbor_row[target_col]

                    # Compute similarity using new standardized method
                    similarity = DiagnosticAnalyzer.compute_similarity_score(
                        current_features, historical_features, idx
                    )

                    col_n1, col_n2, col_n3 = st.columns([2, 1, 1])
                    with col_n1:
                        st.markdown(f"**{rank}. {neighbor_date:%Y-%m-%d}** - {neighbor_row['Type']}")
                    with col_n2:
                        st.markdown(f"Similarity: **{similarity:.1f}%**")
                    with col_n3:
                        st.markdown(f"Actual: **{neighbor_reaction:+.1f} bp**")

            # ================================================================
            # 4. PROBABILISTIC FORECAST
            # ================================================================
            st.markdown("## 4️⃣ Probabilistic Market Reaction Forecast")
            st.markdown("*Not a point prediction - a conditional distribution*")

            # Compute conditional distribution
            dist = ProbabilisticPredictor.conditional_distribution(
                current_features=current_features,
                historical_df=historical_df,
                feature_cols=feature_cols,
                target=target_col,
                k=k_neighbors
            )

            if dist is not None:
                # Display formatted forecast
                forecast_text = ProbabilisticPredictor.format_probabilistic_forecast(dist, target_yield)
                st.code(forecast_text, language=None)

                # Visualize distribution
                col_v1, col_v2 = st.columns(2)

                with col_v1:
                    # Get outcomes of similar statements
                    neighbor_outcomes = historical_df.iloc[neighbor_indices[:k_neighbors]][target_col].dropna()

                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=neighbor_outcomes,
                        nbinsx=15,
                        name='Similar Statements',
                        marker_color='lightblue'
                    ))
                    fig.add_vline(x=dist['median'], line_dash="dash", line_color="red",
                                annotation_text=f"Median: {dist['median']:+.1f} bp")
                    fig.add_vline(x=dist['q10'], line_dash="dot", line_color="orange",
                                annotation_text=f"10th: {dist['q10']:+.1f}")
                    fig.add_vline(x=dist['q90'], line_dash="dot", line_color="orange",
                                annotation_text=f"90th: {dist['q90']:+.1f}")
                    fig.update_layout(
                        title="Conditional Distribution",
                        xaxis_title="Yield Change (bp)",
                        yaxis_title="Frequency",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col_v2:
                    # Probability breakdown
                    st.markdown("### Probability Breakdown")

                    prob_data = pd.DataFrame({
                        'Outcome': [
                            'Yields Rise (>0 bp)',
                            'Yields Fall (<0 bp)',
                            'Large Rise (>10 bp)',
                            'Large Fall (<-10 bp)'
                        ],
                        'Probability': [
                            dist['prob_positive'],
                            1 - dist['prob_positive'],
                            dist['tail_up_10bp'],
                            dist['tail_down_10bp']
                        ]
                    })

                    fig_prob = px.bar(
                        prob_data,
                        x='Probability',
                        y='Outcome',
                        orientation='h',
                        text=prob_data['Probability'].apply(lambda x: f"{x:.0%}")
                    )
                    fig_prob.update_traces(textposition='outside')
                    fig_prob.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_prob, use_container_width=True)

                # Show actual outcome (if this is historical)
                st.markdown("### ✅ Actual Outcome (Historical Validation)")
                actual_reaction = df.loc[current_idx, target_col]
                if pd.notna(actual_reaction):
                    col_a1, col_a2, col_a3 = st.columns(3)

                    with col_a1:
                        st.metric("Actual Market Reaction", f"{actual_reaction:+.1f} bp")

                    with col_a2:
                        # Check if within 80% interval
                        within_80 = dist['q10'] <= actual_reaction <= dist['q90']
                        st.metric("Within 80% Interval?", "✅ Yes" if within_80 else "❌ No")

                    with col_a3:
                        # Percentile of actual outcome
                        actual_percentile = (neighbor_outcomes < actual_reaction).mean() * 100
                        st.metric("Actual Percentile", f"{actual_percentile:.0f}th")

                    if within_80:
                        st.success("✅ The actual market reaction fell within our 80% confidence interval!")
                    else:
                        st.warning("⚠️ The actual market reaction was outside our 80% interval - a tail event!")
            else:
                st.warning("⚠️ Insufficient historical data for probabilistic forecast.")

# ============================================================================
# TAB 2: STATEMENT MAP (PCA Visualization)
# ============================================================================
with tab2:
    st.header("FOMC Statement Map")
    st.markdown("*Explore 25 years of Fed communications in 2D/3D space*")

    # Cache PCA computation
    @st.cache_data
    def compute_pca_map(_df, feature_cols):
        """Compute PCA projection of statements"""
        pca_mapper = StatementMapPCA(n_components=3)
        coords = pca_mapper.fit_transform(_df, feature_cols)
        variance = pca_mapper.get_variance_explained()
        loadings = pca_mapper.get_top_loadings(n_top=5)
        return coords, variance, loadings

    # Compute PCA
    pca_coords, variance_info, loadings_info = compute_pca_map(df, feature_cols)

    # Merge with original data for complete info
    map_df = pca_coords.copy()

    # View options
    col_opt1, col_opt2, col_opt3 = st.columns(3)

    with col_opt1:
        view_mode = st.radio("View Mode", ["2D", "3D"], horizontal=True)

    with col_opt2:
        color_by = st.selectbox(
            "Color By",
            ["Regime", "Year", "Hawkishness", "Market Reaction"],
            index=0
        )

    with col_opt3:
        show_trajectory = st.checkbox("Show Time Trajectory", value=False)

    # Prepare color data
    if color_by == "Regime":
        map_df['Color'] = map_df['Regime']
        color_discrete_map = {
            "Dot-com Bust": "#e74c3c",
            "Recovery & Tightening": "#3498db",
            "Financial Crisis": "#9b59b6",
            "Zero Rate Era": "#2ecc71",
            "Normalization": "#f39c12",
            "COVID Response": "#1abc9c",
            "Inflation Fight": "#e74c3c",
            "Other": "#7f8c8d"
        }
    elif color_by == "Year":
        map_df['Color'] = map_df['Year']
        color_discrete_map = None
    elif color_by == "Hawkishness" and 'Hawkishness' in map_df.columns:
        map_df['Color'] = map_df['Hawkishness']
        color_discrete_map = None
    elif color_by == "Market Reaction" and 'Market_Reaction' in map_df.columns:
        map_df['Color'] = map_df['Market_Reaction']
        color_discrete_map = None
    else:
        map_df['Color'] = map_df['Year']
        color_discrete_map = None

    # Sort by date for trajectory
    map_df = map_df.sort_values('Date')

    # Create hover text
    map_df['Hover'] = map_df.apply(
        lambda row: f"Date: {row['Date'].strftime('%Y-%m-%d')}<br>" +
                   f"Type: {row.get('Type', 'N/A')}<br>" +
                   f"Regime: {row.get('Regime', 'N/A')}<br>" +
                   (f"Hawkishness: {row['Hawkishness']:.2f}<br>" if pd.notna(row.get('Hawkishness')) else "") +
                   (f"Market Reaction: {row['Market_Reaction']:+.1f} bp" if pd.notna(row.get('Market_Reaction')) else ""),
        axis=1
    )

    # Create visualization
    if view_mode == "2D":
        fig = go.Figure()

        # Add trajectory line if enabled
        if show_trajectory:
            fig.add_trace(go.Scatter(
                x=map_df['PC1'],
                y=map_df['PC2'],
                mode='lines',
                line=dict(color='lightgray', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add scatter points
        if color_by == "Regime":
            for regime in map_df['Regime'].unique():
                regime_df = map_df[map_df['Regime'] == regime]
                fig.add_trace(go.Scatter(
                    x=regime_df['PC1'],
                    y=regime_df['PC2'],
                    mode='markers',
                    name=regime,
                    marker=dict(
                        size=10,
                        color=color_discrete_map.get(regime, '#7f8c8d'),
                        opacity=0.7
                    ),
                    text=regime_df['Hover'],
                    hovertemplate='%{text}<extra></extra>'
                ))
        else:
            fig.add_trace(go.Scatter(
                x=map_df['PC1'],
                y=map_df['PC2'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=map_df['Color'],
                    colorscale='RdBu_r' if color_by in ['Hawkishness', 'Market Reaction'] else 'Viridis',
                    showscale=True,
                    colorbar=dict(title=color_by),
                    opacity=0.7
                ),
                text=map_df['Hover'],
                hovertemplate='%{text}<extra></extra>'
            ))

        fig.update_layout(
            title=f"FOMC Statement Space (2D) - {variance_info['total']*100:.1f}% variance explained",
            xaxis_title=f"PC1 ({variance_info['PC1']*100:.1f}%)",
            yaxis_title=f"PC2 ({variance_info['PC2']*100:.1f}%)",
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02
            )
        )

    else:  # 3D view
        if color_by == "Regime":
            fig = go.Figure()
            for regime in map_df['Regime'].unique():
                regime_df = map_df[map_df['Regime'] == regime]
                fig.add_trace(go.Scatter3d(
                    x=regime_df['PC1'],
                    y=regime_df['PC2'],
                    z=regime_df['PC3'],
                    mode='markers',
                    name=regime,
                    marker=dict(
                        size=6,
                        color=color_discrete_map.get(regime, '#7f8c8d'),
                        opacity=0.7
                    ),
                    text=regime_df['Hover'],
                    hovertemplate='%{text}<extra></extra>'
                ))
        else:
            fig = go.Figure(data=[go.Scatter3d(
                x=map_df['PC1'],
                y=map_df['PC2'],
                z=map_df['PC3'],
                mode='markers',
                marker=dict(
                    size=6,
                    color=map_df['Color'],
                    colorscale='RdBu_r' if color_by in ['Hawkishness', 'Market Reaction'] else 'Viridis',
                    showscale=True,
                    colorbar=dict(title=color_by),
                    opacity=0.7
                ),
                text=map_df['Hover'],
                hovertemplate='%{text}<extra></extra>'
            )])

        fig.update_layout(
            title=f"FOMC Statement Space (3D) - {variance_info['total']*100:.1f}% variance explained",
            scene=dict(
                xaxis_title=f"PC1 ({variance_info['PC1']*100:.1f}%)",
                yaxis_title=f"PC2 ({variance_info['PC2']*100:.1f}%)",
                zaxis_title=f"PC3 ({variance_info['PC3']*100:.1f}%)",
            ),
            height=700
        )

    st.plotly_chart(fig, use_container_width=True)

    # Variance explained info
    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
    with col_v1:
        st.metric("PC1 Variance", f"{variance_info['PC1']*100:.1f}%")
    with col_v2:
        st.metric("PC2 Variance", f"{variance_info['PC2']*100:.1f}%")
    with col_v3:
        st.metric("PC3 Variance", f"{variance_info['PC3']*100:.1f}%")
    with col_v4:
        st.metric("Total (3 PCs)", f"{variance_info['total']*100:.1f}%")

    # Top loadings explanation
    with st.expander("What do the Principal Components represent?"):
        st.markdown("""
        **Principal Component Analysis (PCA)** compresses the 99 linguistic features into a few key dimensions.
        Here are the top features driving each component:
        """)

        for pc, info in loadings_info.items():
            st.markdown(f"**{pc}** - Top contributing features:")
            for feat, loading in list(info['loadings'].items())[:5]:
                direction = "+" if loading > 0 else "-"
                st.markdown(f"  - `{feat}`: {direction}{abs(loading):.3f}")

    # Regime summary
    with st.expander("Economic Regime Definitions"):
        st.markdown("""
        | Regime | Period | Description |
        |--------|--------|-------------|
        | Dot-com Bust | 2000-2001 | Tech bubble collapse, rate cuts |
        | Recovery & Tightening | 2002-2006 | Economic recovery, gradual rate hikes |
        | Financial Crisis | 2007-2009 | Great Recession, emergency cuts |
        | Zero Rate Era | 2010-2015 | Near-zero rates, QE programs |
        | Normalization | 2016-2019 | Gradual rate increases |
        | COVID Response | 2020-2021 | Pandemic emergency, back to zero |
        | Inflation Fight | 2022-2025 | Aggressive tightening cycle |
        """)

# ============================================================================
# TAB 3: HISTORICAL DATA
# ============================================================================
with tab3:
    st.header("Historical FOMC Reactions")

    # Filter to statements with market data
    plot_df = df[df[target_col].notna()].copy()

    # Time series plot
    fig_ts = go.Figure()

    # Add scatter plot with hawkishness coloring
    if 'composite_hawk' in df.columns:
        color_data = plot_df['composite_hawk']
        colorbar_title = "Hawkishness"
    else:
        color_data = plot_df[target_col]
        colorbar_title = "Reaction (bp)"

    fig_ts.add_trace(go.Scatter(
        x=plot_df['Date'],
        y=plot_df[target_col],
        mode='markers+lines',
        name=target_yield,
        marker=dict(
            size=10,
            color=color_data,
            colorscale='RdBu_r',
            showscale=True,
            colorbar=dict(title=colorbar_title)
        ),
        text=plot_df['Type'],
        hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Reaction: %{y:+.1f} bp<extra></extra>'
    ))
    fig_ts.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_ts.update_layout(
        title=f"{target_yield} Market Reactions to FOMC Communications",
        xaxis_title="Date",
        yaxis_title="Yield Change (bp)",
        height=500
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # Statistics
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    with col_s1:
        st.metric("Total Statements", len(plot_df))

    with col_s2:
        st.metric("Average Reaction", f"{plot_df[target_col].mean():.2f} bp")

    with col_s3:
        st.metric("Std Deviation", f"{plot_df[target_col].std():.2f} bp")

    with col_s4:
        positive_pct = (plot_df[target_col] > 0).mean()
        st.metric("% Hawkish Reactions", f"{positive_pct:.1%}")

# ============================================================================
# TAB 4: MODEL PERFORMANCE
# ============================================================================
with tab4:
    st.header("Model Performance Metrics")

    st.markdown("""
    This tool uses a **Random Forest** model trained on 432 FOMC statements (2000-2025)
    with **112 features** including:
    - GPT-4, FinBERT, BART sentiment scores
    - Sentence-level change detection
    - **Word-level linguistic features** (hedge words, certainty, tense shifts, negation, etc.)
    """)

    # Load actual model results if available
    try:
        model_results = pd.read_csv('model_results.csv')
        st.dataframe(model_results, use_container_width=True)

        # Plot RMSE comparison
        fig_perf = px.bar(
            model_results,
            x='model',
            y='cv_rmse',
            title="Model Performance (Cross-Validation RMSE)",
            labels={'cv_rmse': 'RMSE (basis points)', 'model': 'Model'},
            text='cv_rmse'
        )
        fig_perf.update_traces(texttemplate='%{text:.2f} bp', textposition='outside')
        st.plotly_chart(fig_perf, use_container_width=True)

    except:
        st.info("Run `python run_analysis.py` to generate model performance metrics.")

    # Feature importance
    st.markdown("### Top Features")
    try:
        feature_imp = pd.read_csv('feature_importance.csv')

        # Show top 20
        top_features = feature_imp.head(20)

        fig_feat = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title="Top 20 Most Important Features (SHAP Values)",
            labels={'importance': 'SHAP Importance', 'feature': 'Feature'}
        )
        fig_feat.update_layout(height=600)
        st.plotly_chart(fig_feat, use_container_width=True)

        # Highlight word-level features
        word_level = feature_imp[feature_imp['feature'].str.startswith('subtle_')]
        if len(word_level) > 0:
            st.markdown("### 🌟 Word-Level Feature Performance")
            st.markdown(f"**{len(word_level)} word-level features** successfully capture market-relevant signals!")
            st.markdown(f"**Highest ranked**: `{word_level.iloc[0]['feature']}` (rank #{feature_imp[feature_imp['feature'] == word_level.iloc[0]['feature']].index[0] + 1})")

            with st.expander("See All Word-Level Features"):
                st.dataframe(word_level, use_container_width=True)

    except:
        st.info("Run `python run_shap_analysis.py` to generate feature importance rankings.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><strong>FOMC Diagnostic Tool</strong> | Built with Streamlit | 2025</p>
    <p><em>Diagnostic analysis, not point predictions. Acknowledges uncertainty.</em></p>
</div>
""", unsafe_allow_html=True)
