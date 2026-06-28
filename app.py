import os
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="BioIntelligence Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure scripts directory is in path
import sys
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
scripts_dir = os.path.join(project_dir, "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

import auth_utils
auth_utils.check_authentication()


# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #0f1116;
        color: #e2e8f0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e293b;
        border-radius: 4px 4px 0px 0px;
        gap: 2px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
        font-weight: bold;
    }
    div[data-testid="metric-container"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to get database connection
DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "biointel.db")

def get_connection():
    conn = sqlite3.connect(DB_FILE)
    return conn

# Sidebar configuration
st.sidebar.title("🧬 BioIntelligence Platform")
st.sidebar.markdown("### Zero-Docker local analytics dashboard")
st.sidebar.info("This interactive interface visualizes local genomic, neuroimaging, and model inference pipelines directly from your host SQLite database.")

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 User Workspace Pages")
st.sidebar.markdown(
    "- **[Interactive Lab](/Interactive_Lab)**: Upload your own VCF or CSV inputs to run predictions.\n"
    "- **[Analysis History](/Analysis_History)**: View, download, or delete past session runs.\n"
    "- **[Comparison View](/Comparison_View)**: Side-by-side analysis comparison charts."
)



# =========================================================================
# Cloud Auto-Initialization: Self-bootstrap if DB or model is missing
# On Streamlit Cloud's ephemeral filesystem, these may need to be rebuilt
# =========================================================================
def _auto_initialize():
    """Auto-initialize database, seed data, and train model if missing."""
    import subprocess
    project_dir_local = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(project_dir_local, "biointel.db")
    model_path = os.path.join(project_dir_local, "models", "genomic_attention.pt")
    
    needs_init = not os.path.exists(db_path)
    
    if needs_init:
        st.info("🔧 First-time setup: Initializing database...")
        init_script = os.path.join(project_dir_local, "db", "init_sqlite.py")
        if os.path.exists(init_script):
            subprocess.run([sys.executable, init_script], check=True)
        
        # Seed with balanced variants
        scale_script = os.path.join(project_dir_local, "scripts", "scale_dataset.py")
        if os.path.exists(scale_script):
            st.info("🧬 Seeding 10,000 balanced ClinVar variants...")
            subprocess.run([sys.executable, scale_script], check=True)
        
        # Run pipelines to populate all tables
        pipeline_script = os.path.join(project_dir_local, "run_pipelines.py")
        if os.path.exists(pipeline_script):
            st.info("⚙️ Running analysis pipelines...")
            subprocess.run([sys.executable, pipeline_script], check=True)
    
    if not os.path.exists(model_path):
        st.info("🧠 Training genomic model (this takes ~60 seconds)...")
        model_script = os.path.join(project_dir_local, "scripts", "genomic_model.py")
        if os.path.exists(model_script):
            subprocess.run([sys.executable, model_script], check=True)

# Run auto-init once per server lifecycle
if not os.path.exists(DB_FILE) or not os.path.exists(os.path.join(project_dir, "models", "genomic_attention.pt")):
    with st.spinner("🚀 Initializing BioIntelligence Platform (first run only)..."):
        _auto_initialize()
    st.rerun()

# Load summary statistics for KPIs
conn = get_connection()
try:
    total_subjects = pd.read_sql_query("SELECT COUNT(*) FROM subjects", conn).iloc[0, 0]
    total_variants = pd.read_sql_query("SELECT COUNT(*) FROM variants", conn).iloc[0, 0]
    path_variants = pd.read_sql_query("SELECT COUNT(*) FROM variants WHERE clinvar_sig = 'Pathogenic'", conn).iloc[0, 0]
    avg_delta = pd.read_sql_query("SELECT AVG(ABS(brain_age_delta)) FROM mri_model_predictions", conn).iloc[0, 0]
except Exception as e:
    st.error(f"Error reading from database: {e}")
    st.stop()
finally:
    conn.close()

# Layout layout - Header KPIs
st.title("🧠 Multi-Modal BioIntelligence Dashboard")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Subjects", f"{total_subjects}")
with col2:
    st.metric("Variants Cataloged", f"{total_variants}")
with col3:
    st.metric("Pathogenic Variants", f"{path_variants}")
with col4:
    st.metric("Mean Abs Brain Age Gap", f"{avg_delta:.2f} yrs" if avg_delta is not None else "N/A")

# Create tabs for the six dashboards
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Ancestry × Brain", 
    "🧬 Variants Analysis", 
    "💻 DNA Models", 
    "📈 Pipeline Monitor", 
    "🧠 Brain Age Predictions",
    "👤 Referrals & Invite Logs"
])

# =========================================================================
# TAB 1: Ancestry × Brain
# =========================================================================
with tab1:
    st.header("Ancestry x Brain Volume Correlation")
    st.write("Examine the relationship between ancestry principal components and structural brain volume.")
    
    conn = get_connection()
    df_pca_brain = pd.read_sql_query("SELECT * FROM subject_pcs_brain_joined", conn)
    df_matrix = pd.read_sql_query("SELECT * FROM pc_brain_correlation_matrix", conn)
    conn.close()
    
    if not df_pca_brain.empty:
        col1_1, col1_2 = st.columns([3, 2])
        
        with col1_1:
            st.subheader("Ancestry PCA Cluster Colored by Brain Volume")
            # Create interactive scatter plot
            fig = px.scatter(
                df_pca_brain,
                x="pc1",
                y="pc2",
                color="total_brain_volume",
                hover_name="subject_id",
                symbol="super_population",
                color_continuous_scale="Purples",
                labels={"pc1": "Ancestry PC1", "pc2": "Ancestry PC2", "total_brain_volume": "Total Brain Volume (mm³)"},
                title="PC1 vs PC2 (Shape by Ancestry Group, Color by Brain Volume)"
            )
            fig.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        with col1_2:
            st.subheader("PC × Brain Region Pearson Correlation Matrix")
            if not df_matrix.empty:
                # Drop duplicates defensively to prevent reshape error
                df_matrix = df_matrix.drop_duplicates(subset=["region_name", "pc_name"])
                # Pivot df_matrix to create correlation matrix
                # columns: pc_name, index: region_name, values: pearson_r
                df_pivot = df_matrix.pivot(index="region_name", columns="pc_name", values="pearson_r")
                # Sort columns PC1, PC2...
                pc_cols = sorted(df_pivot.columns, key=lambda x: int(x.replace("PC", "")))
                df_pivot = df_pivot[pc_cols]
                
                fig_hm = px.imshow(
                    df_pivot,
                    color_continuous_scale="RdBu",
                    color_continuous_midpoint=0,
                    aspect="auto",
                    labels={"color": "Pearson r"},
                    title="Brain Region Volume vs Ancestry PCs Correlations"
                )
                fig_hm.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig_hm, use_container_width=True)
            else:
                st.info("No correlation matrix data available yet.")
    else:
        st.info("No genetics or brain PCA data available.")

# =========================================================================
# TAB 2: Variants
# =========================================================================
with tab2:
    st.header("Genomic Variant Distribution & ClinVar Significance")
    st.write("Browse sequence variant statistics, allele frequencies, and pathogenic mutation classifications.")
    
    conn = get_connection()
    df_vars = pd.read_sql_query("SELECT chrom, pos, clinvar_sig, gnomad_af, gene_symbol FROM genomic_vars WHERE var_id IS NOT NULL", conn)
    conn.close()
    
    if not df_vars.empty:
        col2_1, col2_2 = st.columns(2)
        
        with col2_1:
            st.subheader("Variants Count per Chromosome")
            chrom_counts = df_vars["chrom"].value_counts().reset_index()
            chrom_counts.columns = ["Chromosome", "Count"]
            fig_chrom = px.bar(
                chrom_counts,
                x="Chromosome",
                y="Count",
                color="Chromosome",
                title="Sequence Variant Density per Chromosome",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_chrom.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_chrom, use_container_width=True)
            
        with col2_2:
            st.subheader("gnomAD Population Allele Frequency Distribution")
            fig_af = px.histogram(
                df_vars,
                x="gnomad_af",
                nbins=30,
                color_discrete_sequence=["#10b981"],
                title="gnomad Allele Frequency Histogram (Rare vs Common)",
                labels={"gnomad_af": "Allele Frequency"}
            )
            fig_af.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_af, use_container_width=True)
            
        st.subheader("Pathogenic & Likely Pathogenic ClinVar Variants Table")
        df_pathogenic = df_vars[df_vars["clinvar_sig"].isin(["Pathogenic", "Likely pathogenic"])]
        if not df_pathogenic.empty:
            st.dataframe(
                df_pathogenic.reset_index(drop=True),
                use_container_width=True,
                height=300
            )
        else:
            st.info("No pathogenic variants cataloged in current run.")
    else:
        st.info("No variant distribution data available.")

# =========================================================================
# TAB 3: DNA Models
# =========================================================================
with tab3:
    st.header("DNA Language Model Embeddings & Inferences")
    st.write("Analyze DNABERT-2 / HyenaDNA CLS token embeddings clustered via principal components.")
    
    conn = get_connection()
    df_dna_preds = pd.read_sql_query("SELECT * FROM dna_model_predictions_pca", conn)
    conn.close()
    
    if not df_dna_preds.empty:
        col3_1, col3_2 = st.columns(2)
        
        with col3_1:
            st.subheader("CLS Embedding Space (PCA Dimension Projection)")
            fig_emb = px.scatter(
                df_dna_preds,
                x="pca_dim1",
                y="pca_dim2",
                color="pred_label",
                hover_name="subject_id",
                title="Variant Sequence Context PCA Embedding Layout",
                labels={"pca_dim1": "Embedding PC1", "pca_dim2": "Embedding PC2"},
                color_discrete_map={"Pathogenic": "#ef4444", "Benign": "#10b981", "Likely pathogenic": "#f59e0b", "Uncertain significance": "#6b7280"}
            )
            fig_emb.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(fig_emb, use_container_width=True)
            
        with col3_2:
            st.subheader("Model Pathogenicity Prediction Score Distributions")
            fig_score = px.histogram(
                df_dna_preds,
                x="pred_score",
                color="pred_label",
                nbins=30,
                marginal="rug",
                title="DNABERT-2 Pathogenicity Score Spread",
                labels={"pred_score": "Confidence Score"},
                color_discrete_map={"Pathogenic": "#ef4444", "Benign": "#10b981", "Likely pathogenic": "#f59e0b", "Uncertain significance": "#6b7280"}
            )
            fig_score.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(fig_score, use_container_width=True)
    else:
        st.info("No DNA model inference predictions available.")

# =========================================================================
# TAB 4: Pipeline Monitor
# =========================================================================
with tab4:
    st.header("Pipeline Run Audit & Execution Status")
    st.write("Monitor task start times, running status, durations, and system message logs.")
    
    conn = get_connection()
    df_runs = pd.read_sql_query("SELECT * FROM pipeline_runs ORDER BY started_at DESC", conn)
    conn.close()
    
    if not df_runs.empty:
        # Calculate durations where completed
        df_runs["started_dt"] = pd.to_datetime(df_runs["started_at"])
        df_runs["completed_dt"] = pd.to_datetime(df_runs["completed_at"])
        df_runs["duration_sec"] = (df_runs["completed_dt"] - df_runs["started_dt"]).dt.total_seconds()
        
        col4_1, col4_2 = st.columns([1, 2])
        
        with col4_1:
            st.subheader("Pipeline Run Success Rate")
            run_stats = df_runs["status"].value_counts().reset_index()
            run_stats.columns = ["Status", "Count"]
            fig_pie = px.pie(
                run_stats,
                names="Status",
                values="Count",
                hole=0.4,
                title="Dag Executions Pie Chart",
                color="Status",
                color_discrete_map={"success": "#10b981", "failed": "#ef4444", "running": "#3b82f6"}
            )
            fig_pie.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col4_2:
            st.subheader("Pipeline Processing Duration (seconds)")
            # Filter successful runs
            df_success = df_runs[df_runs["status"] == "success"].dropna(subset=["duration_sec"])
            if not df_success.empty:
                fig_dur = px.bar(
                    df_success,
                    x="dag_id",
                    y="duration_sec",
                    color="dag_id",
                    title="Execution Time per Sequential Pipeline Phase",
                    labels={"duration_sec": "Duration (sec)", "dag_id": "Orchestrator Task ID"},
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_dur.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig_dur, use_container_width=True)
            else:
                st.info("No successful runs with duration records available.")
                
        st.subheader("Pipeline History log")
        st.dataframe(
            df_runs[["run_id", "dag_id", "status", "started_at", "completed_at", "n_subjects_processed", "error_message"]],
            use_container_width=True
        )
    else:
        st.info("No pipeline run records logged in database.")

# =========================================================================
# TAB 5: Brain Age
# =========================================================================
with tab5:
    st.header("Brain Age Estimation via Regional Morphometry")
    st.write("Compare chronological age vs Ridge Regression predicted brain age with confidence intervals.")
    
    conn = get_connection()
    df_age = pd.read_sql_query("SELECT * FROM brain_age_predictions", conn)
    conn.close()
    
    if not df_age.empty:
        col5_1, col5_2 = st.columns(2)
        
        with col5_1:
            st.subheader("Chronological vs Predicted Brain Age Scatter")
            # Let's read raw predictions to get confidence bounds
            conn = get_connection()
            df_raw_predictions = pd.read_sql_query("SELECT subject_id, ci_lower, ci_upper FROM mri_model_predictions", conn)
            conn.close()
            
            df_joined_age = df_age.merge(df_raw_predictions, on="subject_id", how="left")
            
            # Sort by chronological age for clear line rendering
            df_joined_age = df_joined_age.sort_values("chronological_age")
            
            fig_scatter = go.Figure()
            
            # Add ideal x=y reference line
            min_val = min(df_joined_age["chronological_age"].min(), df_joined_age["predicted_brain_age"].min()) - 2
            max_val = max(df_joined_age["chronological_age"].max(), df_joined_age["predicted_brain_age"].max()) + 2
            fig_scatter.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Chronological Age (Ideal)',
                line=dict(color='gray', dash='dash')
            ))
            
            # Add subjects scatter
            fig_scatter.add_trace(go.Scatter(
                x=df_joined_age["chronological_age"],
                y=df_joined_age["predicted_brain_age"],
                mode='markers',
                name='Subject Predictions',
                marker=dict(color='#8b5cf6', size=8),
                text=df_joined_age["subject_id"]
            ))
            
            # Add Confidence Interval bounds
            if "ci_lower" in df_joined_age.columns and not df_joined_age["ci_lower"].isna().all():
                fig_scatter.add_trace(go.Scatter(
                    x=df_joined_age["chronological_age"],
                    y=df_joined_age["ci_upper"],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    name='CI Upper'
                ))
                fig_scatter.add_trace(go.Scatter(
                    x=df_joined_age["chronological_age"],
                    y=df_joined_age["ci_lower"],
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(139, 92, 246, 0.15)',
                    line=dict(width=0),
                    name='95% Confidence Interval'
                ))
                
            fig_scatter.update_layout(
                template="plotly_dark",
                title="Ridge Regression Predicted Brain Age vs Actual Chronological Age",
                xaxis_title="Chronological Age (years)",
                yaxis_title="Predicted Brain Age (years)",
                height=450
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with col5_2:
            st.subheader("Brain Age Delta (Predicted - Chronological) Histogram")
            fig_delta = px.histogram(
                df_joined_age,
                x="brain_age_delta",
                nbins=15,
                color="sex",
                title="Brain Age Delta Spread colored by Sex",
                labels={"brain_age_delta": "Brain Age Delta (years)"},
                color_discrete_map={"M": "#3b82f6", "F": "#ec4899", "U": "#9b9b9b"}
            )
            fig_delta.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(fig_delta, use_container_width=True)
            
        # Display narrative summary
        st.subheader("Subject Integrated LLM Narratives")
        conn = get_connection()
        df_narratives = pd.read_sql_query("SELECT subject_id, analysis_type, final_narrative, ethical_disclaimer FROM llm_narratives", conn)
        conn.close()
        
        if not df_narratives.empty:
            selected_sub = st.selectbox("Select subject ID to read generated clinical narrative & lay summary", df_narratives["subject_id"].unique())
            sub_row = df_narratives[df_narratives["subject_id"] == selected_sub].iloc[0]
            
            st.markdown(f"**Analysis Mode:** `{sub_row['analysis_type'].upper()}`")
            st.text_area("Narrative Summary:", value=sub_row["final_narrative"], height=350, disabled=True)
        else:
            st.info("No LLM narratives generated yet.")
    else:
        st.info("No brain age estimation predictions available.")

# =========================================================================
# TAB 6: Referrals & Invite Logs
# =========================================================================
with tab6:
    auth_utils.render_referral_dashboard()
