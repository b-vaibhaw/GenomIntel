import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Interactive Lab — BioIntelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure project and scripts directories are in path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
scripts_dir = os.path.join(project_dir, "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

import auth_utils
auth_utils.check_authentication()

import user_analysis_engine


# Custom Styling (aligned with app.py dark mode)
st.markdown("""
<style>
    .main {
        background-color: #0f1116;
        color: #e2e8f0;
    }
    div[data-testid="metric-container"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
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
        background-color: #10b981 !important;
        color: white !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Helper for db connection
DB_FILE = os.path.join(project_dir, "biointel.db")
def get_connection():
    return sqlite3.connect(DB_FILE)

# Sidebar - Session Control
st.sidebar.title("🔬 Interactive Lab")

st.sidebar.page_link("app.py", label="Main Dashboard", icon="🏠")
st.sidebar.page_link("pages/1_Interactive_Lab.py", label="Interactive Lab", icon="🔬")
st.sidebar.page_link("pages/2_Analysis_History.py", label="Analysis History", icon="📜")
st.sidebar.page_link("pages/3_Comparison_View.py", label="Comparison View", icon="⚖️")
st.sidebar.markdown("---")


# Get or refresh sessions list
sessions_df = user_analysis_engine.get_all_sessions(username=st.session_state.get("username"))

if not sessions_df.empty:
    session_options = {row["session_id"]: f"{row['session_name']} ({row['analysis_count']} runs)" for _, row in sessions_df.iterrows()}
    active_session_id = st.sidebar.selectbox(
        "Select Active Session",
        options=list(session_options.keys()),
        format_func=lambda x: session_options[x]
    )
    active_session_name = sessions_df[sessions_df["session_id"] == active_session_id].iloc[0]["session_name"]
else:
    st.sidebar.warning("No active session. Please create one below first.")
    active_session_id = None
    active_session_name = None

st.sidebar.markdown("### Create New Session")
new_session_name = st.sidebar.text_input("Session Name", placeholder="e.g. Cohort_Study_A")
new_session_desc = st.sidebar.text_area("Description (optional)", placeholder="Brief details about the target subject cohort...")

if st.sidebar.button("Create Session"):
    if new_session_name.strip():
        new_id = user_analysis_engine.create_session(new_session_name, new_session_desc, username=st.session_state.get("username"))
        st.sidebar.success(f"Session '{new_session_name}' created!")
        st.rerun()
    else:
        st.sidebar.error("Session name cannot be empty.")


# Title
st.title("🔬 Interactive Diagnostic & Analysis Workspace")
st.markdown("Run custom brain age regressions, genetic sequence annotations, and DNA model embeddings.")

# Retrieve and display credits remaining
username = st.session_state.get("username")
credits_remaining = user_analysis_engine.get_user_credits(username) if username else None
is_credit_exhausted = (credits_remaining is not None and credits_remaining <= 0)

if credits_remaining is not None:
    if is_credit_exhausted:
        st.error(f"🛑 **Credit Limit Reached**: You have used all your demo credits (0 remaining). Please contact an administrator to upgrade your account.")
    else:
        st.warning(f"💡 **Demo Account**: You have **{credits_remaining}** credits remaining to test your own data.")

st.markdown("---")

# Main Interface Tabs
tab_brain, tab_genetics, tab_dna = st.tabs([
    "🧠 Brain Age Estimator", 
    "🧬 Custom VCF Annotator", 
    "💻 DNA Model Embedding"
])

# =========================================================================
# TAB 1: Brain Age Estimator
# =========================================================================
with tab_brain:
    st.header("Brain Age Estimation via Regional Morphometry")
    st.write("Input structural regional brain volumes to estimate the subject's biological brain age compared to OASIS/IXI cohorts.")
    
    if active_session_id is None:
        st.warning("⚠️ You must select or create an active session in the sidebar to run analyses.")
    else:
        col_inputs, col_presets = st.columns([3, 1])
        
        with col_presets:
            st.subheader("Presets")
            st.write("Use typical values to prefill inputs:")
            if st.button("Prefill typical healthy 40yo"):
                st.session_state["chron_age"] = 40.0
                st.session_state["sex_input"] = "M"
                st.session_state["vol_hippocampus"] = 3800.0
                st.session_state["vol_entorhinal"] = 1600.0
                st.session_state["vol_fusiform"] = 9500.0
                st.session_state["vol_inferior"] = 11500.0
                st.session_state["vol_middle"] = 12000.0
                
            if st.button("Prefill typical aged/atrophy 75yo"):
                st.session_state["chron_age"] = 75.0
                st.session_state["sex_input"] = "F"
                st.session_state["vol_hippocampus"] = 2800.0
                st.session_state["vol_entorhinal"] = 1100.0
                st.session_state["vol_fusiform"] = 8200.0
                st.session_state["vol_inferior"] = 9800.0
                st.session_state["vol_middle"] = 10100.0

        with col_inputs:
            st.subheader("Subject Metadata")
            sub_label = st.text_input("Subject Label", value="Subject_001", key="brain_sub_label")
            chron_age = st.number_input("Chronological Age (years)", min_value=1.0, max_value=120.0, value=st.session_state.get("chron_age", 40.0), step=1.0, key="brain_chron_age")
            sex = st.selectbox("Sex", options=["M", "F", "U"], index=["M", "F", "U"].index(st.session_state.get("sex_input", "M")), key="brain_sex")
            
            st.subheader("MRI Regional Brain Volumes (mm³)")
            
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                vol_hippo = st.number_input("Hippocampus Volume", min_value=1000.0, max_value=8000.0, value=st.session_state.get("vol_hippocampus", 3800.0), step=50.0)
                vol_ento = st.number_input("Entorhinal Cortex Volume", min_value=500.0, max_value=4000.0, value=st.session_state.get("vol_entorhinal", 1600.0), step=50.0)
                vol_fusi = st.number_input("Fusiform Gyrus Volume", min_value=4000.0, max_value=20000.0, value=st.session_state.get("vol_fusiform", 9500.0), step=100.0)
            with col_v2:
                vol_inf_temp = st.number_input("Inferior Temporal Gyrus Volume", min_value=4000.0, max_value=25000.0, value=st.session_state.get("vol_inferior", 11500.0), step=100.0)
                vol_mid_temp = st.number_input("Middle Temporal Gyrus Volume", min_value=4000.0, max_value=25000.0, value=st.session_state.get("vol_middle", 12000.0), step=100.0)
                
            notes = st.text_area("Analysis Notes", placeholder="e.g. Patient presents with mild cognitive symptoms...", key="brain_notes")
            
            if st.button("Run Brain Age Analysis", type="primary", disabled=is_credit_exhausted):
                with st.spinner("Training Ridge model & calculating biological age..."):
                    vols = {
                        "hippocampus": vol_hippo,
                        "entorhinal": vol_ento,
                        "fusiform": vol_fusi,
                        "inferior_temporal": vol_inf_temp,
                        "middle_temporal": vol_mid_temp
                    }
                    try:
                        res = user_analysis_engine.run_interactive_brain_age(
                            session_id=active_session_id,
                            subject_label=sub_label,
                            chronological_age=chron_age,
                            sex=sex,
                            brain_volumes=vols,
                            notes=notes
                        )
                        st.success(f"Analysis saved to session '{active_session_name}'!")
                        
                        # Display Metrics
                        st.markdown("---")
                        st.subheader("Analysis Results")
                        
                        mcol1, mcol2, mcol3 = st.columns(3)
                        with mcol1:
                            st.metric("Chronological Age", f"{res['chronological_age']:.1f} yrs")
                        with mcol2:
                            st.metric("Predicted Brain Age", f"{res['brain_age_predicted']:.1f} yrs")
                        with mcol3:
                            st.metric(
                                "Brain Age Gap (Delta)", 
                                f"{res['brain_age_delta']:+.1f} yrs" if res['brain_age_delta'] is not None else "N/A",
                                delta=res['brain_age_delta'],
                                delta_color="inverse"
                            )
                            
                        # Plot subject against reference cohort
                        st.subheader("Reference Cohort Comparison")
                        conn = get_connection()
                        ref_df = pd.read_sql_query(
                            "SELECT subject_id, chronological_age, predicted_brain_age FROM brain_age_predictions", 
                            conn
                        )
                        conn.close()
                        
                        fig = go.Figure()
                        
                        # Reference dots
                        if not ref_df.empty:
                            fig.add_trace(go.Scatter(
                                x=ref_df["chronological_age"],
                                y=ref_df["predicted_brain_age"],
                                mode="markers",
                                name="Reference Cohort (OASIS/IXI)",
                                marker=dict(color="#475569", size=6, opacity=0.7)
                            ))
                            
                        # Ideal X=Y line
                        min_age = min(ref_df["chronological_age"].min(), res["chronological_age"]) - 5
                        max_age = max(ref_df["chronological_age"].max(), res["chronological_age"]) + 5
                        fig.add_trace(go.Scatter(
                            x=[min_age, max_age],
                            y=[min_age, max_age],
                            mode="lines",
                            name="Ideal Fit (Gap = 0)",
                            line=dict(color="#10b981", dash="dash")
                        ))
                        
                        # Active Subject
                        fig.add_trace(go.Scatter(
                            x=[res["chronological_age"]],
                            y=[res["brain_age_predicted"]],
                            mode="markers",
                            name=f"Current Subject ({res['subject_label']})",
                            marker=dict(color="#8b5cf6", size=15, symbol="star", line=dict(color="white", width=2))
                        ))
                        
                        fig.update_layout(
                            template="plotly_dark",
                            xaxis_title="Chronological Age (years)",
                            yaxis_title="Predicted Brain Age (years)",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Report Narrative
                        st.subheader("Generated Clinical Report Summary")
                        report_text = user_analysis_engine.generate_brain_age_narrative(
                            res["subject_label"], res["chronological_age"], res["brain_age_predicted"], res["brain_age_delta"]
                        )
                        st.markdown(report_text)
                        
                    except Exception as e:
                        st.error(f"Error running analysis: {e}")

# =========================================================================
# TAB 2: Custom VCF Annotator
# =========================================================================
with tab_genetics:
    st.header("Genomic Variant Annotation & Pathogenicity Cross-Reference")
    st.write("Upload a VCF file or paste variant rows in standard five-column format (`CHROM POS ID REF ALT`) to fetch clinical significance data.")
    
    if active_session_id is None:
        st.warning("⚠️ You must select or create an active session in the sidebar to run analyses.")
    else:
        st.subheader("Input Variant Data")
        
        # Load Demo Data Button
        demo_vcf = ""
        if st.button("Load Demo VCF Rows"):
            demo_vcf = (
                "22\t50442636\trs587780062\tC\tT\n"
                "22\t50443912\trs148560049\tG\tA\n"
                "22\t50444310\trs587780064\tA\tT\n"
                "22\t50446721\trs150383189\tG\tA\n"
                "22\t50448102\t.\tC\tG"
            )
            
        col_vcf_inputs, col_vcf_file = st.columns(2)
        
        with col_vcf_inputs:
            vcf_text = st.text_area(
                "Paste VCF Content (Space or Tab Separated)",
                value=demo_vcf if demo_vcf else "",
                placeholder="22\t50442636\trs587780062\tC\tT\n22\t50443912\trs148560049\tG\tA",
                height=250,
                key="vcf_text_input"
            )
            
        with col_vcf_file:
            uploaded_file = st.file_uploader(
                "Or Upload VCF/TXT file",
                type=["vcf", "txt", "csv"],
                key="vcf_file_uploader"
            )
            if uploaded_file is not None:
                # Read uploaded file content
                try:
                    vcf_text = uploaded_file.read().decode("utf-8")
                    st.success("File uploaded successfully!")
                except Exception as e:
                    st.error(f"Error reading file: {e}")

        vcf_sub_label = st.text_input("Subject/Cohort Label", value="Subject_001", key="vcf_sub_label")
        vcf_notes = st.text_area("Analysis Notes", placeholder="e.g. Genetic screening for developmental delay...", key="vcf_notes")
        
        if st.button("Run Genomic Annotation Pipeline", type="primary", disabled=is_credit_exhausted):
            if not vcf_text.strip():
                st.error("No variant inputs detected. Please paste text or upload a file.")
            else:
                with st.spinner("Parsing coordinates & annotating against ClinVar/gnomAD database..."):
                    try:
                        parsed_variants = user_analysis_engine.parse_user_vcf_text(vcf_text)
                        if not parsed_variants:
                            st.error("Could not parse any valid variants from the input. Make sure columns are CHROM POS ID REF ALT.")
                        else:
                            res = user_analysis_engine.run_variant_annotation_analysis(
                                session_id=active_session_id,
                                subject_label=vcf_sub_label,
                                variants_list=parsed_variants,
                                notes=vcf_notes
                            )
                            st.success(f"Successfully processed {res['total_variants']} variants and saved to session '{active_session_name}'!")
                            
                            # Metrics
                            st.markdown("---")
                            st.subheader("Annotation Summary")
                            
                            gcol1, gcol2, gcol3 = st.columns(3)
                            with gcol1:
                                st.metric("Total Mapped Variants", f"{res['total_variants']}")
                            with gcol2:
                                st.metric("Pathogenic / Likely Pathogenic", f"{res['pathogenic_count']}")
                            with gcol3:
                                st.metric("Rare Variant Count (AF < 1%)", sum(1 for v in res["variants"] if v.get("gnomad_af", 0) < 0.01))
                                
                            # Mapped variants table
                            st.subheader("Annotated Variants Catalog")
                            df_ann = pd.DataFrame(res["variants"])
                            
                            # Format columns for display
                            display_cols = ["variant_id", "chrom", "pos", "ref", "alt", "rsid", "gene_symbol", "consequence", "clinvar_sig", "gnomad_af", "found_in_db"]
                            df_display = df_ann[display_cols].copy()
                            df_display.columns = ["Variant ID", "Chrom", "Position", "Ref", "Alt", "rsID", "Gene Symbol", "Consequence", "ClinVar Significance", "gnomad AF", "Local DB Hit"]
                            
                            # Styling helper for ClinVar significance
                            def highlight_sig(val):
                                if "Pathogenic" in str(val):
                                    return "background-color: #7f1d1d; color: #fecaca;"
                                elif "Benign" in str(val):
                                    return "background-color: #064e3b; color: #a7f3d0;"
                                return ""
                                
                            st.dataframe(df_display, use_container_width=True)
                            
                            # Distribution pie chart
                            st.subheader("ClinVar Classification Ratio")
                            counts_data = pd.DataFrame(list(res["counts"].items()), columns=["Classification", "Count"])
                            fig = px.pie(
                                counts_data,
                                names="Classification",
                                values="Count",
                                color="Classification",
                                hole=0.3,
                                color_discrete_map={
                                    "Pathogenic": "#ef4444", 
                                    "Likely pathogenic": "#f59e0b",
                                    "Uncertain significance": "#6b7280",
                                    "Likely benign": "#6ee7b7",
                                    "Benign": "#10b981"
                                }
                            )
                            fig.update_layout(template="plotly_dark", height=400)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Report narrative
                            st.subheader("Generated Variant Clinical Report")
                            report_text = user_analysis_engine.generate_variant_narrative(
                                res["subject_label"], res["total_variants"], res["counts"], res["pathogenic_count"]
                            )
                            st.markdown(report_text)
                            
                    except Exception as e:
                        st.error(f"Genomic pipeline failed: {e}")

# =========================================================================
# TAB 3: DNA Model Embedding
# =========================================================================
with tab_dna:
    st.header("DNA Language Model Embedding Projection")
    st.write("Paste raw DNA sequences to extract deep semantic embeddings (using DNABERT-2/HyenaDNA representation spaces) and predict pathogenicity risk scores.")
    
    if active_session_id is None:
        st.warning("⚠️ You must select or create an active session in the sidebar to run analyses.")
    else:
        st.subheader("Sequence Inputs")
        
        demo_dna = ""
        if st.button("Load Sample Human Exonic Segment"):
            # A longer synthetic segment
            demo_dna = (
                "ATGGCCAGCATCGTGGAGGAGCCCGAGCTGCTGGACGGGGCCATCTCCTACGCCCTCAAG"
                "CGCGCCGGCGCCGAGGCCGTGCTGGACGTGCTGGAGGAGCTGGAGCCCGAGACCGTGGTG"
                "CGGGCCGTGCGGGAGCGGCTGGGCGCGGGGCCGGAGCTGCGCTACCTGGACCTGCTGCCC"
                "GCCCGCTACCTGCCCGGCTTCCTGGGCGGCGTGGACGTGGACGCCTTCCGCTACGCCTGA"
            )
            
        dna_seq = st.text_area(
            "Paste DNA Sequence (ACGTN only)",
            value=demo_dna if demo_dna else "",
            placeholder="E.g. ATGGCCAGCATCGTGGAGGAGCCCGAGCTGC...",
            height=200,
            key="dna_text_input"
        )
        
        model_choice = st.selectbox(
            "Select DNA Language Model",
            options=["DNABERT-2", "HyenaDNA (small)"],
            key="dna_model_choice"
        )
        
        dna_sub_label = st.text_input("Subject/Sample Label", value="Subject_001", key="dna_sub_label")
        dna_notes = st.text_area("Analysis Notes", placeholder="e.g. Sequencing from exon region of TP53 gene...", key="dna_notes")
        
        if st.button("Compute DNA Embedding", type="primary", disabled=is_credit_exhausted):
            if not dna_seq.strip():
                st.error("Please enter a DNA sequence to analyze.")
            else:
                with st.spinner("Extracting token hidden states and projecting embeddings..."):
                    try:
                        res = user_analysis_engine.run_dna_sequence_analysis(
                            session_id=active_session_id,
                            subject_label=dna_sub_label,
                            sequence=dna_seq,
                            model_choice=model_choice,
                            notes=dna_notes
                        )
                        st.success(f"Analysis saved to session '{active_session_name}'!")
                        
                        # Metrics
                        st.markdown("---")
                        st.subheader("Embedding Score Details")
                        
                        dcol1, dcol2, dcol3 = st.columns(3)
                        with dcol1:
                            st.metric("Sequence Length Mapped", f"{res['sequence_length']} bp")
                        with dcol2:
                            st.metric("Pathogenicity Prediction", f"{res['pred_label']}", delta=None)
                        with dcol3:
                            st.metric(
                                "Confidence Score", 
                                f"{res['pred_score']:.4f}",
                                delta=None
                            )
                            
                        # Layout Scatter
                        st.subheader("CLS Hidden-State Projection onto Latent Space")
                        st.write("Compare the active sequence's PC projection against pathogenic and benign centroids in embedding space.")
                        
                        fig = go.Figure()
                        
                        # Plotted Centroids
                        fig.add_trace(go.Scatter(
                            x=[0.6 if model_choice == "DNABERT-2" else 0.5],
                            y=[0.6 if model_choice == "DNABERT-2" else 0.5],
                            mode="markers+text",
                            name="Pathogenic Centroid",
                            text=["Pathogenic Centroid"],
                            textposition="top center",
                            marker=dict(color="#ef4444", size=12, symbol="diamond")
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=[-0.6 if model_choice == "DNABERT-2" else -0.5],
                            y=[-0.6 if model_choice == "DNABERT-2" else -0.5],
                            mode="markers+text",
                            name="Benign Centroid",
                            text=["Benign Centroid"],
                            textposition="bottom center",
                            marker=dict(color="#10b981", size=12, symbol="diamond")
                        ))
                        
                        # Subject Embedding Point
                        fig.add_trace(go.Scatter(
                            x=[res["pca_dim1"]],
                            y=[res["pca_dim2"]],
                            mode="markers+text",
                            name=f"Sequence ({res['subject_label']})",
                            text=[res["subject_label"]],
                            textposition="top right",
                            marker=dict(color="#3b82f6", size=16, symbol="star", line=dict(color="white", width=2))
                        ))
                        
                        fig.update_layout(
                            template="plotly_dark",
                            xaxis_title="Latent Dimension 1",
                            yaxis_title="Latent Dimension 2",
                            xaxis=dict(range=[-1.2, 1.2]),
                            yaxis=dict(range=[-1.2, 1.2]),
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Generated narrative
                        st.subheader("Model Narrative Report Summary")
                        report_text = user_analysis_engine.generate_dna_narrative(
                            res["subject_label"], res["model_name"], res
                        )
                        st.markdown(report_text)
                        
                    except Exception as e:
                        st.error(f"DNA model analysis failed: {e}")
