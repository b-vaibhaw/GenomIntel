import os
import sys
import json
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Comparison View — BioIntelligence",
    page_icon="⚖️",
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

import crypto_utils


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
</style>
""", unsafe_allow_html=True)

st.sidebar.subheader("🔬 User Workspace Pages")
st.sidebar.markdown(
    "- **[Main Dashboard](/)**: Multi-Modal Analytics Dashboard.\n"
    "- **[Interactive Lab](/Interactive_Lab)**: Upload your own VCF or CSV inputs to run predictions.\n"
    "- **[Analysis History](/Analysis_History)**: View, download, or delete past session runs.\n"
    "- **[Comparison View](/Comparison_View)**: Side-by-side analysis comparison charts."
)
st.sidebar.markdown("---")

st.title("⚖️ Side-by-Side Analysis Comparison View")

st.markdown("Compare neuroimaging predictions or genomic variant annotations from different subjects or sessions side-by-side.")
st.markdown("---")

# Fetch all user analyses from DB
DB_FILE = os.path.join(project_dir, "biointel.db")
def get_all_user_analyses(username: str = None):
    conn = sqlite3.connect(DB_FILE)
    if username:
        df = pd.read_sql_query(
            """
            SELECT ua.analysis_id, ua.session_id, us.session_name, ua.analysis_type, ua.subject_label, ua.created_at, ua.result_json, ua.notes
            FROM user_analyses ua
            JOIN user_sessions us ON ua.session_id = us.session_id
            WHERE us.username = ?
            ORDER BY ua.created_at DESC
            """,
            conn,
            params=(username,)
        )
    else:
        df = pd.read_sql_query(
            """
            SELECT ua.analysis_id, ua.session_id, us.session_name, ua.analysis_type, ua.subject_label, ua.created_at, ua.result_json, ua.notes
            FROM user_analyses ua
            JOIN user_sessions us ON ua.session_id = us.session_id
            WHERE us.username IS NULL
            ORDER BY ua.created_at DESC
            """,
            conn
        )
    conn.close()
    
    # Decrypt columns
    for col in ["session_name", "subject_label", "result_json", "notes"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: crypto_utils.decrypt_data(x) if pd.notna(x) else x)
            
    return df

all_analyses_df = get_all_user_analyses(username=st.session_state.get("username"))


if all_analyses_df.empty:
    st.info("No saved analysis runs found. Please execute some tests under the **Interactive Lab** page first to generate records.")
else:
    # Form option labels
    def make_label(row):
        type_lbls = {
            "brain_age": "Brain Age",
            "variant_annotation": "Genetics Annotation",
            "dna_embedding": "DNA Embedding"
        }
        name = row["subject_label"]
        type_str = type_lbls.get(row["analysis_type"], "Unknown")
        date_str = row["created_at"].split(".")[0].replace("T", " ")
        return f"{name} — {type_str} ({row['session_name']} | {date_str})"
        
    analysis_options = {row["analysis_id"]: make_label(row) for _, row in all_analyses_df.iterrows()}
    
    col_sel1, col_sel2 = st.columns(2)
    
    with col_sel1:
        selected_id1 = st.selectbox(
            "Select First Analysis Run",
            options=list(analysis_options.keys()),
            format_func=lambda x: analysis_options[x],
            key="compare_sel1"
        )
        
    with col_sel2:
        # Filter same type for second selectbox to guide the user, or show all but check later
        selected_id2 = st.selectbox(
            "Select Second Analysis Run",
            options=list(analysis_options.keys()),
            format_func=lambda x: analysis_options[x],
            key="compare_sel2"
        )
        
    # Get rows
    row1 = all_analyses_df[all_analyses_df["analysis_id"] == selected_id1].iloc[0]
    row2 = all_analyses_df[all_analyses_df["analysis_id"] == selected_id2].iloc[0]
    
    if row1["analysis_type"] != row2["analysis_type"]:
        st.error("⚠️ Selected analyses must be of the same type to compare (e.g. both Brain Age or both Genetics).")
    elif selected_id1 == selected_id2:
        st.warning("Please choose two different runs to perform comparison.")
    else:
        # Load JSON results
        res1 = json.loads(row1["result_json"])
        res2 = json.loads(row2["result_json"])
        
        # Display side-by-side details
        st.markdown("<br/>", unsafe_allow_html=True)
        col_res1, col_res2 = st.columns(2)
        
        # ---------------------------------------------------------------------
        # BRAIN AGE COMPARISON
        # ---------------------------------------------------------------------
        if row1["analysis_type"] == "brain_age":
            with col_res1:
                st.subheader(f"Run A: {row1['subject_label']}")
                st.write(f"**Session:** {row1['session_name']}")
                st.write(f"**Created At:** {row1['created_at'].split('.')[0].replace('T', ' ')}")
                
                m1_1, m1_2, m1_3 = st.columns(3)
                m1_1.metric("Chronological Age", f"{res1.get('chronological_age'):.1f} yrs")
                m1_2.metric("Predicted Brain Age", f"{res1.get('brain_age_predicted'):.1f} yrs")
                m1_3.metric("Gap (Delta)", f"{res1.get('brain_age_delta'):+.1f} yrs")
                
            with col_res2:
                st.subheader(f"Run B: {row2['subject_label']}")
                st.write(f"**Session:** {row2['session_name']}")
                st.write(f"**Created At:** {row2['created_at'].split('.')[0].replace('T', ' ')}")
                
                m2_1, m2_2, m2_3 = st.columns(3)
                m2_1.metric("Chronological Age", f"{res2.get('chronological_age'):.1f} yrs")
                m2_2.metric("Predicted Brain Age", f"{res2.get('brain_age_predicted'):.1f} yrs")
                m2_3.metric("Gap (Delta)", f"{res2.get('brain_age_delta'):+.1f} yrs")
                
            st.markdown("---")
            st.subheader("MRI Regional Brain Volumes Comparison (mm³)")
            
            # Form comparison DataFrame
            vols1 = res1.get("brain_volumes", {})
            vols2 = res2.get("brain_volumes", {})
            
            comparison_rows = []
            for region in ["hippocampus", "entorhinal", "fusiform", "inferior_temporal", "middle_temporal"]:
                comparison_rows.append({
                    "Region": region.replace("_", " ").title(),
                    "Volume (mm³)": vols1.get(region, 0),
                    "Subject": row1["subject_label"]
                })
                comparison_rows.append({
                    "Region": region.replace("_", " ").title(),
                    "Volume (mm³)": vols2.get(region, 0),
                    "Subject": row2["subject_label"]
                })
            df_vols = pd.DataFrame(comparison_rows)
            
            # Interactive grouped bar chart
            fig = px.bar(
                df_vols,
                x="Region",
                y="Volume (mm³)",
                color="Subject",
                barmode="group",
                title="Regional Volumetric Comparison",
                color_discrete_sequence=["#8b5cf6", "#ec4899"]
            )
            fig.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(fig, use_container_width=True)
            
        # ---------------------------------------------------------------------
        # VARIANT ANNOTATION COMPARISON
        # ---------------------------------------------------------------------
        elif row1["analysis_type"] == "variant_annotation":
            with col_res1:
                st.subheader(f"Run A: {row1['subject_label']}")
                st.write(f"**Session:** {row1['session_name']}")
                
                m1_1, m1_2 = st.columns(2)
                m1_1.metric("Total Mapped Variants", f"{res1.get('total_variants')}")
                m1_2.metric("Pathogenic / Likely Pathogenic", f"{res1.get('pathogenic_count')}")
                
            with col_res2:
                st.subheader(f"Run B: {row2['subject_label']}")
                st.write(f"**Session:** {row2['session_name']}")
                
                m2_1, m2_2 = st.columns(2)
                m2_1.metric("Total Mapped Variants", f"{res2.get('total_variants')}")
                m2_2.metric("Pathogenic / Likely Pathogenic", f"{res2.get('pathogenic_count')}")
                
            st.markdown("---")
            
            # Intersect and find shared variants
            vars1 = {v["variant_id"]: v for v in res1.get("variants", [])}
            vars2 = {v["variant_id"]: v for v in res2.get("variants", [])}
            
            shared_ids = set(vars1.keys()).intersection(set(vars2.keys()))
            
            st.subheader(f"Common Variants Mapped (Total Shared: {len(shared_ids)})")
            if shared_ids:
                shared_rows = []
                for vid in shared_ids:
                    v1 = vars1[vid]
                    shared_rows.append({
                        "Variant ID": vid,
                        "Gene Symbol": v1.get("gene_symbol"),
                        "Consequence": v1.get("consequence"),
                        "ClinVar Significance": v1.get("clinvar_sig"),
                        "gnomad AF": v1.get("gnomad_af")
                    })
                df_shared = pd.DataFrame(shared_rows)
                st.dataframe(df_shared, use_container_width=True)
            else:
                st.info("No shared variants found between the two runs.")
                
            # Classifications ratio chart side-by-side
            st.subheader("ClinVar Classifications Distribution Profile")
            counts1 = res1.get("counts", {})
            counts2 = res2.get("counts", {})
            
            dist_rows = []
            for cls_lbl in ["Pathogenic", "Likely pathogenic", "Uncertain significance", "Likely benign", "Benign"]:
                dist_rows.append({
                    "Classification": cls_lbl,
                    "Count": counts1.get(cls_lbl, 0),
                    "Subject": row1["subject_label"]
                })
                dist_rows.append({
                    "Classification": cls_lbl,
                    "Count": counts2.get(cls_lbl, 0),
                    "Subject": row2["subject_label"]
                })
            df_dist = pd.DataFrame(dist_rows)
            
            fig = px.bar(
                df_dist,
                x="Classification",
                y="Count",
                color="Subject",
                barmode="group",
                title="Classifications Spread",
                color_discrete_sequence=["#10b981", "#3b82f6"]
            )
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------------------------------------
        # DNA EMBEDDING COMPARISON
        # ---------------------------------------------------------------------
        elif row1["analysis_type"] == "dna_embedding":
            with col_res1:
                st.subheader(f"Run A: {row1['subject_label']}")
                st.write(f"**Model Choice:** {res1.get('model_name')}")
                
                m1_1, m1_2 = st.columns(2)
                m1_1.metric("Prediction Label", f"{res1.get('pred_label')}")
                m1_2.metric("Pathogenicity Score", f"{res1.get('pred_score'):.4f}")
                
            with col_res2:
                st.subheader(f"Run B: {row2['subject_label']}")
                st.write(f"**Model Choice:** {res2.get('model_name')}")
                
                m2_1, m2_2 = st.columns(2)
                m2_1.metric("Prediction Label", f"{res2.get('pred_label')}")
                m2_2.metric("Pathogenicity Score", f"{res2.get('pred_score'):.4f}")
                
            st.markdown("---")
            st.subheader("Embeddings Latent Space Location")
            
            # Map selected models
            model1 = res1.get("model_name", "DNABERT-2")
            model2 = res2.get("model_name", "DNABERT-2")
            
            if model1 != model2:
                st.warning("⚠️ Comparison of DNA hidden space is only available when both runs use the same model (e.g. both DNABERT-2).")
            else:
                fig = go.Figure()
                
                # Centroids
                fig.add_trace(go.Scatter(
                    x=[0.6 if model1 == "DNABERT-2" else 0.5],
                    y=[0.6 if model1 == "DNABERT-2" else 0.5],
                    mode="markers+text",
                    name="Pathogenic Centroid",
                    text=["Pathogenic Centroid"],
                    textposition="top center",
                    marker=dict(color="#ef4444", size=12, symbol="diamond")
                ))
                
                fig.add_trace(go.Scatter(
                    x=[-0.6 if model1 == "DNABERT-2" else -0.5],
                    y=[-0.6 if model1 == "DNABERT-2" else -0.5],
                    mode="markers+text",
                    name="Benign Centroid",
                    text=["Benign Centroid"],
                    textposition="bottom center",
                    marker=dict(color="#10b981", size=12, symbol="diamond")
                ))
                
                # Point 1
                fig.add_trace(go.Scatter(
                    x=[res1["pca_dim1"]],
                    y=[res1["pca_dim2"]],
                    mode="markers+text",
                    name=f"Run A: {row1['subject_label']}",
                    text=[row1['subject_label']],
                    textposition="top right",
                    marker=dict(color="#8b5cf6", size=16, symbol="star", line=dict(color="white", width=2))
                ))
                
                # Point 2
                fig.add_trace(go.Scatter(
                    x=[res2["pca_dim1"]],
                    y=[res2["pca_dim2"]],
                    mode="markers+text",
                    name=f"Run B: {row2['subject_label']}",
                    text=[row2['subject_label']],
                    textposition="bottom left",
                    marker=dict(color="#f59e0b", size=16, symbol="star", line=dict(color="white", width=2))
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
                
            # Cosine similarity breakdown
            st.subheader("Cosine Similarity Profile to Centroids")
            sim_rows = [
                {
                    "Metric": "Similarity to Pathogenic",
                    "Score": res1.get("sim_pathogenic"),
                    "Subject": row1["subject_label"]
                },
                {
                    "Metric": "Similarity to Pathogenic",
                    "Score": res2.get("sim_pathogenic"),
                    "Subject": row2["subject_label"]
                },
                {
                    "Metric": "Similarity to Benign",
                    "Score": res1.get("sim_benign"),
                    "Subject": row1["subject_label"]
                },
                {
                    "Metric": "Similarity to Benign",
                    "Score": res2.get("sim_benign"),
                    "Subject": row2["subject_label"]
                }
            ]
            df_sim = pd.DataFrame(sim_rows)
            
            fig_sim = px.bar(
                df_sim,
                x="Metric",
                y="Score",
                color="Subject",
                barmode="group",
                title="Similarity Comparison Profiles",
                color_discrete_sequence=["#8b5cf6", "#ec4899"]
            )
            fig_sim.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_sim, use_container_width=True)
