import os
import sys
import json
import sqlite3
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Analysis History — BioIntelligence",
    page_icon="📜",
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
    .history-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-left: 5px solid #10b981;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 12px;
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

st.title("📜 Interactive Analysis History Workspace")

st.markdown("Browse, filter, and export historical analysis runs or delete records from SQLite database storage.")
st.markdown("---")

# Refresh sessions
sessions_df = user_analysis_engine.get_all_sessions(username=st.session_state.get("username"))


if sessions_df.empty:
    st.info("No saved analysis sessions found. Please go to the **Interactive Lab** page to create a session and run some tests!")
else:
    # Sidebar - Select Session
    st.sidebar.title("📜 Session Manager")
    st.sidebar.markdown("---")
    
    session_options = {row["session_id"]: f"{row['session_name']} ({row['analysis_count']} runs)" for _, row in sessions_df.iterrows()}
    selected_session_id = st.sidebar.selectbox(
        "Select Session to View",
        options=list(session_options.keys()),
        format_func=lambda x: session_options[x]
    )
    
    session_row = sessions_df[sessions_df["session_id"] == selected_session_id].iloc[0]
    
    # Session Details
    st.header(f"Session: {session_row['session_name']}")
    if session_row['description']:
        st.markdown(f"**Description:** *{session_row['description']}*")
    st.markdown(f"**Created At:** `{session_row['created_at']}` | **Last Activity:** `{session_row['updated_at']}`")
    
    # Load all analyses in session
    analyses_df = user_analysis_engine.get_session_analyses(selected_session_id)
    
    if analyses_df.empty:
        st.info(f"No analysis runs found in session '{session_row['session_name']}'. Run some tests under this session in the Interactive Lab.")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Danger Zone")
        if st.sidebar.button("Delete Session Entirely"):
            user_analysis_engine.delete_session(selected_session_id)
            st.success("Session deleted successfully.")
            st.rerun()
    else:
        # Session metrics
        st.subheader("Session Statistics")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Total Analysis Runs", f"{len(analyses_df)}")
        with mcol2:
            st.metric("Brain Age Estimates", f"{sum(analyses_df['analysis_type'] == 'brain_age')}")
        with mcol3:
            st.metric("Genomic Annotations", f"{sum(analyses_df['analysis_type'] == 'variant_annotation')}")
        with mcol4:
            st.metric("DNA Embeddings", f"{sum(analyses_df['analysis_type'] == 'dna_embedding')}")
            
        st.markdown("---")
        
        # Download buttons
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            # Generate CSV export
            csv_data = analyses_df.to_csv(index=False)
            st.download_button(
                label="📥 Export Session Data to CSV",
                data=csv_data,
                file_name=f"{session_row['session_name']}_analyses.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_dl2:
            st.write("") # placeholder alignment
            
        st.subheader("Run Timeline")
        
        # Filter options
        filter_type = st.selectbox(
            "Filter by analysis type",
            options=["All Types", "Brain Age Estimation", "Genomic Variant Annotation", "DNA Sequence Embedding"]
        )
        
        # Map filter labels to database values
        type_mapping = {
            "Brain Age Estimation": "brain_age",
            "Genomic Variant Annotation": "variant_annotation",
            "DNA Sequence Embedding": "dna_embedding"
        }
        
        # Display each run in timeline order
        for idx, row in analyses_df.iterrows():
            db_type = row["analysis_type"]
            
            # Apply filter
            if filter_type != "All Types" and db_type != type_mapping[filter_type]:
                continue
                
            # Card styling color based on type
            border_color = "#3b82f6"  # blue
            type_label = "Unknown"
            icon = "❓"
            
            if db_type == "brain_age":
                border_color = "#8b5cf6"  # purple
                type_label = "Brain Age Estimation"
                icon = "🧠"
            elif db_type == "variant_annotation":
                border_color = "#10b981"  # green
                type_label = "Genomic Variant Annotation"
                icon = "🧬"
            elif db_type == "dna_embedding":
                border_color = "#f59e0b"  # amber
                type_label = "DNA Sequence Embedding"
                icon = "💻"
                
            # HTML custom card representation
            st.markdown(
                f"""
                <div style="background-color: #1e293b; border: 1px solid #334155; border-left: 5px solid {border_color}; padding: 15px; border-radius: 6px; margin-bottom: 12px;">
                    <span style="font-size: 1.2em; font-weight: bold;">{icon} {type_label}</span>
                    <span style="float: right; color: #94a3b8; font-size: 0.9em;">Run ID: {row['analysis_id']} | Date: {row['created_at']}</span>
                    <br/>
                    <strong>Subject Label:</strong> {row['subject_label']}
                    {f'<br/><strong>Notes:</strong> <em>{row["notes"]}</em>' if row['notes'] else ''}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Card Expandable Details
            with st.expander("View Full Metrics & Diagnostic Report"):
                col_info, col_actions = st.columns([3, 1])
                
                with col_info:
                    st.subheader("Data Summary")
                    try:
                        result_obj = json.loads(row["result_json"])
                    except (json.JSONDecodeError, TypeError):
                        st.error("⚠️ Failed to parse analysis details. (Check decryption key)")
                        result_obj = {}
                    
                    if db_type == "brain_age" and result_obj:
                        st.write(f"**Chronological Age:** {result_obj.get('chronological_age')} years")
                        st.write(f"**Sex:** {result_obj.get('sex')}")
                        st.write(f"**Predicted Brain Age:** {result_obj.get('brain_age_predicted')} years")
                        st.write(f"**95% CI Range:** {result_obj.get('ci_lower')} - {result_obj.get('ci_upper')} years")
                        st.write(f"**Volume metrics:** `{result_obj.get('brain_volumes')}`")
                        
                    elif db_type == "variant_annotation":
                        st.write(f"**Variants parsed:** {result_obj.get('total_variants')}")
                        st.write(f"**Pathogenic variants found:** {result_obj.get('pathogenic_count')}")
                        st.write(f"**Detailed classifications:** {result_obj.get('counts')}")
                        
                    elif db_type == "dna_embedding":
                        st.write(f"**Sequence Length:** {result_obj.get('sequence_length')} bp")
                        st.write(f"**Model choice:** {result_obj.get('model_name')}")
                        st.write(f"**Predicted Label:** {result_obj.get('pred_label')} (Confidence: {result_obj.get('pred_score')})")
                        st.write(f"**Embedding similarities:** Pathogenic={result_obj.get('sim_pathogenic')}, Benign={result_obj.get('sim_benign')}")
                    
                    # Display report text
                    st.subheader("Report Narrative")
                    st.markdown(row["narrative_text"])
                    
                with col_actions:
                    st.subheader("Actions")
                    
                    # Download specific JSON
                    json_str = json.dumps(result_obj, indent=2)
                    st.download_button(
                        label="📥 Download JSON Results",
                        data=json_str,
                        file_name=f"{row['analysis_id']}_result.json",
                        mime="application/json",
                        key=f"dl_btn_{row['analysis_id']}"
                    )
                    
                    st.markdown("---")
                    # Delete this analysis run
                    if st.button("🗑️ Delete Analysis Run", key=f"del_btn_{row['analysis_id']}"):
                        user_analysis_engine.delete_analysis(row["analysis_id"])
                        st.success("Run deleted.")
                        st.rerun()
                        
            st.markdown("<br/>", unsafe_allow_html=True)
            
        # Session deletion option in sidebar danger zone
        st.sidebar.markdown("---")
        st.sidebar.subheader("Danger Zone")
        confirm_del = st.sidebar.checkbox("Confirm deletion")
        if st.sidebar.button("Delete Session Entirely"):
            if confirm_del:
                user_analysis_engine.delete_session(selected_session_id)
                st.success("Session deleted successfully.")
                st.rerun()
            else:
                st.sidebar.error("Please check 'Confirm deletion' checkbox first.")
