#!/usr/bin/env python3
"""
run_project.py — Master orchestrator to run the BioIntelligence Platform Dockerless.
This script checks the local environment, initializes the SQLite database,
runs the sequential pipelines, and launches the Streamlit dashboard.
"""

import os
import sys
import subprocess
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("run_project")

def run_command(cmd, shell=True, check=True, text=True):
    """Run a system command and print output."""
    try:
        subprocess.run(cmd, shell=shell, check=check, text=text)
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Command failed: {cmd}")
        raise e

def main():
    log.info("=============================================================")
    log.info("  BioIntelligence Platform — Master Runner (Dockerless)       ")
    log.info("=============================================================")

    # 1. Check Python Dependencies
    log.info("Checking Python package dependencies...")
    try:
        import streamlit
        import pandas
        import plotly
        import sklearn
        import statsmodels
        import patsy
        log.info("✔ All required Python packages are installed.")
    except ImportError as e:
        missing_pkg = str(e).split("'")[-2] if "'" in str(e) else str(e)
        log.warning(f"Missing dependency: {missing_pkg}")
        log.info("Installing dependencies via pip...")
        run_command("pip install streamlit pandas plotly scikit-learn statsmodels patsy")
        log.info("✔ Dependencies installed successfully.")

    project_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Initialize SQLite Database
    log.info("Initializing SQLite database (biointel.db)...")
    db_init_path = os.path.join(project_dir, "db", "init_sqlite.py")
    if os.path.exists(db_init_path):
        run_command(f"python {db_init_path}")
    else:
        log.error(f"Initialization script not found at {db_init_path}!")
        sys.exit(1)

    # 3. Rerun Sequential Pipelines
    log.info("Running DNA + MRI data processing pipelines (Genetics QC, PCA, FastSurfer MRI, DNA, LLM)...")
    pipelines_path = os.path.join(project_dir, "run_pipelines.py")
    if os.path.exists(pipelines_path):
        run_command(f"python {pipelines_path}")
    else:
        log.error(f"Pipeline runner script not found at {pipelines_path}!")
        sys.exit(1)

    # 4. Launch Streamlit Visualization Dashboard
    log.info("=============================================================")
    log.info("🎉 Multi-modal analysis completed and SQLite seeded!")
    log.info("Launching the Streamlit dashboard in your web browser...")
    log.info("=============================================================")
    
    app_path = os.path.join(project_dir, "app.py")
    try:
        # Launch streamlit in the background or foreground
        # On Windows, we use start to open it in a new window or just let it run in foreground
        run_command(f"streamlit run {app_path}")
    except KeyboardInterrupt:
        log.info("\nStreamlit server stopped by user.")
    except Exception as e:
        log.error(f"Failed to launch Streamlit dashboard: {e}")

if __name__ == "__main__":
    main()
