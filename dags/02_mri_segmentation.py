# =============================================================================
# 02_mri_segmentation.py
# Airflow DAG: mri_segmentation_pipeline
#
# Orchestrates:
#   1. Verify subjects with genetics data exist
#   2. Download IXI T1 dataset
#   3. Download OASIS-3 demo subset
#   4. Run FastSurfer segmentation via Docker
#   5. Parse FastSurfer aseg.stats output
#   6. Load morphometry → brain_morphometry + update subjects.has_mri
#   7. Compute PC × brain-region Pearson correlations + FDR correction
#   8. Mark pipeline run complete
#
# Pre-requisites:
#   Docker accessible from airflow worker, deepmi/fastsurfer:cpu image pulled
#   Python: scipy, statsmodels, numpy, pandas
# =============================================================================

from __future__ import annotations

import glob
import logging
import os
import re
import tarfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator

import dag_utils

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DAG_ID = "mri_segmentation_pipeline"
DAG_OWNER = "biointel"

MRI_ROOT = Path("/opt/biointel/data/mri")
IXI_DIR = MRI_ROOT / "ixi"
OASIS_DIR = MRI_ROOT / "oasis3_demo"
FASTSURFER_OUTPUT_DIR = MRI_ROOT / "fastsurfer_output"

IXI_URL = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar"
IXI_TAR = str(IXI_DIR / "IXI-T1.tar")

OASIS_DEMO_URL = "https://www.oasis-brains.org/files/OASIS-3_demo.zip"
OASIS_ZIP = str(OASIS_DIR / "OASIS-3_demo.zip")

FASTSURFER_IMAGE = "deepmi/fastsurfer:cpu"

FDR_ALPHA = 0.05
MAX_PCS = 20            # correlate PC1-PC20 (most informative)

default_args: Dict[str, Any] = {
    "owner": DAG_OWNER,
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_aseg_stats(stats_file: str) -> List[Dict[str, Any]]:
    """
    Parse a FreeSurfer/FastSurfer aseg.stats file and return a list of
    region dicts with keys: region, volume_mm3, laterality.

    Format of relevant data lines (column-delimited, space-separated):
    # ColHeaders  Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange
    e.g.:
       1   4  8390  8390.4  Left-Lateral-Ventricle   ...
    """
    regions: List[Dict[str, Any]] = []
    with open(stats_file, "r") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                struct_name = parts[4]
                volume_mm3 = float(parts[3])
                # Determine laterality from structure name
                name_lower = struct_name.lower()
                if name_lower.startswith("left-") or name_lower.startswith("lh."):
                    lat = "L"
                elif name_lower.startswith("right-") or name_lower.startswith("rh."):
                    lat = "R"
                else:
                    lat = "B"
                regions.append({
                    "region": struct_name,
                    "volume_mm3": volume_mm3,
                    "thickness_mm": None,
                    "surface_area_mm2": None,
                    "laterality": lat,
                })
            except (ValueError, IndexError):
                continue
    return regions


def _parse_aparc_stats(stats_file: str, hemi: str) -> List[Dict[str, Any]]:
    """
    Parse a FreeSurfer/FastSurfer ?h.aparc.stats cortical parcellation file.
    Extracts thickness and surface area per cortical parcel.

    hemi: 'lh' or 'rh'
    """
    lat = "L" if hemi == "lh" else "R"
    regions: List[Dict[str, Any]] = []
    # ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd
    in_data = False
    with open(stats_file, "r") as fh:
        for line in fh:
            if line.startswith("# ColHeaders"):
                in_data = True
                continue
            if not in_data or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                struct_name = f"{hemi}.{parts[0]}"
                surface_area = float(parts[2])
                thickness = float(parts[4]) if len(parts) > 4 else None
                regions.append({
                    "region": struct_name,
                    "volume_mm3": None,
                    "thickness_mm": thickness,
                    "surface_area_mm2": surface_area,
                    "laterality": lat,
                })
            except (ValueError, IndexError):
                continue
    return regions


# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------


def _check_subjects_in_db(**context: Any) -> None:
    """
    Verify that subjects with has_genetics=True exist.
    Raises an error if none found (MRI pipeline depends on genetics pipeline).
    """
    conn = dag_utils.get_pg_conn()
    try:
        result = dag_utils.fetch_one(
            conn,
            "SELECT COUNT(*) AS n FROM subjects WHERE has_genetics = TRUE",
        )
        n = result["n"] if result else 0
        log.info("Subjects with genetics data: %d", n)
        if n == 0:
            raise ValueError(
                "No subjects with has_genetics=TRUE found. "
                "Run genetics_qc_pca_pipeline first."
            )
    finally:
        conn.close()

    run_id = dag_utils.log_pipeline_start(DAG_ID, config={"step": "mri_segmentation"})
    dag_utils.push_run_id(context, run_id)
    log.info("Started pipeline run_id=%s", run_id)


def _download_ixi_dataset(**context: Any) -> None:
    """
    Download and extract IXI-T1.tar dataset to /opt/biointel/data/mri/ixi/.
    Skips if the archive already exists.
    """
    import urllib.request

    IXI_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = Path(IXI_TAR)

    if tar_path.exists():
        log.info("IXI tar already exists at %s — skipping download.", IXI_TAR)
    else:
        log.info("Downloading IXI-T1.tar from %s …", IXI_URL)
        try:
            urllib.request.urlretrieve(
                IXI_URL,
                IXI_TAR,
                reporthook=lambda b, bs, total: log.info(
                    "  IXI download: %.1f MB / %.1f MB",
                    b * bs / 1e6,
                    total / 1e6 if total > 0 else 0,
                )
                if b % 500 == 0
                else None,
            )
        except Exception as exc:
            raise RuntimeError(f"IXI download failed: {exc}") from exc
        log.info("Download complete: %s", IXI_TAR)

    # Extract if not already extracted (look for at least one .nii.gz)
    niftis = list(IXI_DIR.glob("*.nii.gz"))
    if niftis:
        log.info("IXI NIfTI files already extracted (%d found) — skipping extraction.", len(niftis))
    else:
        log.info("Extracting %s …", IXI_TAR)
        with tarfile.open(IXI_TAR, "r") as tf:
            tf.extractall(str(IXI_DIR))
        niftis = list(IXI_DIR.glob("*.nii.gz"))
        log.info("Extracted %d NIfTI files to %s", len(niftis), IXI_DIR)

    context["ti"].xcom_push(key="n_ixi_subjects", value=len(niftis))


def _download_oasis3_sample(**context: Any) -> None:
    """
    Download the OASIS-3 10-subject public demo set.
    Skips if the zip already exists.
    """
    import urllib.request

    OASIS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = Path(OASIS_ZIP)

    if zip_path.exists():
        log.info("OASIS-3 demo zip already exists at %s — skipping download.", OASIS_ZIP)
    else:
        log.info("Downloading OASIS-3 demo from %s …", OASIS_DEMO_URL)
        try:
            urllib.request.urlretrieve(OASIS_DEMO_URL, OASIS_ZIP)
        except Exception as exc:
            log.warning(
                "OASIS-3 demo download failed (may require registration): %s. "
                "Continuing without OASIS data.",
                exc,
            )
            context["ti"].xcom_push(key="n_oasis_subjects", value=0)
            return
        log.info("Download complete: %s", OASIS_ZIP)

    # Extract
    oasis_subjects_dir = OASIS_DIR / "subjects"
    if oasis_subjects_dir.exists():
        log.info("OASIS-3 already extracted — skipping.")
    else:
        log.info("Extracting OASIS-3 demo zip …")
        with zipfile.ZipFile(OASIS_ZIP, "r") as zf:
            zf.extractall(str(OASIS_DIR))
        log.info("OASIS-3 demo extracted to %s", OASIS_DIR)

    niftis = list(OASIS_DIR.rglob("*.nii.gz")) + list(OASIS_DIR.rglob("*.nii"))
    log.info("OASIS-3 NIfTI files found: %d", len(niftis))
    context["ti"].xcom_push(key="n_oasis_subjects", value=len(niftis))


def _parse_fastsurfer_output(**context: Any) -> None:
    """
    Walk FastSurfer output directories, parse aseg.stats and aparc.stats files,
    accumulate morphometry data, and push to XCom.
    """
    FASTSURFER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_dirs = [
        d for d in FASTSURFER_OUTPUT_DIR.iterdir()
        if d.is_dir()
    ]
    log.info("Found %d FastSurfer output directories", len(output_dirs))

    all_records: List[Dict[str, Any]] = []

    for subj_dir in output_dirs:
        subject_id = subj_dir.name
        stats_dir = subj_dir / "stats"
        if not stats_dir.exists():
            log.warning("No stats dir for subject %s — skipping", subject_id)
            continue

        # aseg.stats — subcortical volumes
        aseg_file = stats_dir / "aseg.stats"
        if aseg_file.exists():
            try:
                regions = _parse_aseg_stats(str(aseg_file))
                for r in regions:
                    r["subject_id"] = subject_id
                all_records.extend(regions)
            except Exception as exc:
                log.warning("Failed to parse aseg.stats for %s: %s", subject_id, exc)

        # Cortical parcellations
        for hemi in ("lh", "rh"):
            aparc_file = stats_dir / f"{hemi}.aparc.stats"
            if aparc_file.exists():
                try:
                    regions = _parse_aparc_stats(str(aparc_file), hemi)
                    for r in regions:
                        r["subject_id"] = subject_id
                    all_records.extend(regions)
                except Exception as exc:
                    log.warning("Failed to parse %s.aparc.stats for %s: %s", hemi, subject_id, exc)

    log.info("Parsed %d total region measurements across %d subjects",
             len(all_records), len(output_dirs))

    # Push to XCom (list of dicts)
    context["ti"].xcom_push(key="morphometry_records", value=all_records)
    context["ti"].xcom_push(key="n_subjects_mri", value=len(output_dirs))


def _load_morphometry_to_postgres(**context: Any) -> None:
    """
    Load parsed morphometry records from XCom → brain_morphometry table.
    Update subjects.has_mri = TRUE for processed subjects.
    """
    records: List[Dict[str, Any]] = context["ti"].xcom_pull(key="morphometry_records") or []
    if not records:
        log.warning("No morphometry records to load — is FastSurfer output present?")
        return

    rows: List[tuple] = []
    subject_ids_seen = set()
    for rec in records:
        subject_id = rec.get("subject_id")
        if not subject_id:
            continue
        subject_ids_seen.add(subject_id)
        rows.append((
            subject_id,
            "FastSurfer",
            rec.get("region", "unknown"),
            rec.get("volume_mm3"),
            rec.get("thickness_mm"),
            rec.get("surface_area_mm2"),
            rec.get("laterality", "B"),
        ))

    conn = dag_utils.get_pg_conn()
    try:
        # Ensure subjects exist (insert placeholder rows if not already there)
        subject_rows = [
            (sid, "MRI_ONLY", "U", None, None, False, True)
            for sid in subject_ids_seen
        ]
        dag_utils.bulk_insert(
            conn,
            "subjects",
            ["subject_id", "dataset_source", "sex", "age_at_scan",
             "ethnicity_label", "has_genetics", "has_mri"],
            subject_rows,
            on_conflict="(subject_id) DO UPDATE SET has_mri = TRUE",
        )

        # Insert morphometry data
        dag_utils.bulk_insert(
            conn,
            "brain_morphometry",
            ["subject_id", "segmentation_tool", "region",
             "volume_mm3", "thickness_mm", "surface_area_mm2", "laterality"],
            rows,
            on_conflict="DO NOTHING",
        )

        # Mark subjects as having MRI
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE subjects SET has_mri = TRUE WHERE subject_id = ANY(%s)",
                (list(subject_ids_seen),),
            )

        conn.commit()
        log.info(
            "Loaded %d morphometry rows for %d subjects",
            len(rows), len(subject_ids_seen),
        )
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _compute_pca_brain_correlations(**context: Any) -> None:
    """
    For each of PC1-PC20 and each brain region:
      1. Pull data from ancestry_pca and brain_morphometry
      2. Compute Pearson r
      3. Apply Benjamini-Hochberg FDR correction across all tests
      4. Insert into pca_brain_correlations
    """
    conn = dag_utils.get_pg_conn()
    try:
        # Fetch PCA scores
        pca_df_rows = dag_utils.fetch_all(
            conn,
            f"SELECT subject_id, {', '.join(f'pc{i}' for i in range(1, MAX_PCS + 1))} "
            f"FROM ancestry_pca",
        )
        if not pca_df_rows:
            log.warning("No PCA data found — skipping correlations.")
            return
        pca_df = pd.DataFrame(pca_df_rows).set_index("subject_id")

        # Fetch morphometry volumes
        morph_df_rows = dag_utils.fetch_all(
            conn,
            "SELECT subject_id, region, volume_mm3 FROM brain_morphometry "
            "WHERE volume_mm3 IS NOT NULL",
        )
        if not morph_df_rows:
            log.warning("No morphometry data found — skipping correlations.")
            return
        morph_df = (
            pd.DataFrame(morph_df_rows)
            .pivot_table(index="subject_id", columns="region", values="volume_mm3", aggfunc="mean")
        )
    finally:
        conn.close()

    # Align subjects
    common_subjects = pca_df.index.intersection(morph_df.index)
    if len(common_subjects) < 5:
        log.warning(
            "Only %d subjects have both PCA and morphometry data — "
            "correlations will be unreliable. Skipping.",
            len(common_subjects),
        )
        return

    pca_aligned = pca_df.loc[common_subjects]
    morph_aligned = morph_df.loc[common_subjects]
    regions = morph_aligned.columns.tolist()
    pc_ids = list(range(1, MAX_PCS + 1))
    n_subjects = len(common_subjects)

    log.info(
        "Computing correlations: %d PCs × %d regions = %d tests (%d subjects)",
        MAX_PCS, len(regions), MAX_PCS * len(regions), n_subjects,
    )

    # Collect raw results
    results: List[Tuple[int, str, float, float]] = []  # pc_id, region, r, p
    for pc_id in pc_ids:
        pc_col = f"pc{pc_id}"
        pc_vals = pca_aligned[pc_col].values
        for region in regions:
            region_vals = morph_aligned[region].values
            # Mask NaN
            mask = ~(np.isnan(pc_vals) | np.isnan(region_vals))
            if mask.sum() < 5:
                continue
            try:
                r, p = scipy_stats.pearsonr(pc_vals[mask], region_vals[mask])
            except Exception:
                continue
            results.append((pc_id, region, float(r), float(p)))

    if not results:
        log.warning("No valid correlation results produced.")
        return

    # FDR correction across all tests
    p_values = [r[3] for r in results]
    reject, p_adjusted, _, _ = multipletests(p_values, alpha=FDR_ALPHA, method="fdr_bh")

    # Build rows for DB
    corr_rows: List[tuple] = []
    for (pc_id, region, r, p), p_fdr, sig in zip(results, p_adjusted, reject):
        corr_rows.append((
            pc_id,
            region,
            r,
            p,
            n_subjects,
            float(p_fdr),
            bool(sig),
        ))

    n_significant = sum(1 for _, sig in zip(results, reject) if sig)
    log.info(
        "Correlation tests: %d total, %d significant at FDR < %.2f",
        len(results), n_significant, FDR_ALPHA,
    )

    # Insert into DB
    conn = dag_utils.get_pg_conn()
    try:
        # Clear old results
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE pca_brain_correlations")
        dag_utils.bulk_insert(
            conn,
            "pca_brain_correlations",
            ["pc_id", "brain_region", "pearson_r", "p_value",
             "n_subjects", "fdr_corrected_p", "is_significant"],
            corr_rows,
            on_conflict="DO NOTHING",
        )
        conn.commit()
        log.info("Inserted %d pca_brain_correlations rows", len(corr_rows))
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _update_pipeline_run(**context: Any) -> None:
    """Mark the pipeline run complete."""
    run_id = dag_utils.pull_run_id(context)
    if not run_id:
        log.warning("No run_id in XCom — skipping pipeline_runs update")
        return
    n_subjects = context["ti"].xcom_pull(key="n_subjects_mri") or 0
    dag_utils.log_pipeline_complete(run_id, n_subjects=n_subjects)
    log.info("Pipeline run_id=%s marked complete (n_subjects=%d)", run_id, n_subjects)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id=DAG_ID,
    description="IXI/OASIS3 download → FastSurfer segmentation → PC-brain correlations",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["biointel", "mri", "fastsurfer", "morphometry"],
) as dag:

    # ------------------------------------------------------------------
    # Task 1: Verify genetics subjects exist
    # ------------------------------------------------------------------
    check_subjects_in_db = PythonOperator(
        task_id="check_subjects_in_db",
        python_callable=_check_subjects_in_db,
    )

    # ------------------------------------------------------------------
    # Task 2: Download IXI dataset
    # ------------------------------------------------------------------
    download_ixi_dataset = PythonOperator(
        task_id="download_ixi_dataset",
        python_callable=_download_ixi_dataset,
        execution_timeout=timedelta(hours=8),
    )

    # ------------------------------------------------------------------
    # Task 3: Download OASIS-3 demo subset
    # ------------------------------------------------------------------
    download_oasis3_sample = PythonOperator(
        task_id="download_oasis3_sample",
        python_callable=_download_oasis3_sample,
        execution_timeout=timedelta(hours=2),
    )

    # ------------------------------------------------------------------
    # Task 4: FastSurfer segmentation via Docker
    #
    # The DockerOperator runs one container per batch of subjects.
    # We mount the MRI data directory and the output directory.
    # The command iterates over all T1 NIfTI files in the input dir.
    # ------------------------------------------------------------------
    run_fastsurfer_segmentation = DockerOperator(
        task_id="run_fastsurfer_segmentation",
        image=FASTSURFER_IMAGE,
        api_version="auto",
        auto_remove="force",
        # Mount host paths into the container
        mounts=[
            # Input MRI data (read-only)
            {
                "source": str(MRI_ROOT),
                "target": "/data",
                "type": "bind",
                "read_only": False,
            },
            # Output directory
            {
                "source": str(FASTSURFER_OUTPUT_DIR),
                "target": "/output",
                "type": "bind",
                "read_only": False,
            },
        ],
        # FastSurfer batch command: iterate over all T1 nii.gz files
        # The loop discovers subjects from /data, runs FastSurfer for each,
        # and writes results to /output/<subject_id>/
        command="""
            bash -c '
            set -euo pipefail
            mkdir -p /output
            shopt -s nullglob
            T1_FILES=(/data/ixi/*.nii.gz /data/oasis3_demo/**/*.nii.gz /data/oasis3_demo/**/*.nii)
            echo "Discovered ${#T1_FILES[@]} T1 files for segmentation"
            for T1 in "${T1_FILES[@]}"; do
                SUBJ=$(basename "$T1" .nii.gz)
                SUBJ=$(basename "$SUBJ" .nii)
                echo "=== Processing subject: $SUBJ ==="
                if [ -d "/output/$SUBJ/stats" ]; then
                    echo "  Already processed — skipping."
                    continue
                fi
                /fastsurfer/run_fastsurfer.sh \
                    --t1 "$T1" \
                    --sid "$SUBJ" \
                    --sd /output \
                    --seg_only \
                    --no_fs_T1 \
                    --threads 4 \
                    --batch_size 4 \
                    || echo "WARNING: FastSurfer failed for $SUBJ — continuing."
            done
            echo "=== FastSurfer batch complete ==="
            '
        """,
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        # CPU mode — no GPU flags needed for cpu image
        device_requests=[],
        # Allow sufficient time for segmentation batch
        execution_timeout=timedelta(hours=24),
        # Environment passed to container
        environment={
            "SUBJECTS_DIR": "/output",
        },
    )

    # ------------------------------------------------------------------
    # Task 5: Parse FastSurfer stats files
    # ------------------------------------------------------------------
    parse_fastsurfer_output = PythonOperator(
        task_id="parse_fastsurfer_output",
        python_callable=_parse_fastsurfer_output,
    )

    # ------------------------------------------------------------------
    # Task 6: Load morphometry to PostgreSQL
    # ------------------------------------------------------------------
    load_morphometry_to_postgres = PythonOperator(
        task_id="load_morphometry_to_postgres",
        python_callable=_load_morphometry_to_postgres,
    )

    # ------------------------------------------------------------------
    # Task 7: Compute PC × brain-region correlations
    # ------------------------------------------------------------------
    compute_pca_brain_correlations = PythonOperator(
        task_id="compute_pca_brain_correlations",
        python_callable=_compute_pca_brain_correlations,
    )

    # ------------------------------------------------------------------
    # Task 8: Mark pipeline run complete
    # ------------------------------------------------------------------
    update_pipeline_run = PythonOperator(
        task_id="update_pipeline_run",
        python_callable=_update_pipeline_run,
        trigger_rule="all_done",
    )

    # ------------------------------------------------------------------
    # Task dependencies
    # ------------------------------------------------------------------
    check_subjects_in_db >> [download_ixi_dataset, download_oasis3_sample]
    [download_ixi_dataset, download_oasis3_sample] >> run_fastsurfer_segmentation
    run_fastsurfer_segmentation >> parse_fastsurfer_output
    parse_fastsurfer_output >> load_morphometry_to_postgres
    load_morphometry_to_postgres >> compute_pca_brain_correlations
    compute_pca_brain_correlations >> update_pipeline_run
