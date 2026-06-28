# =============================================================================
# 01_genetics_qc_pca.py
# Airflow DAG: genetics_qc_pca_pipeline
#
# Orchestrates:
#   1. Data availability check
#   2. 1000 Genomes chr22 download
#   3. GATK/bcftools variant QC
#   4. PLINK2 LD pruning
#   5. PLINK2 PCA (40 PCs)
#   6. Load PCA results -> ancestry_pca table
#   7. Load filtered variants -> variants + subject_variants tables
#   8. Mark pipeline run complete
#
# Pre-requisites (in airflow worker image):
#   bcftools, plink2, cyvcf2 python package
# =============================================================================

from __future__ import annotations

import gzip
import logging
import os
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

import dag_utils

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DAG_ID = "genetics_qc_pca_pipeline"
DAG_OWNER = "biointel"
DATA_DIR = Path("/opt/biointel/data/1kg")
MODELS_DIR = Path("/opt/biointel/models")

# 1000 Genomes chr22 — phase 3 (GRCh37)
VCF_BASE = (
    "ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
)
VCF_URL = f"http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/{VCF_BASE}"
TBI_URL = f"{VCF_URL}.tbi"

VCF_RAW = str(DATA_DIR / VCF_BASE)
VCF_TBI = f"{VCF_RAW}.tbi"
VCF_FILTERED = str(DATA_DIR / "chr22_filtered.vcf.gz")
PLINK_PREFIX = str(DATA_DIR / "chr22_pruned")
PLINK_LD_PREFIX = str(DATA_DIR / "chr22_ld_pruned")
PCA_PREFIX = str(DATA_DIR / "chr22_pca")
DATASET_SOURCE = "1000G"

default_args: Dict[str, Any] = {
    "owner": DAG_OWNER,
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------


def _check_data_availability(**context: Any) -> None:
    """
    Query pipeline_runs for the last successful genetics run.
    Log the result and push the new run_id to XCom.
    """
    conn = dag_utils.get_pg_conn()
    try:
        last_run = dag_utils.fetch_one(
            conn,
            """
            SELECT run_id, started_at, n_subjects_processed
            FROM   pipeline_runs
            WHERE  dag_id = %s
              AND  status = 'success'
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (DAG_ID,),
        )
        if last_run:
            log.info(
                "Last successful run: run_id=%s at %s (%d subjects processed)",
                last_run["run_id"],
                last_run["started_at"],
                last_run["n_subjects_processed"] or 0,
            )
        else:
            log.info("No previous successful run found for dag_id=%s — first run.", DAG_ID)

        # Check how many subjects we already have
        counts = dag_utils.fetch_one(
            conn,
            "SELECT COUNT(*) AS total FROM subjects WHERE dataset_source = %s",
            (DATASET_SOURCE,),
        )
        log.info("Existing subjects from %s: %d", DATASET_SOURCE, counts["total"] if counts else 0)
    finally:
        conn.close()

    # Start a new pipeline_runs record and push its id downstream
    run_id = dag_utils.log_pipeline_start(DAG_ID, config={"dataset": DATASET_SOURCE})
    dag_utils.push_run_id(context, run_id)
    log.info("Started pipeline run_id=%s", run_id)


def _load_pca_to_postgres(**context: Any) -> None:
    """
    Read PLINK2 .eigenvec file, upsert subjects, insert ancestry_pca rows.

    PLINK2 eigenvec format (space-delimited, no header by default, or with
    --out-pca-header):
        #FID IID PC1 PC2 ... PC40
    We treat IID as subject_id.
    """
    eigenvec_path = Path(f"{PCA_PREFIX}.eigenvec")
    if not eigenvec_path.exists():
        raise FileNotFoundError(f"PCA eigenvec file not found: {eigenvec_path}")

    pca_run_id = str(uuid.uuid4())

    # Count variants used (from .log file, best-effort)
    n_variants_used: Optional[int] = None
    log_path = Path(f"{PCA_PREFIX}.log")
    if log_path.exists():
        text = log_path.read_text(errors="replace")
        m = re.search(r"(\d+)\s+variant[s]?\s+remaining", text)
        if m:
            n_variants_used = int(m.group(1))
    log.info("PCA run_id=%s, n_variants_used=%s", pca_run_id, n_variants_used)

    # Parse eigenvec — header line starts with '#FID' or 'FID'
    subjects_rows: List[tuple] = []
    pca_rows: List[tuple] = []

    with open(eigenvec_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#FID") or line.startswith("FID"):
                continue
            parts = line.split()
            # FID IID PC1..PC40  (PLINK2 always outputs FID+IID even if FID='0')
            if len(parts) < 3:
                continue
            _fid, iid = parts[0], parts[1]
            pc_values = [float(v) for v in parts[2:]]
            # Pad to 40 if fewer components were computed
            while len(pc_values) < 40:
                pc_values.append(None)
            pc_values = pc_values[:40]

            subject_id = iid
            subjects_rows.append((subject_id, DATASET_SOURCE, "U", None, None, True, False))
            pca_rows.append(
                (subject_id, *pc_values, pca_run_id, n_variants_used)
            )

    if not subjects_rows:
        raise ValueError("No rows parsed from eigenvec file — check PLINK2 output.")

    log.info("Parsed %d subjects from eigenvec", len(subjects_rows))

    conn = dag_utils.get_pg_conn()
    try:
        # Upsert subjects (do not overwrite existing records with real data)
        dag_utils.bulk_insert(
            conn,
            "subjects",
            ["subject_id", "dataset_source", "sex", "age_at_scan", "ethnicity_label",
             "has_genetics", "has_mri"],
            subjects_rows,
            on_conflict="(subject_id) DO UPDATE SET has_genetics = TRUE, "
                        "dataset_source = EXCLUDED.dataset_source",
        )

        # Insert PCA rows (delete old rows for this dataset first to avoid stale data)
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM ancestry_pca WHERE subject_id IN "
                "(SELECT subject_id FROM subjects WHERE dataset_source = %s)",
                (DATASET_SOURCE,),
            )
            log.info("Deleted old ancestry_pca rows for dataset=%s", DATASET_SOURCE)

        pca_columns = (
            ["subject_id"]
            + [f"pc{i}" for i in range(1, 41)]
            + ["pca_run_id", "n_variants_used"]
        )
        dag_utils.bulk_insert(conn, "ancestry_pca", pca_columns, pca_rows)

        conn.commit()
        log.info("Inserted %d ancestry_pca rows", len(pca_rows))
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    # Stash count for final task
    context["ti"].xcom_push(key="n_subjects_pca", value=len(subjects_rows))


def _load_variants_to_postgres(**context: Any) -> None:
    """
    Parse the filtered VCF with cyvcf2 and bulk-insert into variants +
    subject_variants tables.

    For memory efficiency, we process the VCF in streaming fashion and
    commit in batches.
    """
    try:
        from cyvcf2 import VCF  # type: ignore
    except ImportError as exc:
        raise ImportError("cyvcf2 is required: pip install cyvcf2") from exc

    vcf_path = VCF_FILTERED
    if not os.path.exists(vcf_path):
        raise FileNotFoundError(f"Filtered VCF not found: {vcf_path}")

    BATCH_SIZE = 5000
    variant_batch: List[tuple] = []
    sv_batch: List[tuple] = []
    n_variants_inserted = 0
    n_sv_inserted = 0

    vcf = VCF(vcf_path)
    samples: List[str] = vcf.samples  # IIDs in VCF header

    conn = dag_utils.get_pg_conn()
    try:
        for record in vcf:
            chrom = str(record.CHROM)
            pos = int(record.POS)
            ref = str(record.REF)
            # Handle multi-allelic — iterate over each ALT
            for alt_idx, alt in enumerate(record.ALT):
                variant_id = f"{chrom}:{pos}:{ref}:{alt}"
                rsid = record.ID if record.ID else None
                # Attempt to get gene/consequence from INFO (not in 1KG VCF, but included for schema completeness)
                gene_symbol = None
                consequence = None
                gnomad_af = None
                clinvar_sig = None
                info = dict(record.INFO) if record.INFO else {}
                if "AF" in info:
                    af_val = info["AF"]
                    gnomad_af = float(af_val[alt_idx]) if hasattr(af_val, "__len__") else float(af_val)

                variant_batch.append(
                    (variant_id, chrom, pos, ref, alt, rsid,
                     gene_symbol, consequence, clinvar_sig, gnomad_af, DATASET_SOURCE)
                )

                # Genotypes per sample
                for s_idx, sample in enumerate(samples):
                    gt = record.genotypes[s_idx]  # list like [0, 1, True]
                    alleles = gt[:2]
                    gt_str = "/".join(str(a) if a is not None else "." for a in alleles)
                    gq_val = None
                    dp_val = None
                    try:
                        gq_arr = record.format("GQ")
                        if gq_arr is not None:
                            gq_val = int(gq_arr[s_idx][0])
                    except Exception:
                        pass
                    try:
                        dp_arr = record.format("DP")
                        if dp_arr is not None:
                            dp_val = int(dp_arr[s_idx][0])
                    except Exception:
                        pass

                    sv_batch.append((sample, variant_id, gt_str, gq_val, dp_val))

            # Flush batches
            if len(variant_batch) >= BATCH_SIZE:
                dag_utils.bulk_insert(
                    conn, "variants",
                    ["variant_id", "chrom", "pos", "ref", "alt", "rsid",
                     "gene_symbol", "consequence", "clinvar_sig", "gnomad_af", "dataset_source"],
                    variant_batch,
                    on_conflict="(variant_id) DO NOTHING",
                )
                n_variants_inserted += len(variant_batch)
                variant_batch.clear()

            if len(sv_batch) >= BATCH_SIZE * 10:
                dag_utils.bulk_insert(
                    conn, "subject_variants",
                    ["subject_id", "variant_id", "genotype", "gq", "dp"],
                    sv_batch,
                    on_conflict="(subject_id, variant_id) DO UPDATE SET "
                                "genotype = EXCLUDED.genotype, gq = EXCLUDED.gq, dp = EXCLUDED.dp",
                )
                n_sv_inserted += len(sv_batch)
                sv_batch.clear()
                conn.commit()

        vcf.close()

        # Final flush
        if variant_batch:
            dag_utils.bulk_insert(
                conn, "variants",
                ["variant_id", "chrom", "pos", "ref", "alt", "rsid",
                 "gene_symbol", "consequence", "clinvar_sig", "gnomad_af", "dataset_source"],
                variant_batch,
                on_conflict="(variant_id) DO NOTHING",
            )
            n_variants_inserted += len(variant_batch)

        if sv_batch:
            dag_utils.bulk_insert(
                conn, "subject_variants",
                ["subject_id", "variant_id", "genotype", "gq", "dp"],
                sv_batch,
                on_conflict="(subject_id, variant_id) DO UPDATE SET "
                            "genotype = EXCLUDED.genotype, gq = EXCLUDED.gq, dp = EXCLUDED.dp",
            )
            n_sv_inserted += len(sv_batch)

        conn.commit()
        log.info(
            "Variants inserted: %d unique variants, %d subject_variant rows",
            n_variants_inserted, n_sv_inserted,
        )
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    context["ti"].xcom_push(key="n_variants", value=n_variants_inserted)


def _update_pipeline_run(**context: Any) -> None:
    """Mark the pipeline_runs record as successfully completed."""
    run_id = dag_utils.pull_run_id(context)
    if not run_id:
        log.warning("No run_id in XCom — skipping pipeline_runs update")
        return
    n_subjects = context["ti"].xcom_pull(key="n_subjects_pca") or 0
    dag_utils.log_pipeline_complete(run_id, n_subjects=n_subjects)
    log.info("Pipeline run_id=%s marked complete (n_subjects=%d)", run_id, n_subjects)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id=DAG_ID,
    description="1000 Genomes chr22 QC → LD pruning → PCA → load to PostgreSQL",
    default_args=default_args,
    schedule=None,          # manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["biointel", "genetics", "pca", "qc"],
) as dag:

    # ------------------------------------------------------------------
    # Task 1: Check data availability
    # ------------------------------------------------------------------
    check_data_availability = PythonOperator(
        task_id="check_data_availability",
        python_callable=_check_data_availability,
    )

    # ------------------------------------------------------------------
    # Task 2: Download 1KG chr22 VCF + index
    # ------------------------------------------------------------------
    download_1kg_chr22 = BashOperator(
        task_id="download_1kg_chr22",
        bash_command=f"""
set -euo pipefail
mkdir -p {DATA_DIR}

VCF_RAW="{VCF_RAW}"
VCF_TBI="{VCF_TBI}"
VCF_URL="{VCF_URL}"
TBI_URL="{TBI_URL}"

if [ -f "$VCF_RAW" ]; then
    echo "VCF already exists at $VCF_RAW — skipping download."
else
    echo "Downloading VCF from $VCF_URL ..."
    wget --progress=dot:giga -O "$VCF_RAW" "$VCF_URL"
    echo "Download complete: $VCF_RAW"
fi

if [ -f "$VCF_TBI" ]; then
    echo "TBI index already exists — skipping."
else
    echo "Downloading TBI index from $TBI_URL ..."
    wget --progress=dot:giga -O "$VCF_TBI" "$TBI_URL"
    echo "Index download complete: $VCF_TBI"
fi
""",
        execution_timeout=timedelta(hours=6),
    )

    # ------------------------------------------------------------------
    # Task 3: GATK/bcftools variant QC → filtered VCF
    # ------------------------------------------------------------------
    run_gatk_variant_qc = BashOperator(
        task_id="run_gatk_variant_qc",
        bash_command=f"""
set -euo pipefail
VCF_RAW="{VCF_RAW}"
VCF_FILTERED="{VCF_FILTERED}"
STATS_FILE="{DATA_DIR}/chr22_filtered_stats.txt"

echo "=== bcftools filter: PASS variants only, biallelic SNPs ==="
bcftools view \\
    --apply-filters PASS \\
    --type snps \\
    --min-ac 1:minor \\
    --max-alleles 2 \\
    --min-alleles 2 \\
    --output-type z \\
    --output "$VCF_FILTERED" \\
    "$VCF_RAW"

echo "=== Indexing filtered VCF ==="
bcftools index --tbi "$VCF_FILTERED"

echo "=== bcftools stats on filtered VCF ==="
bcftools stats "$VCF_FILTERED" > "$STATS_FILE"

echo "=== Variant counts ==="
bcftools stats "$VCF_FILTERED" | grep "^SN" | head -20

echo "Filtered VCF written to $VCF_FILTERED"
""",
        execution_timeout=timedelta(hours=2),
    )

    # ------------------------------------------------------------------
    # Task 4: PLINK2 LD pruning
    # ------------------------------------------------------------------
    run_plink2_ld_prune = BashOperator(
        task_id="run_plink2_ld_prune",
        bash_command=f"""
set -euo pipefail
VCF_FILTERED="{VCF_FILTERED}"
PLINK_PREFIX="{PLINK_PREFIX}"
PLINK_LD_PREFIX="{PLINK_LD_PREFIX}"

echo "=== PLINK2: compute LD prune list ==="
plink2 \\
    --vcf "$VCF_FILTERED" \\
    --indep-pairwise 50 5 0.2 \\
    --out "$PLINK_PREFIX" \\
    --threads 4

echo "=== PLINK2: extract pruned variants ==="
plink2 \\
    --vcf "$VCF_FILTERED" \\
    --extract "${{PLINK_PREFIX}}.prune.in" \\
    --make-pgen \\
    --out "$PLINK_LD_PREFIX" \\
    --threads 4

PRUNE_IN=$(wc -l < "${{PLINK_PREFIX}}.prune.in")
echo "LD-pruned variant count: $PRUNE_IN"
""",
        execution_timeout=timedelta(hours=2),
    )

    # ------------------------------------------------------------------
    # Task 5: PLINK2 PCA (40 PCs)
    # ------------------------------------------------------------------
    compute_pca = BashOperator(
        task_id="compute_pca",
        bash_command=f"""
set -euo pipefail
PLINK_LD_PREFIX="{PLINK_LD_PREFIX}"
PCA_PREFIX="{PCA_PREFIX}"

echo "=== PLINK2: PCA with 40 components (approx) ==="
plink2 \\
    --pfile "$PLINK_LD_PREFIX" \\
    --pca 40 approx \\
    --out "$PCA_PREFIX" \\
    --threads 4

echo "=== PCA complete: eigenvalues ==="
head -5 "${{PCA_PREFIX}}.eigenval"

echo "=== First 3 rows of eigenvec ==="
head -4 "${{PCA_PREFIX}}.eigenvec"
""",
        execution_timeout=timedelta(hours=3),
    )

    # ------------------------------------------------------------------
    # Task 6: Load PCA → ancestry_pca + subjects
    # ------------------------------------------------------------------
    load_pca_to_postgres = PythonOperator(
        task_id="load_pca_to_postgres",
        python_callable=_load_pca_to_postgres,
    )

    # ------------------------------------------------------------------
    # Task 7: Load filtered variants → variants + subject_variants
    # ------------------------------------------------------------------
    load_variants_to_postgres = PythonOperator(
        task_id="load_variants_to_postgres",
        python_callable=_load_variants_to_postgres,
    )

    # ------------------------------------------------------------------
    # Task 8: Mark pipeline run complete
    # ------------------------------------------------------------------
    update_pipeline_run = PythonOperator(
        task_id="update_pipeline_run",
        python_callable=_update_pipeline_run,
        trigger_rule="all_done",  # run even if upstream partially failed
    )

    # ------------------------------------------------------------------
    # Task dependencies
    # ------------------------------------------------------------------
    (
        check_data_availability
        >> download_1kg_chr22
        >> run_gatk_variant_qc
        >> run_plink2_ld_prune
        >> compute_pca
        >> load_pca_to_postgres
        >> load_variants_to_postgres
        >> update_pipeline_run
    )
