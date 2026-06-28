"""
genetics_qc.py — Genetics QC, PCA loading, and variant cataloging
Handles: VCF parsing (cyvcf2), PCA eigenvec loading, PostgreSQL persistence.
"""
import os
import re
import logging
import subprocess
import tempfile
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

try:
    import cyvcf2
except ImportError:  # pragma: no cover
    cyvcf2 = None  # type: ignore

from db_utils import get_pg_conn, bulk_insert, bulk_upsert

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get("BIOINTEL_DATA_DIR", "/opt/biointel/data")
N_PCS = 40                         # number of principal components stored


# ===========================================================================
# 1. PCA eigenvec loader
# ===========================================================================

def load_pca_eigenvec(
    eigenvec_path: str,
    run_id: str,
    n_variants: int,
) -> int:
    """Load a PLINK2 ``.eigenvec`` file into the database.

    The file is expected to have columns::

        #FID  IID  PC1  PC2  ... PC40

    For each sample:

    * Upserts a row in ``subjects`` (subject_id = IID).
    * Bulk-inserts a row in ``ancestry_pca`` containing all 40 PC scores.

    Parameters
    ----------
    eigenvec_path:
        Absolute path to the ``.eigenvec`` file.
    run_id:
        Identifier for this PCA run (stored in ``ancestry_pca.run_id``).
    n_variants:
        Number of variants used in this PCA run.

    Returns
    -------
    int
        Number of subjects successfully loaded.
    """
    if not os.path.isfile(eigenvec_path):
        raise FileNotFoundError(f"eigenvec file not found: {eigenvec_path}")

    subject_rows: List[Tuple] = []   # for subjects upsert
    pca_rows: List[Tuple] = []       # for ancestry_pca insert

    logger.info("Reading eigenvec file: %s", eigenvec_path)

    with open(eigenvec_path, "r") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#FID"):
                # Skip header line (PLINK2 emits #FID header)
                continue

            parts = line.split()
            if len(parts) < 3:
                logger.warning("Line %d: too few columns, skipping: %s", lineno, line)
                continue

            fid = parts[0]
            iid = parts[1]
            pc_values = parts[2:]

            # Pad or truncate to exactly N_PCS
            pc_floats: List[Optional[float]] = []
            for i in range(N_PCS):
                if i < len(pc_values):
                    try:
                        pc_floats.append(float(pc_values[i]))
                    except ValueError:
                        pc_floats.append(None)
                else:
                    pc_floats.append(None)

            # subjects upsert: (subject_id, dataset_source, has_genetics)
            subject_rows.append((iid, "1000G", True))

            # ancestry_pca insert row
            # columns: subject_id, run_id, n_variants, pc1 .. pc40
            pca_row = (iid, run_id, n_variants) + tuple(pc_floats)
            pca_rows.append(pca_row)

    if not pca_rows:
        logger.warning("No PCA rows parsed from %s", eigenvec_path)
        return 0

    logger.info("Parsed %d subjects from eigenvec.", len(pca_rows))

    with get_pg_conn() as conn:
        # Upsert subjects
        bulk_upsert(
            conn=conn,
            table="subjects",
            columns=["subject_id", "dataset_source", "has_genetics"],
            conflict_cols=["subject_id"],
            rows=subject_rows,
        )

        # Bulk-insert ancestry_pca
        pc_cols = [f"pc{i}" for i in range(1, N_PCS + 1)]
        ancestry_cols = ["subject_id", "run_id", "n_variants"] + pc_cols
        bulk_insert(
            conn=conn,
            table="ancestry_pca",
            columns=ancestry_cols,
            rows=pca_rows,
            on_conflict="(subject_id, run_id) DO NOTHING",
        )

    n_loaded = len(pca_rows)
    logger.info("load_pca_eigenvec: loaded %d subjects (run_id=%s)", n_loaded, run_id)
    return n_loaded


# ===========================================================================
# 2. Variant loader from VCF
# ===========================================================================

def load_variants_from_vcf(
    vcf_path: str,
    max_variants: int = 500_000,
) -> int:
    """Parse a VCF file and bulk-insert variants into the ``variants`` table.

    For each ALT allele a separate row is created.  ``variant_id`` is built as
    ``chrom:pos:ref:alt``.  If INFO/CSQ is present, the first gene token from
    the VEP CSQ field is extracted.

    Parameters
    ----------
    vcf_path:
        Path to the VCF (or VCF.gz) file.
    max_variants:
        Stop after this many variants (useful for testing).

    Returns
    -------
    int
        Number of variant rows loaded.
    """
    if cyvcf2 is None:
        raise ImportError("cyvcf2 is required for VCF parsing (pip install cyvcf2)")

    if not os.path.isfile(vcf_path):
        raise FileNotFoundError(f"VCF not found: {vcf_path}")

    logger.info("Parsing VCF for variants: %s (max=%d)", vcf_path, max_variants)
    vcf = cyvcf2.VCF(vcf_path)

    # Detect CSQ field header to find gene position
    csq_gene_idx: Optional[int] = None
    for header_line in vcf.header_iter():
        info = header_line.info(extra=True)
        if info.get("ID") == "CSQ":
            desc = info.get("Description", "")
            # Format: Allele|Consequence|IMPACT|SYMBOL|...
            match = re.search(r"Format: ([^\"]+)", desc)
            if match:
                csq_fields = match.group(1).split("|")
                try:
                    csq_gene_idx = csq_fields.index("SYMBOL")
                    logger.debug("CSQ SYMBOL field at index %d", csq_gene_idx)
                except ValueError:
                    pass
            break

    rows: List[Tuple] = []
    n_parsed = 0

    for variant in vcf:
        if n_parsed >= max_variants:
            break

        chrom = variant.CHROM
        pos = variant.POS
        ref = variant.REF

        alts = variant.ALT or []
        if not alts:
            alts = ["."]

        # rsID
        rs_id = variant.ID or "."

        # Gene from CSQ INFO
        gene: Optional[str] = None
        if csq_gene_idx is not None:
            csq_raw = variant.INFO.get("CSQ")
            if csq_raw:
                # CSQ can be multiple comma-sep annotations
                first_csq = str(csq_raw).split(",")[0]
                parts = first_csq.split("|")
                if csq_gene_idx < len(parts):
                    gene = parts[csq_gene_idx] or None

        for alt in alts:
            variant_id = f"{chrom}:{pos}:{ref}:{alt}"
            rows.append((variant_id, chrom, pos, ref, alt, rs_id, gene))
            n_parsed += 1

        if n_parsed % 100_000 == 0:
            logger.info("  ...parsed %d variants so far", n_parsed)

    vcf.close()

    if not rows:
        logger.warning("No variants extracted from %s", vcf_path)
        return 0

    columns = ["variant_id", "chrom", "pos", "ref", "alt", "rs_id", "gene"]
    with get_pg_conn() as conn:
        n_inserted = bulk_insert(
            conn=conn,
            table="variants",
            columns=columns,
            rows=rows,
            on_conflict="(variant_id) DO NOTHING",
        )

    logger.info("load_variants_from_vcf: %d rows inserted from %s", n_inserted, vcf_path)
    return n_inserted


# ===========================================================================
# 3. Genotype loader from VCF
# ===========================================================================

def load_genotypes_from_vcf(
    vcf_path: str,
    sample_ids: List[str],
) -> int:
    """Parse genotypes for a set of samples and insert into ``subject_variants``.

    Only samples present in *sample_ids* are processed.  Missing genotypes
    (``./. ``) are skipped.  Each row in ``subject_variants`` contains the
    variant_id, subject_id, and dosage (0/1/2 for diploid).

    Parameters
    ----------
    vcf_path:
        Path to the VCF or VCF.gz file.
    sample_ids:
        List of sample IDs to extract (must match VCF sample names).

    Returns
    -------
    int
        Total genotype rows inserted.
    """
    if cyvcf2 is None:
        raise ImportError("cyvcf2 is required for VCF parsing (pip install cyvcf2)")

    if not os.path.isfile(vcf_path):
        raise FileNotFoundError(f"VCF not found: {vcf_path}")

    sample_set = set(sample_ids)
    logger.info(
        "Parsing genotypes from %s for %d samples", vcf_path, len(sample_set)
    )

    vcf = cyvcf2.VCF(vcf_path)
    vcf_samples: List[str] = list(vcf.samples)

    # Build index map: sample_index -> sample_id (only for requested samples)
    sample_idx_map: Dict[int, str] = {
        idx: sname
        for idx, sname in enumerate(vcf_samples)
        if sname in sample_set
    }

    if not sample_idx_map:
        logger.warning(
            "None of the %d requested samples were found in VCF sample list.",
            len(sample_set),
        )
        vcf.close()
        return 0

    logger.info(
        "Found %d/%d requested samples in VCF.", len(sample_idx_map), len(sample_set)
    )

    CHUNK = 5000
    rows: List[Tuple] = []
    total_inserted = 0

    with get_pg_conn() as conn:
        for variant in vcf:
            chrom = variant.CHROM
            pos = variant.POS
            ref = variant.REF
            alts = variant.ALT or ["."]

            # genotypes: array of shape (n_samples, 3) — [allele0, allele1, phased]
            gts = variant.genotypes  # list of [a0, a1, phased]

            for alt_idx, alt in enumerate(alts):
                alt_allele_num = alt_idx + 1  # allele number in VCF (REF=0)
                variant_id = f"{chrom}:{pos}:{ref}:{alt}"

                for s_idx, s_id in sample_idx_map.items():
                    gt = gts[s_idx]
                    a0, a1 = gt[0], gt[1]

                    # Skip missing (-1) calls
                    if a0 < 0 or a1 < 0:
                        continue

                    # Dosage of this alt allele
                    dosage = (1 if a0 == alt_allele_num else 0) + (
                        1 if a1 == alt_allele_num else 0
                    )

                    if dosage == 0:
                        continue  # homozygous ref — skip to save space

                    rows.append((variant_id, s_id, dosage))

            if len(rows) >= CHUNK:
                n = bulk_insert(
                    conn=conn,
                    table="subject_variants",
                    columns=["variant_id", "subject_id", "dosage"],
                    rows=rows,
                    on_conflict="(variant_id, subject_id) DO NOTHING",
                )
                total_inserted += n
                rows = []

        # Flush remainder
        if rows:
            n = bulk_insert(
                conn=conn,
                table="subject_variants",
                columns=["variant_id", "subject_id", "dosage"],
                rows=rows,
                on_conflict="(variant_id, subject_id) DO NOTHING",
            )
            total_inserted += n

    vcf.close()
    logger.info(
        "load_genotypes_from_vcf: %d genotype rows inserted.", total_inserted
    )
    return total_inserted


# ===========================================================================
# 4. Variant QC stats via bcftools
# ===========================================================================

def compute_variant_qc_stats(vcf_path: str) -> Dict[str, Any]:
    """Run ``bcftools stats`` and parse key QC metrics.

    Requires ``bcftools`` to be on PATH.

    Parameters
    ----------
    vcf_path:
        Path to the VCF or VCF.gz file.

    Returns
    -------
    dict with keys:
        * ``n_total`` — total variant records
        * ``n_pass`` — FILTER=PASS variants
        * ``n_snp`` — SNP count
        * ``n_indel`` — indel count
        * ``ts_tv_ratio`` — transition/transversion ratio
        * ``mean_depth`` — mean DP across sites (if DP is in FORMAT/INFO)
    """
    if not os.path.isfile(vcf_path):
        raise FileNotFoundError(f"VCF not found: {vcf_path}")

    # Check bcftools availability
    try:
        subprocess.run(
            ["bcftools", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "bcftools is not available on PATH — install it before calling "
            "compute_variant_qc_stats()"
        ) from exc

    logger.info("Running bcftools stats on %s", vcf_path)
    result = subprocess.run(
        ["bcftools", "stats", vcf_path],
        capture_output=True,
        text=True,
        check=True,
    )
    stats_text = result.stdout

    metrics: Dict[str, Any] = {
        "n_total": 0,
        "n_pass": 0,
        "n_snp": 0,
        "n_indel": 0,
        "ts_tv_ratio": None,
        "mean_depth": None,
    }

    # bcftools stats summary lines begin with "SN"
    # Format: SN\t0\tcolumn_name:\tvalue
    for line in stats_text.splitlines():
        if line.startswith("SN"):
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            key = parts[2].strip().rstrip(":")
            val = parts[3].strip()

            if key == "number of records":
                metrics["n_total"] = int(val)
            elif key == "number of SNPs":
                metrics["n_snp"] = int(val)
            elif key == "number of indels":
                metrics["n_indel"] = int(val)
            elif key == "number of PASS records":
                metrics["n_pass"] = int(val)

        # Ts/Tv is in the TSTV section: TSTV\t0\t<ts>\t<tv>\t<ts/tv>
        elif line.startswith("TSTV"):
            parts = line.split("\t")
            if len(parts) >= 5:
                try:
                    metrics["ts_tv_ratio"] = float(parts[4])
                except ValueError:
                    pass

        # Mean depth from DP lines (if available)
        elif line.startswith("DP"):
            # DP\t0\t<bin>\t<count>\t<fraction>
            # We skip these bin lines; mean_depth requires a different approach
            pass

    # Extract mean depth from the "Mean DP" line in PSC section if present
    for line in stats_text.splitlines():
        if "Mean depth:" in line or "mean depth" in line.lower():
            m = re.search(r"[\d.]+", line.split(":")[-1])
            if m:
                metrics["mean_depth"] = float(m.group())
                break

    logger.info(
        "QC stats: %d total, %d PASS, %d SNP, %d indel, Ts/Tv=%.3f",
        metrics["n_total"],
        metrics["n_pass"],
        metrics["n_snp"],
        metrics["n_indel"],
        metrics["ts_tv_ratio"] or 0.0,
    )
    return metrics


# ===========================================================================
# 5. ClinVar annotation
# ===========================================================================

def annotate_variants_clinvar(
    variants_list: List[str],
    clinvar_vcf_path: str,
) -> int:
    """Cross-reference variant IDs against a ClinVar VCF and update the DB.

    For each matching variant the ``clinvar_sig`` column in the ``variants``
    table is updated with the ClinVar CLNSIG value.

    Parameters
    ----------
    variants_list:
        List of variant IDs (``chrom:pos:ref:alt``) to annotate.
    clinvar_vcf_path:
        Path to the ClinVar VCF (or VCF.gz) file.

    Returns
    -------
    int
        Number of variants annotated.
    """
    if cyvcf2 is None:
        raise ImportError("cyvcf2 is required for VCF parsing (pip install cyvcf2)")

    if not os.path.isfile(clinvar_vcf_path):
        raise FileNotFoundError(f"ClinVar VCF not found: {clinvar_vcf_path}")

    if not variants_list:
        logger.warning("annotate_variants_clinvar called with empty variants_list.")
        return 0

    # Build lookup: variant_id -> True (set for O(1) membership)
    target_set = set(variants_list)
    logger.info(
        "Annotating %d variants from ClinVar VCF: %s",
        len(target_set),
        clinvar_vcf_path,
    )

    clinvar_vcf = cyvcf2.VCF(clinvar_vcf_path)
    annotations: Dict[str, str] = {}  # variant_id -> CLNSIG

    for cv_var in clinvar_vcf:
        chrom = cv_var.CHROM.lstrip("chr")  # normalise chr prefix
        pos = cv_var.POS
        ref = cv_var.REF
        alts = cv_var.ALT or ["."]

        clnsig = cv_var.INFO.get("CLNSIG", None)
        if clnsig is None:
            continue
        clnsig = str(clnsig)

        for alt in alts:
            # Try both chr-prefixed and plain chromosome
            for chrom_form in (chrom, f"chr{chrom}"):
                vid = f"{chrom_form}:{pos}:{ref}:{alt}"
                if vid in target_set:
                    annotations[vid] = clnsig
                    break

    clinvar_vcf.close()

    if not annotations:
        logger.info("No ClinVar matches found for the provided variant list.")
        return 0

    logger.info("Found %d ClinVar matches; updating variants table.", len(annotations))

    # Update variants table
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            update_sql = (
                "UPDATE variants SET clinvar_sig = %s "
                "WHERE variant_id = %s"
            )
            update_data = [
                (sig, vid) for vid, sig in annotations.items()
            ]
            psycopg2.extras.execute_batch(cur, update_sql, update_data, page_size=500)
        conn.commit()

    import psycopg2.extras  # re-import for type checker

    n_annotated = len(annotations)
    logger.info(
        "annotate_variants_clinvar: %d variants annotated with ClinVar signatures.",
        n_annotated,
    )
    return n_annotated


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Genetics QC utilities")
    sub = parser.add_subparsers(dest="cmd")

    p_eigenvec = sub.add_parser("load-pca", help="Load PCA eigenvec file")
    p_eigenvec.add_argument("eigenvec")
    p_eigenvec.add_argument("--run-id", default="default_run")
    p_eigenvec.add_argument("--n-variants", type=int, default=0)

    p_vcf = sub.add_parser("load-variants", help="Load variants from VCF")
    p_vcf.add_argument("vcf")
    p_vcf.add_argument("--max-variants", type=int, default=500_000)

    p_qc = sub.add_parser("qc-stats", help="Compute QC stats for a VCF")
    p_qc.add_argument("vcf")

    args = parser.parse_args()

    if args.cmd == "load-pca":
        n = load_pca_eigenvec(args.eigenvec, args.run_id, args.n_variants)
        print(f"Loaded {n} subjects.")
    elif args.cmd == "load-variants":
        n = load_variants_from_vcf(args.vcf, args.max_variants)
        print(f"Loaded {n} variants.")
    elif args.cmd == "qc-stats":
        stats = compute_variant_qc_stats(args.vcf)
        print(json.dumps(stats, indent=2))
    else:
        parser.print_help()
