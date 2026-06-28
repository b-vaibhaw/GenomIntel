"""
mri_analysis.py — MRI morphometry parsing and PCA-brain correlation analysis
Handles: FastSurfer aseg.stats parsing, brain_morphometry loading,
         PCA-brain Pearson correlation with FDR correction.
"""
import os
import re
import logging
import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from db_utils import get_pg_conn, bulk_insert, query_to_df

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
MODEL_DIR = os.environ.get("BIOINTEL_MODEL_DIR", "/opt/biointel/models")

# Regions used for brain-age estimation (consistent with literature)
BRAIN_AGE_FEATURES = [
    "hippocampus",
    "entorhinal",
    "fusiform",
    "inferior_temporal",
    "middle_temporal",
]


# ===========================================================================
# 1. Parse FastSurfer / FreeSurfer aseg.stats
# ===========================================================================

def parse_fastsurfer_aseg_stats(stats_file: str) -> List[Dict[str, Any]]:
    """Parse a FreeSurfer/FastSurfer ``aseg.stats`` file.

    Each non-comment line after the column header contains volumetric data for
    one segmented structure.  Additionally, if ``lh.aparc.stats`` and
    ``rh.aparc.stats`` are present in the same ``stats/`` directory, they are
    also parsed for cortical thickness and surface area.

    Parameters
    ----------
    stats_file:
        Absolute path to the ``aseg.stats`` file.

    Returns
    -------
    List[dict]
        One dict per brain region with keys:
        ``region``, ``volume_mm3``, ``thickness_mm``,
        ``surface_area_mm2``, ``laterality``.
    """
    if not os.path.isfile(stats_file):
        raise FileNotFoundError(f"aseg.stats not found: {stats_file}")

    logger.info("Parsing aseg.stats: %s", stats_file)

    # ------------------------------------------------------------------
    # aseg.stats column layout (space-separated, variable whitespace):
    # Index  SegId  NVoxels  Volume_mm3  StructName  normMean  normStdDev  ...
    # ------------------------------------------------------------------
    # Column indices (0-based after splitting)
    # 0:Index 1:SegId 2:NVoxels 3:Volume_mm3 4:StructName 5:normMean 6:normStdDev
    # There can be trailing columns; we only need up to index 6.
    COL_VOLUME = 3
    COL_NAME = 4
    COL_NORM_MEAN = 5
    COL_NORM_STD = 6

    regions: List[Dict[str, Any]] = []

    with open(stats_file, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                region = parts[COL_NAME]
                volume_mm3 = float(parts[COL_VOLUME])
                norm_mean = float(parts[COL_NORM_MEAN])
                norm_std = float(parts[COL_NORM_STD])
            except (ValueError, IndexError) as exc:
                logger.debug("Skipping malformed aseg line: %s (%s)", line, exc)
                continue

            regions.append(
                {
                    "region": region,
                    "volume_mm3": volume_mm3,
                    "thickness_mm": None,        # aseg doesn't have thickness
                    "surface_area_mm2": None,    # aseg doesn't have surface area
                    "laterality": _infer_laterality(region),
                    "norm_mean": norm_mean,
                    "norm_std": norm_std,
                }
            )

    # ------------------------------------------------------------------
    # Attempt to also parse cortical parcellation (aparc) files
    # ------------------------------------------------------------------
    stats_dir = os.path.dirname(stats_file)
    for hemi in ("lh", "rh"):
        aparc_path = os.path.join(stats_dir, f"{hemi}.aparc.stats")
        if os.path.isfile(aparc_path):
            aparc_regions = _parse_aparc_stats(aparc_path, hemi)
            regions.extend(aparc_regions)
            logger.info(
                "Parsed %d regions from %s.aparc.stats", len(aparc_regions), hemi
            )

    logger.info(
        "parse_fastsurfer_aseg_stats: %d total regions from %s",
        len(regions),
        stats_file,
    )
    return regions


def _infer_laterality(struct_name: str) -> str:
    """Return 'left', 'right', or 'bilateral' based on the structure name."""
    lower = struct_name.lower()
    if lower.startswith("left-") or lower.startswith("ctx-lh"):
        return "left"
    if lower.startswith("right-") or lower.startswith("ctx-rh"):
        return "right"
    return "bilateral"


def _parse_aparc_stats(aparc_path: str, hemi: str) -> List[Dict[str, Any]]:
    """Parse an ``lh.aparc.stats`` or ``rh.aparc.stats`` file.

    Columns (space-separated):
    StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd

    Parameters
    ----------
    aparc_path:
        Path to the aparc.stats file.
    hemi:
        ``"lh"`` or ``"rh"``.

    Returns
    -------
    list of region dicts
    """
    laterality = "left" if hemi == "lh" else "right"
    regions: List[Dict[str, Any]] = []

    with open(aparc_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                struct_name = parts[0]
                surf_area = float(parts[2])
                gray_vol = float(parts[3])
                thick_avg = float(parts[4])
            except (ValueError, IndexError):
                continue

            regions.append(
                {
                    "region": f"{hemi}_{struct_name}",
                    "volume_mm3": gray_vol,
                    "thickness_mm": thick_avg,
                    "surface_area_mm2": surf_area,
                    "laterality": laterality,
                    "norm_mean": None,
                    "norm_std": None,
                }
            )

    return regions


# ===========================================================================
# 2. Discover FastSurfer subjects
# ===========================================================================

def discover_fastsurfer_subjects(output_dir: str) -> List[str]:
    """Walk *output_dir* and return the IDs of subjects that have ``aseg.stats``.

    A subject is considered valid if the path::

        <output_dir>/<subject_id>/stats/aseg.stats

    exists.

    Parameters
    ----------
    output_dir:
        Root directory containing FastSurfer/FreeSurfer subject folders.

    Returns
    -------
    list of subject_id strings (the directory name, not the full path)
    """
    if not os.path.isdir(output_dir):
        raise NotADirectoryError(f"Output directory not found: {output_dir}")

    subject_ids: List[str] = []

    for entry in sorted(os.listdir(output_dir)):
        subject_path = os.path.join(output_dir, entry)
        if not os.path.isdir(subject_path):
            continue
        aseg_path = os.path.join(subject_path, "stats", "aseg.stats")
        if os.path.isfile(aseg_path):
            subject_ids.append(entry)

    logger.info(
        "discover_fastsurfer_subjects: found %d subjects in %s",
        len(subject_ids),
        output_dir,
    )
    return subject_ids


# ===========================================================================
# 3. Load morphometry to DB
# ===========================================================================

def load_morphometry_to_db(
    subject_id: str,
    regions: List[Dict[str, Any]],
) -> int:
    """Persist brain morphometry data for a single subject.

    Inserts rows into ``brain_morphometry`` and marks the subject as having
    MRI data (``subjects.has_mri = TRUE``).

    Parameters
    ----------
    subject_id:
        The subject identifier (must exist in or be upserted into ``subjects``).
    regions:
        List of region dicts as returned by :func:`parse_fastsurfer_aseg_stats`.

    Returns
    -------
    int
        Number of rows inserted into ``brain_morphometry``.
    """
    if not regions:
        logger.warning(
            "load_morphometry_to_db called with no regions for subject %s", subject_id
        )
        return 0

    rows: List[Tuple] = []
    for r in regions:
        rows.append(
            (
                subject_id,
                r["region"],
                r.get("volume_mm3"),
                r.get("thickness_mm"),
                r.get("surface_area_mm2"),
                r.get("laterality"),
            )
        )

    columns = [
        "subject_id",
        "region",
        "volume_mm3",
        "thickness_mm",
        "surface_area_mm2",
        "laterality",
    ]

    with get_pg_conn() as conn:
        # Upsert subject (ensure row exists, set has_mri=True)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO subjects (subject_id, has_mri)
                VALUES (%s, TRUE)
                ON CONFLICT (subject_id)
                DO UPDATE SET has_mri = TRUE
                """,
                (subject_id,),
            )

        n_inserted = bulk_insert(
            conn=conn,
            table="brain_morphometry",
            columns=columns,
            rows=rows,
            on_conflict="(subject_id, region) DO NOTHING",
        )

    logger.info(
        "load_morphometry_to_db: %d rows inserted for subject %s",
        n_inserted,
        subject_id,
    )
    return n_inserted


# ===========================================================================
# 4. PCA × brain-region correlation
# ===========================================================================

def compute_pca_brain_correlations(n_pcs: int = 20) -> int:
    """Compute Pearson correlations between PCs and brain volumes with FDR correction.

    Performs an inner join of ``ancestry_pca`` and ``brain_morphometry`` on
    ``subject_id`` so that only subjects with both genetics and MRI data are
    included.

    For each pair (PC_i, brain_region) a Pearson *r* and raw *p*-value are
    computed.  All p-values are then corrected with Benjamini-Hochberg FDR.
    Results are bulk-inserted into ``pca_brain_correlations``.

    Parameters
    ----------
    n_pcs:
        Number of principal components to correlate (1-based, up to 40).

    Returns
    -------
    int
        Number of correlation records inserted.
    """
    from scipy import stats as scipy_stats
    from statsmodels.stats.multitest import multipletests

    n_pcs = min(n_pcs, 40)
    logger.info("Computing PCA–brain correlations for PC1–PC%d", n_pcs)

    # ------------------------------------------------------------------
    # Fetch joined data
    # ------------------------------------------------------------------
    pc_cols_sql = ", ".join(f"ap.pc{i}" for i in range(1, n_pcs + 1))
    sql = f"""
        SELECT
            ap.subject_id,
            {pc_cols_sql},
            bm.region,
            bm.volume_mm3
        FROM ancestry_pca ap
        INNER JOIN brain_morphometry bm
            ON ap.subject_id = bm.subject_id
        WHERE bm.volume_mm3 IS NOT NULL
        ORDER BY ap.subject_id, bm.region
    """
    logger.info("Fetching PCA + morphometry data...")
    df = query_to_df(sql)

    if df.empty:
        logger.warning(
            "No subjects with both PCA and brain morphometry data found. "
            "Aborting correlation analysis."
        )
        return 0

    logger.info(
        "Fetched %d rows for %d unique subjects and %d unique regions.",
        len(df),
        df["subject_id"].nunique(),
        df["region"].nunique(),
    )

    # Pivot: index=subject_id, columns=region, values=volume_mm3
    pivot = df.pivot_table(
        index="subject_id", columns="region", values="volume_mm3", aggfunc="first"
    )

    # PC matrix aligned to pivot index
    pc_col_names = [f"pc{i}" for i in range(1, n_pcs + 1)]
    # We need subject-level PC values (take first occurrence per subject)
    pc_df = (
        df[["subject_id"] + pc_col_names]
        .drop_duplicates(subset=["subject_id"])
        .set_index("subject_id")
    )
    # Align both DataFrames to the same subjects
    common_subjects = pivot.index.intersection(pc_df.index)
    pivot = pivot.loc[common_subjects]
    pc_df = pc_df.loc[common_subjects]

    regions = list(pivot.columns)
    n_tests = n_pcs * len(regions)
    logger.info(
        "Running %d correlations (%d PCs x %d regions) on %d subjects.",
        n_tests,
        n_pcs,
        len(regions),
        len(common_subjects),
    )

    # ------------------------------------------------------------------
    # Collect raw correlations
    # ------------------------------------------------------------------
    records: List[Dict[str, Any]] = []   # will hold all (pc_id, region, r, p)
    all_p: List[float] = []

    for pc_i in range(1, n_pcs + 1):
        pc_col = f"pc{pc_i}"
        pc_values = pc_df[pc_col].values.astype(float)

        for region in regions:
            vol_values = pivot[region].values.astype(float)

            # Drop rows where either value is NaN
            mask = (~np.isnan(pc_values)) & (~np.isnan(vol_values))
            pc_clean = pc_values[mask]
            vol_clean = vol_values[mask]

            if len(pc_clean) < 10:
                logger.debug(
                    "Skipping PC%d × %s: only %d non-NaN samples",
                    pc_i,
                    region,
                    len(pc_clean),
                )
                r, p = np.nan, np.nan
            else:
                r, p = scipy_stats.pearsonr(pc_clean, vol_clean)

            records.append(
                {
                    "pc_id": pc_i,
                    "region": region,
                    "r": r,
                    "p_raw": p,
                    "n_samples": int(mask.sum()),
                }
            )
            all_p.append(p if not np.isnan(p) else 1.0)

    # ------------------------------------------------------------------
    # FDR correction (Benjamini-Hochberg)
    # ------------------------------------------------------------------
    logger.info("Applying Benjamini-Hochberg FDR correction...")
    reject_arr, fdr_p_arr, _, _ = multipletests(all_p, method="fdr_bh", alpha=0.05)

    for i, rec in enumerate(records):
        rec["p_fdr"] = float(fdr_p_arr[i])
        rec["is_significant"] = bool(reject_arr[i])

    # ------------------------------------------------------------------
    # Bulk-insert into pca_brain_correlations
    # ------------------------------------------------------------------
    db_rows: List[Tuple] = [
        (
            r["pc_id"],
            r["region"],
            float(r["r"]) if not np.isnan(r["r"]) else None,
            float(r["p_raw"]) if not np.isnan(r["p_raw"]) else None,
            r["p_fdr"],
            r["is_significant"],
            r["n_samples"],
        )
        for r in records
    ]

    columns = [
        "pc_id",
        "region",
        "pearson_r",
        "p_value_raw",
        "p_value_fdr",
        "is_significant",
        "n_samples",
    ]

    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM pca_brain_correlations")
        n_inserted = bulk_insert(
            conn=conn,
            table="pca_brain_correlations",
            columns=columns,
            rows=db_rows,
            on_conflict="DO NOTHING",
        )

    sig_count = sum(1 for r in records if r["is_significant"])
    logger.info(
        "compute_pca_brain_correlations: %d correlations inserted "
        "(%d significant at FDR < 5%%).",
        n_inserted,
        sig_count,
    )
    return n_inserted


# ===========================================================================
# 5. Brain age estimation
# ===========================================================================

def estimate_brain_age(
    subject_id: str,
    brain_volumes: Dict[str, float],
) -> Dict[str, Any]:
    """Predict brain age from hippocampal and temporal lobe volumes.

    Uses a Ridge regression model trained on subjects already loaded into the
    ``brain_morphometry`` and ``subjects`` tables (OASIS/IXI cohort).  If
    fewer than 30 reference subjects are available the function falls back to
    a published population-mean heuristic.

    The prediction is stored in ``mri_model_predictions`` and returned as a
    dict.

    Parameters
    ----------
    subject_id:
        Target subject identifier.
    brain_volumes:
        Dict mapping region name → volume_mm3.  Must contain keys from
        :data:`BRAIN_AGE_FEATURES` (missing keys are imputed with the
        training-set mean).

    Returns
    -------
    dict with keys:
        * ``brain_age_predicted`` — predicted brain age in years
        * ``brain_age_delta`` — predicted minus chronological age (if available)
        * ``ci_lower`` — 95 % confidence interval lower bound
        * ``ci_upper`` — 95 % confidence interval upper bound
        * ``n_train`` — number of training subjects used
        * ``disclaimer`` — disclaimer string
    """
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score

    DISCLAIMER = (
        "Brain age is an experimental research metric and should not be used "
        "for clinical decision-making.  Results depend on data quality and "
        "cohort composition."
    )

    # ------------------------------------------------------------------
    # Load training data from DB
    # ------------------------------------------------------------------
    feature_cols_sql = ", ".join(
        f"MAX(CASE WHEN region = '{r}' THEN volume_mm3 END) AS {r}"
        for r in BRAIN_AGE_FEATURES
    )
    train_sql = f"""
        SELECT
            s.subject_id,
            s.age_at_scan,
            {feature_cols_sql}
        FROM subjects s
        INNER JOIN brain_morphometry bm ON s.subject_id = bm.subject_id
        WHERE s.age_at_scan IS NOT NULL
          AND s.subject_id != %(subject_id)s
        GROUP BY s.subject_id, s.age_at_scan
        HAVING COUNT(bm.region) > 0
    """
    logger.info("Loading training data for brain-age model (excluding %s)", subject_id)
    train_df = query_to_df(train_sql, params={"subject_id": subject_id})

    # Drop rows with missing target
    train_df = train_df.dropna(subset=["age_at_scan"])

    n_train = len(train_df)
    logger.info("Brain-age training set: %d subjects", n_train)

    # ------------------------------------------------------------------
    # Build feature matrix
    # ------------------------------------------------------------------
    X_cols = BRAIN_AGE_FEATURES
    y_col = "age_at_scan"

    if n_train >= 30:
        X_train = train_df[X_cols].values.astype(float)
        y_train = train_df[y_col].values.astype(float)

        # Impute NaN with column means
        col_means = np.nanmean(X_train, axis=0)
        for col_idx in range(X_train.shape[1]):
            nan_mask = np.isnan(X_train[:, col_idx])
            X_train[nan_mask, col_idx] = col_means[col_idx]

        # Fit Ridge regression pipeline
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)),
            ]
        )
        pipeline.fit(X_train, y_train)

        # Cross-val RMSE for CI estimation
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=min(5, n_train), scoring="neg_mean_squared_error"
        )
        rmse = float(np.sqrt(-cv_scores.mean()))
        ci_half = 1.96 * rmse  # approximate 95 % CI

        # Predict for target subject
        x_subject = np.array(
            [brain_volumes.get(feat, col_means[i]) for i, feat in enumerate(X_cols)],
            dtype=float,
        )
        x_subject = x_subject.reshape(1, -1)
        brain_age_pred = float(pipeline.predict(x_subject)[0])
        model_type = "ridge_regression"

    else:
        # Fallback: simple population-mean heuristic using hippocampal volume
        # Published norm: hippocampal volume ~ 3800 mm³ at age 40,
        # decreasing ~1 % per year after 40.
        logger.warning(
            "Insufficient training subjects (%d < 30); using heuristic fallback.",
            n_train,
        )
        hipp_vol = brain_volumes.get("hippocampus", 3800.0)
        # Age = 40 + (3800 - hippocampus) / 38   (rough linear heuristic)
        brain_age_pred = 40.0 + (3800.0 - hipp_vol) / 38.0
        rmse = 5.0  # conservative uncertainty for heuristic
        ci_half = 1.96 * rmse
        model_type = "heuristic_fallback"

    ci_lower = brain_age_pred - ci_half
    ci_upper = brain_age_pred + ci_half

    # ------------------------------------------------------------------
    # Brain age delta (requires chronological age)
    # ------------------------------------------------------------------
    chron_age_sql = (
        "SELECT age_at_scan FROM subjects WHERE subject_id = %(subject_id)s"
    )
    age_df = query_to_df(chron_age_sql, params={"subject_id": subject_id})
    chron_age = (
        float(age_df.iloc[0]["age_at_scan"])
        if not age_df.empty and age_df.iloc[0]["age_at_scan"] is not None
        else None
    )
    brain_age_delta = (
        round(brain_age_pred - chron_age, 2) if chron_age is not None else None
    )

    result: Dict[str, Any] = {
        "subject_id": subject_id,
        "brain_age_predicted": round(brain_age_pred, 2),
        "brain_age_delta": brain_age_delta,
        "ci_lower": round(ci_lower, 2),
        "ci_upper": round(ci_upper, 2),
        "n_train": n_train,
        "model_type": model_type,
        "rmse_cv": round(rmse, 3),
        "disclaimer": DISCLAIMER,
    }

    # ------------------------------------------------------------------
    # Persist prediction
    # ------------------------------------------------------------------
    try:
        with get_pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO mri_model_predictions
                        (subject_id, model_name, brain_age_predicted,
                         brain_age_delta, ci_lower, ci_upper,
                         n_train_subjects, model_type, disclaimer,
                         created_at)
                    VALUES
                        (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (subject_id, model_name)
                    DO UPDATE SET
                        brain_age_predicted = EXCLUDED.brain_age_predicted,
                        brain_age_delta     = EXCLUDED.brain_age_delta,
                        ci_lower            = EXCLUDED.ci_lower,
                        ci_upper            = EXCLUDED.ci_upper,
                        n_train_subjects    = EXCLUDED.n_train_subjects,
                        model_type          = EXCLUDED.model_type,
                        created_at          = EXCLUDED.created_at
                    """,
                    (
                        subject_id,
                        "brain_age_v1",
                        result["brain_age_predicted"],
                        result["brain_age_delta"],
                        result["ci_lower"],
                        result["ci_upper"],
                        n_train,
                        model_type,
                        DISCLAIMER,
                        datetime.datetime.utcnow(),
                    ),
                )
        logger.info(
            "Brain-age prediction stored for %s: %.1f years (delta=%s)",
            subject_id,
            brain_age_pred,
            brain_age_delta,
        )
    except Exception as exc:
        logger.error(
            "Failed to persist brain-age prediction for %s: %s", subject_id, exc
        )

    return result


# ===========================================================================
# CLI entry point
# ===========================================================================
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="MRI analysis utilities")
    sub = parser.add_subparsers(dest="cmd")

    p_disc = sub.add_parser("discover", help="Discover FastSurfer subjects")
    p_disc.add_argument("output_dir")

    p_parse = sub.add_parser("parse-stats", help="Parse aseg.stats file")
    p_parse.add_argument("stats_file")

    p_load = sub.add_parser("load-subject", help="Load morphometry to DB")
    p_load.add_argument("subject_id")
    p_load.add_argument("stats_file")

    p_corr = sub.add_parser("correlate", help="Compute PCA-brain correlations")
    p_corr.add_argument("--n-pcs", type=int, default=20)

    p_age = sub.add_parser("brain-age", help="Estimate brain age for a subject")
    p_age.add_argument("subject_id")
    p_age.add_argument(
        "--volumes",
        nargs="+",
        metavar="REGION=VOLUME",
        help="E.g. hippocampus=3500 entorhinal=1200",
    )

    args = parser.parse_args()

    if args.cmd == "discover":
        subjects = discover_fastsurfer_subjects(args.output_dir)
        for s in subjects:
            print(s)

    elif args.cmd == "parse-stats":
        regions = parse_fastsurfer_aseg_stats(args.stats_file)
        print(json.dumps(regions, indent=2))

    elif args.cmd == "load-subject":
        regions = parse_fastsurfer_aseg_stats(args.stats_file)
        n = load_morphometry_to_db(args.subject_id, regions)
        print(f"Inserted {n} rows for {args.subject_id}")

    elif args.cmd == "correlate":
        n = compute_pca_brain_correlations(n_pcs=args.n_pcs)
        print(f"Computed {n} correlations.")

    elif args.cmd == "brain-age":
        volumes: Dict[str, float] = {}
        if args.volumes:
            for pair in args.volumes:
                region, vol = pair.split("=")
                volumes[region.strip()] = float(vol.strip())
        result = estimate_brain_age(args.subject_id, volumes)
        print(json.dumps(result, indent=2))

    else:
        parser.print_help()
