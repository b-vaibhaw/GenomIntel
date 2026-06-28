-- =============================================================================
-- BioIntelligence Platform — PostgreSQL Schema
-- DB:   biointel  |  host: postgres  |  port: 5432  |  user: biointel
-- Run:  psql -h postgres -U biointel -d biointel -f init.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ---------------------------------------------------------------------------
-- 1. subjects
--    One row per research participant / scan session.
--    ethnicity_label stores the coarse label from the source metadata only;
--    continuous ancestry variation is captured by ancestry_pca below.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS subjects (
    subject_id       VARCHAR(64)  PRIMARY KEY,
    dataset_source   VARCHAR(64)  NOT NULL,          -- '1000G', 'IXI', 'OASIS3'
    sex              CHAR(1)      CHECK (sex IN ('M', 'F', 'U')),
    age_at_scan      FLOAT,
    ethnicity_label  VARCHAR(64),                    -- coarse label from source metadata only
    has_genetics     BOOLEAN      DEFAULT FALSE,
    has_mri          BOOLEAN      DEFAULT FALSE,
    created_at       TIMESTAMPTZ  DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- 2. ancestry_pca
--    PC1-PC40 continuous scores derived from genome-wide SNPs.
--    These are mathematical coordinates in genotype space — NOT racial
--    categories.  They are used solely as covariates in association analyses.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ancestry_pca (
    id               BIGSERIAL    PRIMARY KEY,
    subject_id       VARCHAR(64)  REFERENCES subjects(subject_id) ON DELETE CASCADE,
    -- PC scores (continuous; NOT racial/ethnic classifications)
    pc1  FLOAT, pc2  FLOAT, pc3  FLOAT, pc4  FLOAT, pc5  FLOAT,
    pc6  FLOAT, pc7  FLOAT, pc8  FLOAT, pc9  FLOAT, pc10 FLOAT,
    pc11 FLOAT, pc12 FLOAT, pc13 FLOAT, pc14 FLOAT, pc15 FLOAT,
    pc16 FLOAT, pc17 FLOAT, pc18 FLOAT, pc19 FLOAT, pc20 FLOAT,
    pc21 FLOAT, pc22 FLOAT, pc23 FLOAT, pc24 FLOAT, pc25 FLOAT,
    pc26 FLOAT, pc27 FLOAT, pc28 FLOAT, pc29 FLOAT, pc30 FLOAT,
    pc31 FLOAT, pc32 FLOAT, pc33 FLOAT, pc34 FLOAT, pc35 FLOAT,
    pc36 FLOAT, pc37 FLOAT, pc38 FLOAT, pc39 FLOAT, pc40 FLOAT,
    pca_run_id       VARCHAR(64),
    n_variants_used  INTEGER,
    computed_at      TIMESTAMPTZ  DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_apca_subject ON ancestry_pca(subject_id);

-- ---------------------------------------------------------------------------
-- 3. variants
--    Catalogue of observed genetic variants.
--    variant_id is the canonical chrom:pos:ref:alt key (GRCh38 coords).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS variants (
    variant_id      VARCHAR(128) PRIMARY KEY,        -- chrom:pos:ref:alt
    chrom           VARCHAR(8)   NOT NULL,
    pos             BIGINT       NOT NULL,
    ref             TEXT         NOT NULL,
    alt             TEXT         NOT NULL,
    rsid            VARCHAR(32),
    gene_symbol     VARCHAR(32),
    consequence     VARCHAR(64),
    clinvar_sig     VARCHAR(64),
    gnomad_af       FLOAT,
    dataset_source  VARCHAR(64),
    created_at      TIMESTAMPTZ  DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_variants_chrom_pos ON variants(chrom, pos);
CREATE INDEX IF NOT EXISTS idx_variants_gene      ON variants(gene_symbol);
CREATE INDEX IF NOT EXISTS idx_variants_rsid      ON variants(rsid);

-- ---------------------------------------------------------------------------
-- 4. subject_variants
--    Genotype calls per subject/variant pair.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS subject_variants (
    id          BIGSERIAL    PRIMARY KEY,
    subject_id  VARCHAR(64)  REFERENCES subjects(subject_id)  ON DELETE CASCADE,
    variant_id  VARCHAR(128) REFERENCES variants(variant_id)  ON DELETE CASCADE,
    genotype    VARCHAR(8),                         -- '0/1', '1/1', etc.
    gq          INTEGER,                            -- genotype quality (Phred)
    dp          INTEGER,                            -- read depth
    created_at  TIMESTAMPTZ  DEFAULT NOW(),
    UNIQUE (subject_id, variant_id)
);
CREATE INDEX IF NOT EXISTS idx_sv_subject ON subject_variants(subject_id);
CREATE INDEX IF NOT EXISTS idx_sv_variant ON subject_variants(variant_id);

-- ---------------------------------------------------------------------------
-- 5. brain_morphometry
--    Volumetric and surface metrics per brain region from FastSurfer.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS brain_morphometry (
    id                   BIGSERIAL    PRIMARY KEY,
    subject_id           VARCHAR(64)  REFERENCES subjects(subject_id) ON DELETE CASCADE,
    segmentation_tool    VARCHAR(32)  DEFAULT 'FastSurfer',
    region               VARCHAR(128) NOT NULL,
    volume_mm3           FLOAT,
    thickness_mm         FLOAT,
    surface_area_mm2     FLOAT,
    laterality           CHAR(1)      CHECK (laterality IN ('L', 'R', 'B')),  -- Left/Right/Bilateral
    created_at           TIMESTAMPTZ  DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_bm_subject ON brain_morphometry(subject_id);
CREATE INDEX IF NOT EXISTS idx_bm_region  ON brain_morphometry(region);

-- ---------------------------------------------------------------------------
-- 6. dna_model_predictions
--    Embeddings and predictions from DNABERT-2 / HyenaDNA.
--    embedding stores the CLS-token float vector from the transformer.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dna_model_predictions (
    id               BIGSERIAL    PRIMARY KEY,
    subject_id       VARCHAR(64)  REFERENCES subjects(subject_id) ON DELETE CASCADE,
    model_name       VARCHAR(64)  NOT NULL,          -- 'DNABERT-2', 'HyenaDNA'
    model_version    VARCHAR(32),
    sequence_context TEXT,                           -- the DNA window used
    embedding        FLOAT[],                        -- CLS token embedding vector
    pred_label       VARCHAR(64),
    pred_score       FLOAT,
    variant_id       VARCHAR(128),                   -- optional link to variants
    created_at       TIMESTAMPTZ  DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_dna_pred_subject ON dna_model_predictions(subject_id);
CREATE INDEX IF NOT EXISTS idx_dna_pred_model   ON dna_model_predictions(model_name);

-- ---------------------------------------------------------------------------
-- 7. mri_model_predictions
--    Brain age predictions from MRI-based regression models.
--    Includes confidence intervals and full transparency metadata.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mri_model_predictions (
    id                        BIGSERIAL   PRIMARY KEY,
    subject_id                VARCHAR(64) REFERENCES subjects(subject_id) ON DELETE CASCADE,
    model_name                VARCHAR(64) NOT NULL,
    brain_age_predicted       FLOAT,
    brain_age_delta           FLOAT,                 -- predicted_age - chronological_age
    confidence_interval_lower FLOAT,
    confidence_interval_upper FLOAT,
    n_subjects_training       INTEGER,               -- training set size for transparency
    disclaimer                TEXT        DEFAULT 'RESEARCH USE ONLY — not a clinical diagnosis',
    created_at                TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_mri_pred_subject ON mri_model_predictions(subject_id);

-- ---------------------------------------------------------------------------
-- 8. pca_brain_correlations
--    Pearson r between each ancestry PC and each brain morphometry region.
--    FDR correction applied across all PC x region tests.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pca_brain_correlations (
    id              BIGSERIAL    PRIMARY KEY,
    pc_id           INTEGER      NOT NULL CHECK (pc_id BETWEEN 1 AND 40),
    brain_region    VARCHAR(128) NOT NULL,
    pearson_r       FLOAT,
    p_value         FLOAT,
    n_subjects      INTEGER,
    fdr_corrected_p FLOAT,
    is_significant  BOOLEAN,                         -- FDR-adjusted p < 0.05
    computed_at     TIMESTAMPTZ  DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_pbc_pc_id        ON pca_brain_correlations(pc_id);
CREATE INDEX IF NOT EXISTS idx_pbc_brain_region ON pca_brain_correlations(brain_region);

-- ---------------------------------------------------------------------------
-- 9. gene_expression
--    TPM-normalised expression values per subject x gene x tissue.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gene_expression (
    id             BIGSERIAL   PRIMARY KEY,
    subject_id     VARCHAR(64) REFERENCES subjects(subject_id) ON DELETE CASCADE,
    gene_id        VARCHAR(32) NOT NULL,             -- Ensembl gene ID
    gene_symbol    VARCHAR(32),
    tissue         VARCHAR(64),
    tpm            FLOAT       NOT NULL,
    dataset_source VARCHAR(64),
    created_at     TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_ge_subject ON gene_expression(subject_id);
CREATE INDEX IF NOT EXISTS idx_ge_gene    ON gene_expression(gene_symbol);
CREATE INDEX IF NOT EXISTS idx_ge_tissue  ON gene_expression(tissue);

-- ---------------------------------------------------------------------------
-- 10. pipeline_runs
--     Audit table for every Airflow DAG execution.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id               UUID         PRIMARY KEY DEFAULT uuid_generate_v4(),
    dag_id               VARCHAR(128) NOT NULL,
    run_type             VARCHAR(32)  DEFAULT 'manual',
    status               VARCHAR(32)  NOT NULL DEFAULT 'running',  -- running/success/failed
    started_at           TIMESTAMPTZ  DEFAULT NOW(),
    completed_at         TIMESTAMPTZ,
    n_subjects_processed INTEGER,
    error_message        TEXT,
    log_uri              TEXT,
    config               JSONB
);
CREATE INDEX IF NOT EXISTS idx_pr_dag_id ON pipeline_runs(dag_id);
CREATE INDEX IF NOT EXISTS idx_pr_status ON pipeline_runs(status);

-- ---------------------------------------------------------------------------
-- 11. llm_narratives
--     Research summaries generated by the LLM ensemble.
--     Mandatory ethical_disclaimer column ensures every row carries a
--     clear statement that output is for research use only.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS llm_narratives (
    id                 BIGSERIAL    PRIMARY KEY,
    subject_id         VARCHAR(64)  REFERENCES subjects(subject_id) ON DELETE CASCADE,
    analysis_type      VARCHAR(64)  NOT NULL,        -- 'genetics', 'mri', 'combined'
    narrative_text     TEXT         NOT NULL,
    model_name         VARCHAR(64)  NOT NULL,         -- primary LLM
    model_ensemble     JSONB,                         -- if multiple models contributed
    prompt_template    VARCHAR(128),
    generation_params  JSONB,
    ethical_disclaimer TEXT         NOT NULL,
    generated_at       TIMESTAMPTZ  DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_llm_subject ON llm_narratives(subject_id);
CREATE INDEX IF NOT EXISTS idx_llm_type    ON llm_narratives(analysis_type);

-- ---------------------------------------------------------------------------
-- View 1: one row per subject with key summary stats
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_subject_overview AS
SELECT
    s.subject_id,
    s.dataset_source,
    s.sex,
    s.age_at_scan,
    s.has_genetics,
    s.has_mri,
    ap.pc1,
    ap.pc2,
    ap.pc3,
    COUNT(DISTINCT sv.variant_id)  AS n_variants,
    COUNT(DISTINCT bm.region)      AS n_brain_regions
FROM subjects s
LEFT JOIN ancestry_pca     ap ON s.subject_id = ap.subject_id
LEFT JOIN subject_variants sv ON s.subject_id = sv.subject_id
LEFT JOIN brain_morphometry bm ON s.subject_id = bm.subject_id
GROUP BY
    s.subject_id,
    s.dataset_source,
    s.sex,
    s.age_at_scan,
    s.has_genetics,
    s.has_mri,
    ap.pc1,
    ap.pc2,
    ap.pc3;

-- ---------------------------------------------------------------------------
-- View 2: only statistically significant PC-brain correlations
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_significant_pca_brain AS
SELECT
    pc_id,
    brain_region,
    pearson_r,
    p_value,
    fdr_corrected_p,
    n_subjects,
    computed_at
FROM pca_brain_correlations
WHERE is_significant = TRUE
ORDER BY fdr_corrected_p ASC;

-- ---------------------------------------------------------------------------
-- 12. subject_pcs view (for llm_narrative.py)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW subject_pcs AS
SELECT 
    ap.subject_id,
    ap.pc1, ap.pc2, ap.pc3, ap.pc4, ap.pc5, ap.pc6, ap.pc7, ap.pc8, ap.pc9, ap.pc10,
    s.ethnicity_label AS ancestry_label,
    s.dataset_source
FROM ancestry_pca ap
JOIN subjects s ON ap.subject_id = s.subject_id;

-- ---------------------------------------------------------------------------
-- 13. brain_age_predictions view (for llm_narrative.py and Superset)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW brain_age_predictions AS
SELECT 
    mmp.id AS ba_pred_id,
    mmp.subject_id,
    NULL::VARCHAR(64) AS session_id,
    s.age_at_scan AS chronological_age,
    mmp.brain_age_predicted AS predicted_brain_age,
    mmp.brain_age_predicted,
    mmp.brain_age_delta,
    mmp.model_name,
    mmp.created_at,
    mmp.created_at AS predicted_at,
    s.dataset_source AS cohort,
    s.dataset_source,
    s.ethnicity_label AS super_population,
    s.sex
FROM mri_model_predictions mmp
JOIN subjects s ON mmp.subject_id = s.subject_id;

-- ---------------------------------------------------------------------------
-- 14. brain_region_volumes view (for llm_narrative.py)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW brain_region_volumes AS
SELECT 
    subject_id,
    region AS region_name,
    volume_mm3,
    (volume_mm3 - AVG(volume_mm3) OVER (PARTITION BY region)) / NULLIF(STDDEV(volume_mm3) OVER (PARTITION BY region), 0) AS z_score_from_mean
FROM brain_morphometry;

-- ---------------------------------------------------------------------------
-- 15. pca_embeddings view (for Superset)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW pca_embeddings AS
SELECT 
    ap.id AS emb_id,
    ap.subject_id,
    'ancestry'::VARCHAR(64) AS embedding_type,
    ap.pc1, ap.pc2, ap.pc3, ap.pc4, ap.pc5,
    0.0::FLOAT AS explained_variance_ratio,
    ap.computed_at,
    s.dataset_source AS cohort,
    s.ethnicity_label AS super_population,
    s.sex,
    s.age_at_scan AS age_at_enrolment
FROM ancestry_pca ap
JOIN subjects s ON ap.subject_id = s.subject_id;

-- ---------------------------------------------------------------------------
-- 16. genomic_vars view (for Superset)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW genomic_vars AS
SELECT 
    v.variant_id AS var_id,
    sv.subject_id,
    v.chrom,
    v.pos,
    v.ref,
    v.alt,
    v.gene_symbol,
    v.consequence,
    v.clinvar_sig,
    v.gnomad_af,
    v.gnomad_af AS gnomad_af_afr,
    v.gnomad_af AS gnomad_af_eur,
    v.gnomad_af AS gnomad_af_sas,
    v.gnomad_af AS gnomad_af_eas,
    v.created_at AS ingested_at
FROM variants v
LEFT JOIN subject_variants sv ON v.variant_id = sv.variant_id;

-- ---------------------------------------------------------------------------
-- 17. dna_model_predictions_pca view (for Superset)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW dna_model_predictions_pca AS
SELECT 
    id,
    subject_id,
    model_name,
    model_version,
    pred_label,
    pred_score,
    variant_id,
    created_at,
    COALESCE(embedding[1], 0.0) AS pca_dim1,
    COALESCE(embedding[2], 0.0) AS pca_dim2
FROM dna_model_predictions;

-- ---------------------------------------------------------------------------
-- 18. subject_pcs_brain_joined view (for Superset)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW subject_pcs_brain_joined AS
WITH subject_brain_vols AS (
    SELECT 
        subject_id,
        SUM(volume_mm3) AS total_brain_volume
    FROM brain_morphometry
    GROUP BY subject_id
)
SELECT 
    ap.subject_id,
    ap.pc1, ap.pc2, ap.pc3, ap.pc4, ap.pc5, ap.pc6, ap.pc7, ap.pc8, ap.pc9, ap.pc10,
    sbv.total_brain_volume,
    s.ethnicity_label AS super_population,
    s.dataset_source AS cohort
FROM ancestry_pca ap
LEFT JOIN subject_brain_vols sbv ON ap.subject_id = sbv.subject_id
LEFT JOIN subjects s ON ap.subject_id = s.subject_id;

-- ---------------------------------------------------------------------------
-- 19. pc_brain_correlation_matrix view (for Superset)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW pc_brain_correlation_matrix AS
SELECT 
    brain_region AS region_name,
    'PC' || pc_id AS pc_name,
    pearson_r
FROM pca_brain_correlations;

-- ---------------------------------------------------------------------------
-- 20. brain_morphometry_wide view (for Superset)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW brain_morphometry_wide AS
WITH wide_morph AS (
    SELECT 
        subject_id,
        MAX(CASE WHEN region = 'Left-Frontal' THEN volume_mm3 END) AS lh_frontal_vol_mm3,
        MAX(CASE WHEN region = 'Right-Frontal' THEN volume_mm3 END) AS rh_frontal_vol_mm3,
        MAX(CASE WHEN region = 'Left-Temporal' THEN volume_mm3 END) AS lh_temporal_vol_mm3,
        MAX(CASE WHEN region = 'Right-Temporal' THEN volume_mm3 END) AS rh_temporal_vol_mm3,
        MAX(CASE WHEN region = 'Left-Parietal' THEN volume_mm3 END) AS lh_parietal_vol_mm3,
        MAX(CASE WHEN region = 'Right-Parietal' THEN volume_mm3 END) AS rh_parietal_vol_mm3,
        MAX(CASE WHEN region = 'Left-Occipital' THEN volume_mm3 END) AS lh_occipital_vol_mm3,
        MAX(CASE WHEN region = 'Right-Occipital' THEN volume_mm3 END) AS rh_occipital_vol_mm3,
        MAX(CASE WHEN region = 'Left-Hippocampus' THEN volume_mm3 END) AS hippocampus_lh_vol_mm3,
        MAX(CASE WHEN region = 'Right-Hippocampus' THEN volume_mm3 END) AS hippocampus_rh_vol_mm3,
        MAX(CASE WHEN region = 'Left-Amygdala' THEN volume_mm3 END) AS amygdala_lh_vol_mm3,
        MAX(CASE WHEN region = 'Right-Amygdala' THEN volume_mm3 END) AS amygdala_rh_vol_mm3,
        SUM(volume_mm3) AS total_intracranial_vol_mm3,
        MAX(CASE WHEN region = 'White-Matter' THEN volume_mm3 END) AS wm_vol_mm3,
        MAX(created_at) AS processed_at
    FROM brain_morphometry
    GROUP BY subject_id
)
SELECT 
    w.subject_id AS morph_id,
    w.subject_id,
    NULL::VARCHAR(64) AS session_id,
    s.age_at_scan,
    s.dataset_source AS cohort,
    w.lh_frontal_vol_mm3,
    w.rh_frontal_vol_mm3,
    w.lh_temporal_vol_mm3,
    w.rh_temporal_vol_mm3,
    w.lh_parietal_vol_mm3,
    w.rh_parietal_vol_mm3,
    w.lh_occipital_vol_mm3,
    w.rh_occipital_vol_mm3,
    w.hippocampus_lh_vol_mm3,
    w.hippocampus_rh_vol_mm3,
    w.amygdala_lh_vol_mm3,
    w.amygdala_rh_vol_mm3,
    w.total_intracranial_vol_mm3,
    w.wm_vol_mm3,
    w.processed_at
FROM wide_morph w
JOIN subjects s ON w.subject_id = s.subject_id;

-- ---------------------------------------------------------------------------
-- Grant minimal privileges (adjust roles / usernames as needed)
-- ---------------------------------------------------------------------------
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES    IN SCHEMA public TO biointel;
-- GRANT USAGE, SELECT                  ON ALL SEQUENCES IN SCHEMA public TO biointel;

-- End of schema

