#!/usr/bin/env python3
"""
init_sqlite.py — Initializes the SQLite database for the BioIntelligence Platform.
Translates the PostgreSQL schema into SQLite tables, indexes, and views.
Registers a custom standard deviation aggregator for statistical queries.
"""

import os
import sqlite3
import math
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("init_sqlite")

class SqliteStdDev:
    """Custom standard deviation aggregator for SQLite."""
    def __init__(self):
        self.values = []
    def step(self, value):
        if value is not None:
            self.values.append(float(value))
    def finalize(self):
        n = len(self.values)
        if n < 2:
            return 0.0
        mean = sum(self.values) / n
        variance = sum((x - mean) ** 2 for x in self.values) / (n - 1)
        return math.sqrt(variance)

def create_schema(db_path: str):
    logger.info("Initializing SQLite database: %s", db_path)
    
    conn = sqlite3.connect(db_path)
    
    # Enable registering custom functions
    conn.create_aggregate("STDDEV", 1, SqliteStdDev)
    
    cursor = conn.cursor()
    
    # 1. subjects table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS subjects (
        subject_id TEXT PRIMARY KEY,
        dataset_source TEXT NOT NULL,
        sex TEXT CHECK (sex IN ('M','F','U')),
        age_at_scan REAL,
        ethnicity_label TEXT,
        has_genetics INTEGER DEFAULT 0,
        has_mri INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # 2. ancestry_pca table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ancestry_pca (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_id TEXT REFERENCES subjects(subject_id) ON DELETE CASCADE,
        pc1 REAL, pc2 REAL, pc3 REAL, pc4 REAL, pc5 REAL,
        pc6 REAL, pc7 REAL, pc8 REAL, pc9 REAL, pc10 REAL,
        pc11 REAL, pc12 REAL, pc13 REAL, pc14 REAL, pc15 REAL,
        pc16 REAL, pc17 REAL, pc18 REAL, pc19 REAL, pc20 REAL,
        pc21 REAL, pc22 REAL, pc23 REAL, pc24 REAL, pc25 REAL,
        pc26 REAL, pc27 REAL, pc28 REAL, pc29 REAL, pc30 REAL,
        pc31 REAL, pc32 REAL, pc33 REAL, pc34 REAL, pc35 REAL,
        pc36 REAL, pc37 REAL, pc38 REAL, pc39 REAL, pc40 REAL,
        pca_run_id TEXT,
        n_variants_used INTEGER,
        computed_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # 3. variants table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS variants (
        variant_id TEXT PRIMARY KEY,
        chrom TEXT NOT NULL,
        pos INTEGER NOT NULL,
        ref TEXT NOT NULL,
        alt TEXT NOT NULL,
        rsid TEXT,
        gene_symbol TEXT,
        consequence TEXT,
        clinvar_sig TEXT,
        gnomad_af REAL,
        dataset_source TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_variants_chrom_pos ON variants(chrom, pos);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_variants_gene ON variants(gene_symbol);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_variants_rsid ON variants(rsid);")
    
    # 4. subject_variants table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS subject_variants (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_id TEXT REFERENCES subjects(subject_id) ON DELETE CASCADE,
        variant_id TEXT REFERENCES variants(variant_id) ON DELETE CASCADE,
        genotype TEXT,
        gq INTEGER,
        dp INTEGER,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(subject_id, variant_id)
    );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sv_subject ON subject_variants(subject_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sv_variant ON subject_variants(variant_id);")
    
    # 5. brain_morphometry table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS brain_morphometry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_id TEXT REFERENCES subjects(subject_id) ON DELETE CASCADE,
        segmentation_tool TEXT DEFAULT 'FastSurfer',
        region TEXT NOT NULL,
        volume_mm3 REAL,
        thickness_mm REAL,
        surface_area_mm2 REAL,
        laterality TEXT CHECK (laterality IN ('L','R','B')),
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bm_subject ON brain_morphometry(subject_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bm_region ON brain_morphometry(region);")
    
    # 6. dna_model_predictions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dna_model_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_id TEXT REFERENCES subjects(subject_id) ON DELETE CASCADE,
        model_name TEXT NOT NULL,
        model_version TEXT,
        sequence_context TEXT,
        embedding TEXT, -- stored as JSON string array
        pred_label TEXT,
        pred_score REAL,
        variant_id TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # 7. mri_model_predictions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS mri_model_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_id TEXT REFERENCES subjects(subject_id) ON DELETE CASCADE,
        model_name TEXT NOT NULL,
        brain_age_predicted REAL,
        brain_age_delta REAL,
        ci_lower REAL,
        ci_upper REAL,
        n_train_subjects INTEGER,
        model_type TEXT,
        disclaimer TEXT DEFAULT 'RESEARCH USE ONLY',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(subject_id, model_name)
    );
    """)
    
    # 8. pca_brain_correlations table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pca_brain_correlations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pc_id INTEGER NOT NULL CHECK (pc_id BETWEEN 1 AND 40),
        region TEXT NOT NULL,
        pearson_r REAL,
        p_value_raw REAL,
        p_value_fdr REAL,
        is_significant INTEGER,
        n_samples INTEGER,
        computed_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # 9. gene_expression table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS gene_expression (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_id TEXT REFERENCES subjects(subject_id) ON DELETE CASCADE,
        gene_id TEXT NOT NULL,
        gene_symbol TEXT,
        tissue TEXT,
        tpm REAL NOT NULL,
        dataset_source TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ge_gene ON gene_expression(gene_symbol);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ge_tissue ON gene_expression(tissue);")
    
    # 10. pipeline_runs table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pipeline_runs (
        run_id TEXT PRIMARY KEY,
        dag_id TEXT NOT NULL,
        run_type TEXT DEFAULT 'manual',
        status TEXT NOT NULL DEFAULT 'running',
        started_at TEXT DEFAULT CURRENT_TIMESTAMP,
        completed_at TEXT,
        n_subjects_processed INTEGER,
        error_message TEXT,
        log_uri TEXT,
        config TEXT -- stored as JSON string
    );
    """)
    
    # 11. llm_narratives table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS llm_narratives (
        narrative_id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_id TEXT REFERENCES subjects(subject_id) ON DELETE CASCADE,
        analysis_type TEXT NOT NULL DEFAULT 'combined',
        biomistral_text TEXT,
        llama_text TEXT,
        qwen_json TEXT, -- stored as JSON string
        final_narrative TEXT,
        ethical_disclaimer TEXT NOT NULL,
        model_versions TEXT, -- stored as JSON string
        generation_params TEXT, -- stored as JSON string
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(subject_id, analysis_type)
    );
    """)
    
    # 11.3 invite_codes table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS invite_codes (
        code TEXT PRIMARY KEY,
        created_by TEXT,
        max_uses INTEGER DEFAULT 1,
        used_count INTEGER DEFAULT 0,
        allotted_credits INTEGER DEFAULT NULL,
        status TEXT DEFAULT 'active',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 11.5 users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        email TEXT,
        security_question_1 TEXT,
        security_answer_hash_1 TEXT,
        security_question_2 TEXT,
        security_answer_hash_2 TEXT,
        salt_questions TEXT,
        referred_by_code TEXT REFERENCES invite_codes(code) ON DELETE SET NULL,
        credits_remaining INTEGER DEFAULT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Attempt to alter existing users table if they were created before
    for col_name in ["security_question_1", "security_answer_hash_1", "security_question_2", "security_answer_hash_2", "salt_questions"]:
        try:
            cursor.execute(f"ALTER TABLE users ADD COLUMN {col_name} TEXT;")
        except sqlite3.OperationalError:
            pass
            
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN referred_by_code TEXT REFERENCES invite_codes(code) ON DELETE SET NULL;")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE invite_codes ADD COLUMN allotted_credits INTEGER DEFAULT NULL;")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE users ADD COLUMN credits_remaining INTEGER DEFAULT NULL;")
    except sqlite3.OperationalError:
        pass

    # Seed initial invite codes if empty
    cursor.execute("SELECT COUNT(*) FROM invite_codes WHERE code = 'BIOINTEL2026';")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO invite_codes (code, created_by, max_uses, used_count) VALUES ('BIOINTEL2026', 'system', 9999, 0);")

    cursor.execute("SELECT COUNT(*) FROM invite_codes WHERE code = 'DEMO_FREE';")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO invite_codes (code, created_by, max_uses, used_count, allotted_credits) VALUES ('DEMO_FREE', 'system', 1000, 0, 3);")




    # 12. user_sessions: named analysis sessions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_sessions (
        session_id TEXT PRIMARY KEY,
        session_name TEXT NOT NULL,
        description TEXT,
        username TEXT REFERENCES users(username) ON DELETE CASCADE,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        analysis_count INTEGER DEFAULT 0
    );
    """)

    # Attempt to alter existing table in case it was already created without the username column
    try:
        cursor.execute("ALTER TABLE user_sessions ADD COLUMN username TEXT REFERENCES users(username) ON DELETE CASCADE;")
    except sqlite3.OperationalError:
        pass


    # 13. user_uploads: tracks files uploaded per session  
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_uploads (
        upload_id TEXT PRIMARY KEY,
        session_id TEXT REFERENCES user_sessions(session_id) ON DELETE CASCADE,
        filename TEXT NOT NULL,
        file_type TEXT,
        upload_time TEXT DEFAULT CURRENT_TIMESTAMP,
        file_size_bytes INTEGER,
        row_count INTEGER,
        status TEXT DEFAULT 'processed'
    );
    """)

    # 14. user_analyses: stores every analysis result
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_analyses (
        analysis_id TEXT PRIMARY KEY,
        session_id TEXT REFERENCES user_sessions(session_id) ON DELETE CASCADE,
        analysis_type TEXT NOT NULL,  -- 'brain_age', 'variant_annotation', 'dna_embedding'
        subject_label TEXT,
        input_params_json TEXT,
        result_json TEXT,
        brain_age_predicted REAL,
        brain_age_delta REAL,
        ci_lower REAL,
        ci_upper REAL,
        variant_count INTEGER,
        narrative_text TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        notes TEXT
    );
    """)

    
    # =========================================================================
    # CREATE VIEWS
    # =========================================================================
    
    # View 1: v_subject_overview
    cursor.execute("""
    CREATE VIEW IF NOT EXISTS v_subject_overview AS
    SELECT
        s.subject_id,
        s.dataset_source,
        s.sex,
        s.age_at_scan,
        s.has_genetics,
        s.has_mri,
        ap.pc1, ap.pc2, ap.pc3,
        (SELECT COUNT(DISTINCT sv.variant_id) FROM subject_variants sv WHERE sv.subject_id = s.subject_id) AS n_variants,
        (SELECT COUNT(DISTINCT bm.region) FROM brain_morphometry bm WHERE bm.subject_id = s.subject_id) AS n_brain_regions
    FROM subjects s
    LEFT JOIN ancestry_pca ap ON s.subject_id = ap.subject_id;
    """)
    
    # View 2: v_significant_pca_brain
    cursor.execute("""
    CREATE VIEW IF NOT EXISTS v_significant_pca_brain AS
    SELECT pc_id, region AS brain_region, pearson_r, p_value_raw AS p_value, p_value_fdr AS fdr_corrected_p, n_samples AS n_subjects, computed_at
    FROM pca_brain_correlations
    WHERE is_significant = 1
    ORDER BY p_value_fdr ASC;
    """)
    
    # View 3: subject_pcs
    cursor.execute("""
    CREATE VIEW IF NOT EXISTS subject_pcs AS
    SELECT 
        ap.subject_id,
        ap.pc1, ap.pc2, ap.pc3, ap.pc4, ap.pc5, ap.pc6, ap.pc7, ap.pc8, ap.pc9, ap.pc10,
        s.ethnicity_label AS ancestry_label,
        s.dataset_source
    FROM ancestry_pca ap
    JOIN subjects s ON ap.subject_id = s.subject_id;
    """)
    
    # View 4: brain_age_predictions
    cursor.execute("""
    CREATE VIEW IF NOT EXISTS brain_age_predictions AS
    SELECT 
        mmp.id AS ba_pred_id,
        mmp.subject_id,
        NULL AS session_id,
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
    """)
    
    # View 5: brain_region_volumes
    cursor.execute("""
    CREATE VIEW IF NOT EXISTS brain_region_volumes AS
    WITH stats AS (
        SELECT region AS stat_region, AVG(volume_mm3) AS avg_vol, STDDEV(volume_mm3) AS std_vol
        FROM brain_morphometry
        GROUP BY region
    )
    SELECT 
        bm.subject_id,
        bm.region AS region_name,
        bm.volume_mm3,
        (bm.volume_mm3 - stats.avg_vol) / NULLIF(stats.std_vol, 0) AS z_score_from_mean
    FROM brain_morphometry bm
    JOIN stats ON bm.region = stats.stat_region;
    """)
    
    # View 6: pca_embeddings
    cursor.execute("""
    CREATE VIEW IF NOT EXISTS pca_embeddings AS
    SELECT 
        ap.id AS emb_id,
        ap.subject_id,
        'ancestry' AS embedding_type,
        ap.pc1, ap.pc2, ap.pc3, ap.pc4, ap.pc5,
        0.0 AS explained_variance_ratio,
        ap.computed_at,
        s.dataset_source AS cohort,
        s.ethnicity_label AS super_population,
        s.sex,
        s.age_at_scan AS age_at_enrolment
    FROM ancestry_pca ap
    JOIN subjects s ON ap.subject_id = s.subject_id;
    """)
    
    # View 7: genomic_vars
    cursor.execute("""
    CREATE VIEW IF NOT EXISTS genomic_vars AS
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
    """)
    
    # View 8: dna_model_predictions_pca
    cursor.execute("""
    CREATE VIEW IF NOT EXISTS dna_model_predictions_pca AS
    SELECT 
        id,
        subject_id,
        model_name,
        model_version,
        pred_label,
        pred_score,
        variant_id,
        created_at,
        -- Extract from JSON text array. Element 0 is index 0
        coalesce(json_extract(embedding, '$[0]'), 0.0) AS pca_dim1,
        coalesce(json_extract(embedding, '$[1]'), 0.0) AS pca_dim2
    FROM dna_model_predictions;
    """)
    
    # View 9: subject_pcs_brain_joined
    cursor.execute("""
    CREATE VIEW IF NOT EXISTS subject_pcs_brain_joined AS
    SELECT 
        ap.subject_id,
        ap.pc1, ap.pc2, ap.pc3, ap.pc4, ap.pc5, ap.pc6, ap.pc7, ap.pc8, ap.pc9, ap.pc10,
        (SELECT SUM(bm.volume_mm3) FROM brain_morphometry bm WHERE bm.subject_id = ap.subject_id) AS total_brain_volume,
        s.ethnicity_label AS super_population,
        s.dataset_source AS cohort
    FROM ancestry_pca ap
    JOIN subjects s ON ap.subject_id = s.subject_id;
    """)
    
    # View 10: pc_brain_correlation_matrix
    cursor.execute("""
    CREATE VIEW IF NOT EXISTS pc_brain_correlation_matrix AS
    SELECT 
        region AS region_name,
        'PC' || pc_id AS pc_name,
        pearson_r
    FROM pca_brain_correlations;
    """)
    
    # View 11: brain_morphometry_wide
    cursor.execute("""
    CREATE VIEW IF NOT EXISTS brain_morphometry_wide AS
    SELECT 
        subject_id AS morph_id,
        subject_id,
        NULL AS session_id,
        (SELECT s.age_at_scan FROM subjects s WHERE s.subject_id = bm.subject_id) AS age_at_scan,
        (SELECT s.dataset_source FROM subjects s WHERE s.subject_id = bm.subject_id) AS cohort,
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
    GROUP BY subject_id;
    """)
    
    conn.commit()
    conn.close()
    logger.info("✔ SQLite Database and bridging views initialized successfully.")

if __name__ == "__main__":
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "biointel.db")
    create_schema(db_path)
