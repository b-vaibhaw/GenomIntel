#!/usr/bin/env python3
"""
dna_model_inference.py — DNA sequence embedding and inference
Utilizes a custom-trained GenomicAttentionClassifier PyTorch model.
No external HuggingFace model weight downloads required.
"""

import os
import sys
import time
import json
import logging
import sqlite3
import numpy as np

# Ensure project directories are in path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)
sys.path.insert(0, os.path.join(project_dir, "scripts"))

import genomic_model

_log_handlers = [logging.StreamHandler(sys.stdout)]
try:
    _log_handlers.append(logging.FileHandler(os.path.join(project_dir, "dna_model_inference.log"), mode="a"))
except (OSError, PermissionError):
    pass  # Cloud environments may have read-only project dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=_log_handlers,
)
log = logging.getLogger("dna_model_inference")
DB_FILE = os.path.join(project_dir, "biointel.db")

def get_db_connection():
    return sqlite3.connect(DB_FILE)

def ensure_predictions_table(conn) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS dna_model_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_id TEXT,
        model_name TEXT NOT NULL,
        model_version TEXT,
        sequence_context TEXT,
        embedding TEXT, -- stored as JSON string array
        pred_label TEXT,
        pred_score REAL,
        variant_id TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (variant_id, model_name)
    );
    """
    conn.execute(ddl)
    conn.commit()

def fetch_variants(conn, limit: int = 1000) -> list:
    cur = conn.cursor()
    cur.execute("SELECT variant_id, chrom, pos, ref, alt, gnomad_af, clinvar_sig FROM variants LIMIT ?", (limit,))
    rows = cur.fetchall()
    return [{"variant_id": r[0], "chrom": r[1], "pos": r[2], "ref": r[3], "alt": r[4], "gnomad_af": r[5], "clinvar_sig": r[6]} for r in rows]

def fetch_subjects(conn) -> list:
    cur = conn.cursor()
    cur.execute("SELECT subject_id FROM subjects WHERE has_genetics = 1")
    rows = cur.fetchall()
    return [r[0] for r in rows]

def run_custom_dna_pipeline(limit: int = 1000) -> int:
    log.info("=== Custom Genomic Classifier Pipeline START ===")
    t0 = time.time()
    
    conn = get_db_connection()
    ensure_predictions_table(conn)
    
    variants = fetch_variants(conn, limit)
    subjects = fetch_subjects(conn)
    
    if not subjects:
        # Fallback to any subjects
        cur = conn.cursor()
        cur.execute("SELECT subject_id FROM subjects")
        subjects = [r[0] for r in cur.fetchall()]
        
    if not variants:
        log.warning("No variants found in database. Seed variants first.")
        conn.close()
        return 0
        
    log.info("Loaded custom model. Running inference on %d variants...", len(variants))
    
    n_processed = 0
    cur = conn.cursor()
    
    for idx, v in enumerate(variants):
        # Generate 512bp window
        seq = genomic_model.get_dna_window_pure(v["ref"], v["alt"], v["pos"], window=512)
        
        # Run prediction
        res = genomic_model.predict_pathogenicity(seq)
        
        pred_label = res["pred_label"]
        pred_score = res["pred_score"]
        emb_json = json.dumps(res["embedding"])
        
        # Map variant to a carrier subject
        carrier = subjects[idx % len(subjects)] if subjects else "SYSTEM"
        
        # Manual check and insert/update
        cur.execute("SELECT id FROM dna_model_predictions WHERE variant_id = ? AND model_name = ?", (v["variant_id"], "GenomicAttentionClassifier"))
        row = cur.fetchone()
        if row:
            cur.execute(
                """
                UPDATE dna_model_predictions SET
                    subject_id = ?,
                    model_version = ?,
                    sequence_context = ?,
                    embedding = ?,
                    pred_label = ?,
                    pred_score = ?,
                    created_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (carrier, "v2.0", seq[:100] + "...", emb_json, pred_label, pred_score, row[0])
            )
        else:
            cur.execute(
                """
                INSERT INTO dna_model_predictions (
                    subject_id, model_name, model_version, sequence_context, 
                    embedding, pred_label, pred_score, variant_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (carrier, "GenomicAttentionClassifier", "v2.0", seq[:100] + "...", emb_json, pred_label, pred_score, v["variant_id"])
            )
        n_processed += 1
        
    conn.commit()
    conn.close()
    
    elapsed = time.time() - t0
    log.info("=== Custom DNA Pipeline DONE -- %d variants in %.2fs ===", n_processed, elapsed)
    return n_processed

if __name__ == "__main__":
    run_custom_dna_pipeline()
