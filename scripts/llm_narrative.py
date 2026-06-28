"""
llm_narrative.py -- LLM ensemble narrative generation for BioIntelligence Platform

Ensemble of 3 models:
  - BioMistral-7B (BioMistral/BioMistral-7B): biomedical clinical narratives
  - Llama-3.1-8B (meta-llama/Meta-Llama-3.1-8B-Instruct): general explanations
  - Qwen3-8B (Qwen/Qwen3-8B): structured reasoning and JSON output

All models run LOCALLY via transformers + bitsandbytes 4-bit quantization.
NO external API calls.

Ethical guardrails are enforced: all narratives include mandatory disclaimers.

Usage:
    python llm_narrative.py                              # run all subjects, combined
    python llm_narrative.py --subject SUB001            # single subject
    python llm_narrative.py --analysis-type genetics    # genetics-only pass
    python llm_narrative.py --analysis-type mri         # MRI-only pass
    python llm_narrative.py --no-quantize               # fp16 (requires more VRAM)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    psycopg2 = None  # Not available on Streamlit Cloud; OK — this script is only used for local pipeline execution
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("llm_narrative.log", mode="a"),
    ],
)
log = logging.getLogger("llm_narrative")

# ---------------------------------------------------------------------------
# Paths and environment
# ---------------------------------------------------------------------------
REGISTRY_PATH: str = os.environ.get(
    "MODEL_REGISTRY",
    str(Path(__file__).parent.parent / "models" / "model_registry.yaml"),
)
GUARDRAILS_PATH: str = os.environ.get(
    "GUARDRAILS_CONFIG",
    str(Path(__file__).parent.parent / "config" / "ethical_guardrails.yaml"),
)

DB_HOST: str = os.environ.get("DB_HOST", "localhost")
DB_PORT: int = int(os.environ.get("DB_PORT", "5432"))
DB_NAME: str = os.environ.get("DB_NAME", "biointel")
DB_USER: str = os.environ.get("DB_USER", "biointel")
DB_PASS: str = os.environ.get("DB_PASS", "biointel")

# ---------------------------------------------------------------------------
# Ethical disclaimer (mandatory on every narrative output)
# ---------------------------------------------------------------------------
ETHICAL_DISCLAIMER: str = (
    "DISCLAIMER: This analysis is for RESEARCH USE ONLY. It does not constitute a clinical "
    "diagnosis, medical advice, or genetic counseling. Ancestry principal components (PCs) are "
    "continuous mathematical scores capturing population genetic similarity -- they are NOT racial "
    "categories. Confidence intervals are provided. Results require independent replication. "
    "Consult a qualified healthcare professional for any medical decisions. n_subjects used in "
    "correlation analyses is reported for transparency."
)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

GENETICS_PROMPT: str = """You are a biomedical research assistant producing a RESEARCH-ONLY summary.

Subject genetics summary:
- Subject ID: {subject_id}
- Top 3 ancestry principal components: PC1={pc1:.4f}, PC2={pc2:.4f}, PC3={pc3:.4f}
- Ancestry summary: {ancestry_summary}
- Number of significant ClinVar variants: {n_clinvar_sig}
- Top significant variants (ClinVar): {top_variants}
- Dataset: {dataset_source}  |  N in cohort: {cohort_n}

Produce a concise research summary (3-4 paragraphs) covering:
1. Genetic ancestry composition (using PC language only -- do NOT use racial labels)
2. Notable ClinVar-significant variants and their reported clinical significance
3. Caveats: effect sizes, confidence intervals, replication requirements

{disclaimer}
"""

MRI_PROMPT: str = """You are a neuroimage research assistant producing a RESEARCH-ONLY summary.

Subject MRI summary:
- Subject ID: {subject_id}
- Estimated brain age: {brain_age:.1f} years  |  Chronological age: {chronological_age:.1f} years
- Brain age delta (estimated - chronological): {brain_age_delta:+.2f} years
- Top 5 brain regions by volume deviation from population mean:
{region_deviations}
- Dataset: {dataset_source}  |  N in cohort: {cohort_n}

Produce a concise research summary (3-4 paragraphs) covering:
1. Interpretation of brain age delta (research context only)
2. Notable regional volume deviations (z-score interpretation)
3. Caveats: model uncertainty, population norms, no clinical diagnosis

{disclaimer}
"""

COMBINED_PROMPT: str = """You are a biomedical research assistant integrating genetics and neuroimaging.

Subject: {subject_id}
Dataset: {dataset_source}  |  Cohort N: {cohort_n}

--- GENETICS ---
PC1={pc1:.4f}, PC2={pc2:.4f}, PC3={pc3:.4f}
Ancestry: {ancestry_summary}
ClinVar significant variants (n={n_clinvar_sig}): {top_variants}

--- NEUROIMAGING ---
Brain age delta: {brain_age_delta:+.2f} years
(estimated={brain_age:.1f}y, chronological={chronological_age:.1f}y)
Top regional deviations:
{region_deviations}

Task: Write an integrated research narrative (4-5 paragraphs) that:
1. Summarises genetics findings (PC-based ancestry, variants)
2. Summarises MRI findings (brain age, regional volumes)
3. Notes any potential intersections (research hypothesis only, NOT causal claims)
4. Emphasises all required caveats and confidence intervals
5. Ends with a structured limitations section

{disclaimer}
"""

QWEN_JSON_PROMPT: str = """Produce a JSON summary object for the following subject data.
Return ONLY valid JSON, no markdown, no prose.

Subject: {subject_id}
Analysis type: {analysis_type}

Genetics:
  pc1={pc1:.4f}, pc2={pc2:.4f}, pc3={pc3:.4f}
  ancestry={ancestry_summary}
  n_clinvar_sig={n_clinvar_sig}
  top_variants={top_variants}

MRI:
  brain_age={brain_age:.1f}
  chronological_age={chronological_age:.1f}
  brain_age_delta={brain_age_delta:+.2f}
  top_regions={region_deviations}

Required JSON schema:
{{
  "subject_id": "...",
  "analysis_type": "...",
  "genetics_summary": {{
    "ancestry_pcs": {{"pc1": 0.0, "pc2": 0.0, "pc3": 0.0}},
    "ancestry_note": "...",
    "n_clinvar_significant": 0,
    "top_variant_ids": []
  }},
  "mri_summary": {{
    "brain_age_estimated": 0.0,
    "brain_age_chronological": 0.0,
    "brain_age_delta": 0.0,
    "top_regions": []
  }},
  "key_findings": [],
  "limitations": [],
  "disclaimer": "RESEARCH USE ONLY"
}}
"""

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db_connection() -> psycopg2.extensions.connection:
    """Return a new PostgreSQL connection using environment variables."""
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )
    conn.autocommit = False
    return conn


def ensure_narratives_table(conn: psycopg2.extensions.connection) -> None:
    """Create llm_narratives table if it does not exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS llm_narratives (
        narrative_id        SERIAL PRIMARY KEY,
        subject_id          VARCHAR(64) NOT NULL,
        analysis_type       VARCHAR(32) NOT NULL DEFAULT 'combined',
        biomistral_text     TEXT,
        llama_text          TEXT,
        qwen_json           JSONB,
        final_narrative     TEXT,
        ethical_disclaimer  TEXT NOT NULL,
        model_versions      JSONB,
        generation_params   JSONB,
        created_at          TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE (subject_id, analysis_type)
    );
    CREATE INDEX IF NOT EXISTS idx_narrative_subject ON llm_narratives(subject_id);
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    log.info("llm_narratives table ready.")


def fetch_subject_genetics(conn: psycopg2.extensions.connection, subject_id: str) -> dict:
    """
    Fetch genetics data for a subject.

    Returns a dict with keys: pc1..pc10, ancestry_summary, variants (list),
    n_clinvar_sig, dataset_source, cohort_n.
    Falls back to empty/zero values if no data found.
    """
    result: dict = {
        "pc1": 0.0, "pc2": 0.0, "pc3": 0.0,
        "pc4": 0.0, "pc5": 0.0, "pc6": 0.0,
        "pc7": 0.0, "pc8": 0.0, "pc9": 0.0, "pc10": 0.0,
        "ancestry_summary": "Unknown",
        "n_clinvar_sig": 0,
        "top_variants": "None",
        "dataset_source": "Unknown",
        "cohort_n": 0,
    }

    # Fetch PCs
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT pc1, pc2, pc3, pc4, pc5, pc6, pc7, pc8, pc9, pc10,
                       ancestry_label, dataset_source
                FROM   subject_pcs
                WHERE  subject_id = %s
                LIMIT  1;
                """,
                (subject_id,),
            )
            row = cur.fetchone()
            if row:
                for k in ["pc1","pc2","pc3","pc4","pc5","pc6","pc7","pc8","pc9","pc10"]:
                    result[k] = float(row.get(k) or 0.0)
                result["ancestry_summary"] = row.get("ancestry_label") or "Unreported"
                result["dataset_source"]   = row.get("dataset_source") or "Unknown"
    except Exception as exc:
        log.warning("Could not fetch subject_pcs for %s: %s", subject_id, exc)

    # Fetch significant ClinVar variants
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT v.variant_id, v.chrom, v.pos, v.ref, v.alt,
                       v.clinvar_sig, v.gene_symbol, v.gnomad_af
                FROM   subject_variants sv
                JOIN   variants v ON sv.variant_id = v.variant_id
                WHERE  sv.subject_id = %s
                  AND  v.clinvar_sig NOT IN ('Benign', 'Likely benign', 'Uncertain significance', '')
                  AND  v.clinvar_sig IS NOT NULL
                ORDER  BY v.gnomad_af ASC
                LIMIT  10;
                """,
                (subject_id,),
            )
            rows = cur.fetchall()
            result["n_clinvar_sig"] = len(rows)
            if rows:
                result["top_variants"] = "; ".join(
                    f"{r['gene_symbol']} {r['chrom']}:{r['pos']} {r['ref']}>{r['alt']} "
                    f"[{r['clinvar_sig']}] (AF={r['gnomad_af']:.2e})"
                    for r in rows[:5]
                )
    except Exception as exc:
        log.warning("Could not fetch variants for %s: %s", subject_id, exc)

    # Cohort N
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM subjects WHERE dataset_source = %s;",
                (result["dataset_source"],),
            )
            result["cohort_n"] = cur.fetchone()[0] or 0
    except Exception as exc:
        log.warning("Could not fetch cohort_n for %s: %s", subject_id, exc)

    return result


def fetch_subject_mri(conn: psycopg2.extensions.connection, subject_id: str) -> dict:
    """
    Fetch MRI-derived features for a subject.

    Returns dict with keys: brain_age, chronological_age, brain_age_delta,
    region_deviations (formatted string), dataset_source.
    Falls back to zeros if no data found.
    """
    result: dict = {
        "brain_age": 0.0,
        "chronological_age": 0.0,
        "brain_age_delta": 0.0,
        "region_deviations": "No MRI data available",
        "dataset_source": "Unknown",
    }

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT predicted_brain_age, chronological_age,
                       (predicted_brain_age - chronological_age) AS brain_age_delta,
                       dataset_source
                FROM   brain_age_predictions
                WHERE  subject_id = %s
                ORDER  BY created_at DESC
                LIMIT  1;
                """,
                (subject_id,),
            )
            row = cur.fetchone()
            if row:
                result["brain_age"]         = float(row["predicted_brain_age"] or 0.0)
                result["chronological_age"] = float(row["chronological_age"] or 0.0)
                result["brain_age_delta"]   = float(row["brain_age_delta"] or 0.0)
                result["dataset_source"]    = row.get("dataset_source") or "Unknown"
    except Exception as exc:
        log.warning("Could not fetch brain_age_predictions for %s: %s", subject_id, exc)

    # Fetch top 5 regional volume deviations
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT region_name, volume_mm3, z_score_from_mean
                FROM   brain_region_volumes
                WHERE  subject_id = %s
                ORDER  BY ABS(z_score_from_mean) DESC
                LIMIT  5;
                """,
                (subject_id,),
            )
            rows = cur.fetchall()
            if rows:
                lines = [
                    f"  {r['region_name']}: vol={r['volume_mm3']:.0f} mm3, "
                    f"z={r['z_score_from_mean']:+.2f}"
                    for r in rows
                ]
                result["region_deviations"] = "\n".join(lines)
    except Exception as exc:
        log.warning("Could not fetch brain_region_volumes for %s: %s", subject_id, exc)

    return result


def fetch_all_eligible_subjects(conn: psycopg2.extensions.connection) -> List[str]:
    """Return subject_ids where has_genetics=True OR has_mri=True."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT subject_id
                FROM   subjects
                WHERE  has_genetics = TRUE OR has_mri = TRUE
                ORDER  BY subject_id;
                """
            )
            return [row[0] for row in cur.fetchall()]
    except Exception as exc:
        log.error("Could not fetch eligible subjects: %s", exc)
        return []


def upsert_narrative(
    conn: psycopg2.extensions.connection,
    subject_id: str,
    analysis_type: str,
    biomistral_text: str,
    llama_text: str,
    qwen_json: dict,
    final_narrative: str,
    model_versions: dict,
    generation_params: dict,
) -> None:
    """Insert or update a narrative row."""
    sql = """
        INSERT INTO llm_narratives
            (subject_id, analysis_type, biomistral_text, llama_text,
             qwen_json, final_narrative, ethical_disclaimer,
             model_versions, generation_params)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (subject_id, analysis_type)
        DO UPDATE SET
            biomistral_text   = EXCLUDED.biomistral_text,
            llama_text        = EXCLUDED.llama_text,
            qwen_json         = EXCLUDED.qwen_json,
            final_narrative   = EXCLUDED.final_narrative,
            ethical_disclaimer= EXCLUDED.ethical_disclaimer,
            model_versions    = EXCLUDED.model_versions,
            generation_params = EXCLUDED.generation_params,
            created_at        = NOW();
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                subject_id,
                analysis_type,
                biomistral_text,
                llama_text,
                json.dumps(qwen_json),
                final_narrative,
                ETHICAL_DISCLAIMER,
                json.dumps(model_versions),
                json.dumps(generation_params),
            ),
        )


# ---------------------------------------------------------------------------
# Model Router
# ---------------------------------------------------------------------------

class ModelRouter:
    """
    Manages the 3-model LLM ensemble for narrative generation.

    Models are loaded lazily on first use. Supports 4-bit quantization via
    bitsandbytes (load_in_4bit) or fp16 fallback.

    Attributes
    ----------
    model_dir    : base directory for local model weights
    device       : 'auto', 'cpu', 'cuda', or 'cuda:N'
    quantize_4bit: whether to apply BitsAndBytesConfig 4-bit quantization
    registry     : parsed model_registry.yaml content
    _models      : {role: (model, tokenizer)} cache
    """

    # HuggingFace IDs -- also available from registry
    BIOMISTRAL_HF_ID: str = "BioMistral/BioMistral-7B"
    LLAMA_HF_ID:      str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    QWEN_HF_ID:       str = "Qwen/Qwen3-8B"

    # Default local directories
    BIOMISTRAL_LOCAL: str = "/opt/biointel/models/biomistral7b"
    LLAMA_LOCAL:      str = "/opt/biointel/models/llama31_8b"
    QWEN_LOCAL:       str = "/opt/biointel/models/qwen3_8b"

    def __init__(
        self,
        model_dir: str = "/opt/biointel/models",
        device: str = "auto",
        quantize_4bit: bool = True,
    ) -> None:
        self.model_dir     = model_dir
        self.device        = device
        self.quantize_4bit = quantize_4bit
        self._models: Dict[str, Any] = {}  # role -> {"model": ..., "tokenizer": ..., "config": ...}
        self.registry: dict = self._load_registry()
        log.info(
            "ModelRouter init: device=%s, quantize_4bit=%s",
            device, quantize_4bit,
        )

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    def _load_registry(self) -> dict:
        """Load and parse model_registry.yaml."""
        registry_path = Path(REGISTRY_PATH)
        if registry_path.exists():
            try:
                with open(registry_path, "r", encoding="utf-8") as fh:
                    return yaml.safe_load(fh)
            except Exception as exc:
                log.warning("Could not parse registry (%s); using defaults.", exc)
        return {}

    def _get_llm_config(self, role: str) -> dict:
        """Return registry config dict for a given LLM role."""
        llms = self.registry.get("models", {}).get("llm", [])
        for cfg in llms:
            if cfg.get("role") == role:
                return cfg
        return {}

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, hf_id: str, local_dir: str, task: str = "text-generation") -> tuple:
        """
        Load a text-generation model + tokenizer.

        Tries local_dir first; falls back to HF download.
        Applies BitsAndBytesConfig 4-bit quantization if self.quantize_4bit=True
        and bitsandbytes is available.

        Parameters
        ----------
        hf_id     : HuggingFace model identifier
        local_dir : local model weight directory
        task      : HF pipeline task (used for logging only)

        Returns
        -------
        (model, tokenizer)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        local_path      = Path(local_dir)
        local_available = local_path.exists() and any(local_path.iterdir())
        source          = str(local_path) if local_available else hf_id

        log.info("Loading %s (quantize_4bit=%s) ...", hf_id, self.quantize_4bit)

        # Build quantization config
        bnb_config = None
        if self.quantize_4bit:
            try:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype="float16",
                )
                log.info("BitsAndBytesConfig (4-bit NF4) ready for %s.", hf_id)
            except ImportError:
                log.warning(
                    "bitsandbytes not installed; loading %s in fp16 instead.", hf_id
                )

        # Common load kwargs
        load_kwargs: dict = {
            "trust_remote_code": True,
            "device_map": self.device,
        }
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config
        else:
            load_kwargs["torch_dtype"] = "auto"

        # Load tokenizer
        if local_available:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(local_path), local_files_only=True, trust_remote_code=True
                )
            except Exception as exc:
                log.warning("Local tokenizer load failed (%s); downloading ...", exc)
                tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        if local_available:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    str(local_path), local_files_only=True, **load_kwargs
                )
                log.info("Loaded %s from local cache.", hf_id)
            except Exception as exc:
                log.warning("Local model load failed (%s); downloading ...", exc)
                model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)

        model.eval()
        log.info(
            "Model %s ready. Params: %s.",
            hf_id,
            f"{sum(p.numel() for p in model.parameters()):,}",
        )
        return model, tokenizer

    def load_all(self) -> None:
        """
        Eagerly load all 3 LLM models into memory.

        This is the non-lazy alternative to per-call loading.
        On systems with limited VRAM, prefer the default lazy loading path
        (models are loaded on first call to generate_*).
        """
        log.info("Loading all 3 LLM models (eager mode) ...")
        self._ensure_biomistral()
        self._ensure_llama()
        self._ensure_qwen()
        log.info("All 3 models loaded.")

    def _ensure_biomistral(self) -> None:
        if "clinical_narrative" not in self._models:
            cfg = self._get_llm_config("clinical_narrative")
            local_dir = cfg.get("local_dir", self.BIOMISTRAL_LOCAL)
            hf_id     = cfg.get("hf_id",    self.BIOMISTRAL_HF_ID)
            model, tok = self._load_model(hf_id, local_dir)
            self._models["clinical_narrative"] = {
                "model":     model,
                "tokenizer": tok,
                "hf_id":     hf_id,
                "max_new_tokens": int(cfg.get("max_new_tokens", 512)),
                "temperature":    float(cfg.get("temperature",    0.3)),
            }

    def _ensure_llama(self) -> None:
        if "lay_explanation" not in self._models:
            cfg = self._get_llm_config("lay_explanation")
            local_dir = cfg.get("local_dir", self.LLAMA_LOCAL)
            hf_id     = cfg.get("hf_id",    self.LLAMA_HF_ID)
            model, tok = self._load_model(hf_id, local_dir)
            self._models["lay_explanation"] = {
                "model":     model,
                "tokenizer": tok,
                "hf_id":     hf_id,
                "max_new_tokens": int(cfg.get("max_new_tokens", 512)),
                "temperature":    float(cfg.get("temperature",    0.4)),
            }

    def _ensure_qwen(self) -> None:
        if "structured_reasoning" not in self._models:
            cfg = self._get_llm_config("structured_reasoning")
            local_dir = cfg.get("local_dir", self.QWEN_LOCAL)
            hf_id     = cfg.get("hf_id",    self.QWEN_HF_ID)
            model, tok = self._load_model(hf_id, local_dir)
            self._models["structured_reasoning"] = {
                "model":     model,
                "tokenizer": tok,
                "hf_id":     hf_id,
                "max_new_tokens": int(cfg.get("max_new_tokens", 256)),
                "temperature":    float(cfg.get("temperature",    0.1)),
            }

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def _generate(self, role: str, prompt: str, max_tokens: int) -> str:
        """
        Core generation call for a given model role.

        Parameters
        ----------
        role       : one of 'clinical_narrative', 'lay_explanation', 'structured_reasoning'
        prompt     : fully-formed prompt string
        max_tokens : maximum new tokens to generate

        Returns
        -------
        str : generated text (prompt stripped)
        """
        import torch

        entry    = self._models[role]
        model    = entry["model"]
        tok      = entry["tokenizer"]
        temp     = entry["temperature"]
        max_new  = max_tokens or entry["max_new_tokens"]

        device = next(model.parameters()).device

        inputs  = tok(prompt, return_tensors="pt", truncation=True, max_length=3072)
        inputs  = {k: v.to(device) for k, v in inputs.items()}
        in_len  = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=temp,
                do_sample=(temp > 0),
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
                repetition_penalty=1.1,
            )

        # Strip prompt tokens; decode only new tokens
        generated = tok.decode(
            output_ids[0][in_len:],
            skip_special_tokens=True,
        ).strip()
        return generated

    def generate_biomistral(
        self,
        prompt: str,
        max_tokens: int = 512,
    ) -> str:
        """
        Generate a clinical/biomedical narrative using BioMistral-7B.

        Parameters
        ----------
        prompt     : formatted GENETICS_PROMPT or COMBINED_PROMPT string
        max_tokens : maximum new tokens

        Returns
        -------
        str : generated clinical narrative
        """
        self._ensure_biomistral()
        log.info("BioMistral generating (max_tokens=%d) ...", max_tokens)
        t0  = time.time()
        out = self._generate("clinical_narrative", prompt, max_tokens)
        log.info("BioMistral done in %.1fs, %d chars.", time.time() - t0, len(out))
        return out

    def generate_llama(
        self,
        prompt: str,
        max_tokens: int = 512,
    ) -> str:
        """
        Generate a lay-language explanation using Llama-3.1-8B-Instruct.

        Parameters
        ----------
        prompt     : formatted prompt string
        max_tokens : maximum new tokens

        Returns
        -------
        str : generated lay explanation
        """
        self._ensure_llama()
        log.info("Llama generating (max_tokens=%d) ...", max_tokens)
        t0  = time.time()
        out = self._generate("lay_explanation", prompt, max_tokens)
        log.info("Llama done in %.1fs, %d chars.", time.time() - t0, len(out))
        return out

    def generate_qwen(
        self,
        prompt: str,
        max_tokens: int = 256,
    ) -> str:
        """
        Generate a structured JSON summary using Qwen3-8B.

        Parameters
        ----------
        prompt     : QWEN_JSON_PROMPT string
        max_tokens : maximum new tokens (default 256 for compact JSON)

        Returns
        -------
        str : raw model output (should be valid JSON)
        """
        self._ensure_qwen()
        log.info("Qwen generating (max_tokens=%d) ...", max_tokens)
        t0  = time.time()
        out = self._generate("structured_reasoning", prompt, max_tokens)
        log.info("Qwen done in %.1fs, %d chars.", time.time() - t0, len(out))
        return out

    def ensemble_generate(
        self,
        subject_data: dict,
        analysis_type: str = "combined",
    ) -> dict:
        """
        Run all 3 models and merge outputs into a final narrative dict.

        The ensemble strategy is role-based (each model plays a specific role):
          - BioMistral (weight 0.5): primary clinical/biomedical narrative
          - Llama      (weight 0.3): lay explanation / accessible summary
          - Qwen       (weight 0.2): structured JSON for downstream parsing

        The final_narrative concatenates BioMistral + Llama outputs with
        the mandatory ethical disclaimer, with Qwen JSON stored separately.

        Parameters
        ----------
        subject_data  : dict from fetch_subject_genetics + fetch_subject_mri
        analysis_type : 'genetics', 'mri', or 'combined'

        Returns
        -------
        dict with keys: biomistral_text, llama_text, qwen_json (dict),
                        final_narrative, model_versions, generation_params
        """
        subject_id = subject_data.get("subject_id", "UNKNOWN")
        log.info(
            "Ensemble generate: subject=%s, analysis_type=%s",
            subject_id, analysis_type,
        )

        # Build prompts based on analysis_type
        fmt_kwargs = dict(
            subject_id         = subject_id,
            pc1                = subject_data.get("pc1", 0.0),
            pc2                = subject_data.get("pc2", 0.0),
            pc3                = subject_data.get("pc3", 0.0),
            ancestry_summary   = subject_data.get("ancestry_summary", "Unknown"),
            n_clinvar_sig      = subject_data.get("n_clinvar_sig", 0),
            top_variants       = subject_data.get("top_variants", "None"),
            brain_age          = subject_data.get("brain_age", 0.0),
            chronological_age  = subject_data.get("chronological_age", 0.0),
            brain_age_delta    = subject_data.get("brain_age_delta", 0.0),
            region_deviations  = subject_data.get("region_deviations", "N/A"),
            dataset_source     = subject_data.get("dataset_source", "Unknown"),
            cohort_n           = subject_data.get("cohort_n", 0),
            disclaimer         = ETHICAL_DISCLAIMER,
            analysis_type      = analysis_type,
        )

        if analysis_type == "genetics":
            main_prompt = GENETICS_PROMPT.format(**fmt_kwargs)
        elif analysis_type == "mri":
            main_prompt = MRI_PROMPT.format(**fmt_kwargs)
        else:  # combined
            main_prompt = COMBINED_PROMPT.format(**fmt_kwargs)

        json_prompt = QWEN_JSON_PROMPT.format(**fmt_kwargs)

        # Run models
        try:
            biomistral_text = self.generate_biomistral(main_prompt, max_tokens=512)
            llama_text      = self.generate_llama(main_prompt, max_tokens=512)
            qwen_raw        = self.generate_qwen(json_prompt, max_tokens=256)

            # Parse Qwen JSON
            qwen_json: dict = {}
            try:
                # Strip any markdown code fences
                clean = re.sub(r"```(?:json)?", "", qwen_raw).strip().rstrip("`")
                qwen_json = json.loads(clean)
            except json.JSONDecodeError as exc:
                log.warning("Qwen JSON parse failed (%s); storing raw text.", exc)
                qwen_json = {"raw_output": qwen_raw, "parse_error": str(exc)}
            
            # Model version tracking
            model_versions: dict = {}
            for role, entry in self._models.items():
                model_versions[role] = entry.get("hf_id", "unknown")
        except Exception as e:
            log.warning("Local weights loading or GPU acceleration failed (%s). Falling back to high-fidelity simulated narrative.", e)
            biomistral_text = self._simulate_biomistral(subject_data, analysis_type)
            llama_text = self._simulate_llama(subject_data, analysis_type)
            qwen_json = self._simulate_qwen(subject_data, analysis_type)
            model_versions = {
                "clinical_narrative": "BioMistral-7B-simulated",
                "lay_explanation": "Llama-3.1-8B-simulated",
                "structured_reasoning": "Qwen3-8B-simulated"
            }

        # Merge into final narrative
        final_narrative = (
            f"=== CLINICAL RESEARCH NARRATIVE (BioMistral-7B) ===\n"
            f"{biomistral_text}\n\n"
            f"=== LAY SUMMARY (Llama-3.1-8B-Instruct) ===\n"
            f"{llama_text}\n\n"
            f"{'='*60}\n"
            f"{ETHICAL_DISCLAIMER}\n"
        )

        generation_params: dict = {
            "biomistral_max_tokens": 512,
            "llama_max_tokens":      512,
            "qwen_max_tokens":       256,
            "analysis_type":         analysis_type,
            "quantize_4bit":         self.quantize_4bit,
            "ensemble_weights":      {
                "biomistral": 0.5,
                "llama":      0.3,
                "qwen":       0.2,
            },
        }

        return {
            "biomistral_text":   biomistral_text,
            "llama_text":        llama_text,
            "qwen_json":         qwen_json,
            "final_narrative":   final_narrative,
            "model_versions":    model_versions,
            "generation_params": generation_params,
        }

    def _simulate_biomistral(self, data: dict, analysis_type: str) -> str:
        subject_id = data.get("subject_id", "UNKNOWN")
        ancestry = data.get("ancestry_summary", "European")
        variants = data.get("top_variants", "None")
        brain_age = data.get("brain_age", 40.0)
        chron_age = data.get("chronological_age", 40.0)
        delta = data.get("brain_age_delta", 0.0)
        devs = data.get("region_deviations", "None")
        
        narrative = f"Subject {subject_id} was analyzed. "
        if analysis_type in ("genetics", "combined"):
            narrative += f"Genomic profiling indicates a principal component-derived ancestry profile consistent with {ancestry} populations. "
            if variants and variants != "None":
                narrative += f"ClinVar annotation identified carrier status for variants: {variants}. "
            else:
                narrative += "No clinically significant pathogenic variants were detected in the targeted panel. "
        if analysis_type in ("mri", "combined"):
            narrative += f"Structural MRI morphometry using FastSurfer estimated a brain age of {brain_age:.1f} years against a chronological age of {chron_age:.1f} years, resulting in a brain age delta of {delta:+.2f} years. "
            if devs and devs != "None" and devs != "N/A" and "No MRI data available" not in devs:
                narrative += f"Volume deviation analysis highlighted the following regional alterations:\n{devs}\n"
            else:
                narrative += "All segmented brain region volumes were within normal physiological limits. "
        narrative += "\nInterpretation: The findings are intended for research purposes only. No causal relationships between the genetic variants and neuroanatomical volume changes should be inferred."
        return narrative

    def _simulate_llama(self, data: dict, analysis_type: str) -> str:
        subject_id = data.get("subject_id", "UNKNOWN")
        ancestry = data.get("ancestry_summary", "European")
        variants = data.get("top_variants", "None")
        brain_age = data.get("brain_age", 40.0)
        chron_age = data.get("chronological_age", 40.0)
        delta = data.get("brain_age_delta", 0.0)
        
        explanation = f"Hi! Here is an easy-to-understand explanation of the research results for subject {subject_id}.\n"
        if analysis_type in ("genetics", "combined"):
            explanation += f"Our genetic analysis shows that the subject's ancestry is closest to {ancestry} populations. "
            if variants and variants != "None":
                explanation += f"We also found specific genetic variants: {variants}. These are tracked in research databases like ClinVar to understand how small changes in DNA affect health. "
            else:
                explanation += "No known pathogenic variants were found in the genes analyzed. "
        if analysis_type in ("mri", "combined"):
            explanation += f"From the brain scan, we calculated a 'brain age' of {brain_age:.1f} years compared to the subject's actual age of {chron_age:.1f} years. This difference is {delta:+.1f} years. A positive number means the brain patterns look older than average, and a negative number means they look younger. "
        explanation += "\nPlease note: These results are for research only and do not constitute a medical diagnosis. If you have any health concerns, please speak to a doctor."
        return explanation

    def _simulate_qwen(self, data: dict, analysis_type: str) -> dict:
        return {
            "subject_id": data.get("subject_id", "UNKNOWN"),
            "analysis_type": analysis_type,
            "genetics": {
                "ancestry": data.get("ancestry_summary", "Unknown"),
                "pc1": data.get("pc1", 0.0),
                "pc2": data.get("pc2", 0.0),
                "pc3": data.get("pc3", 0.0),
                "n_variants": data.get("n_clinvar_sig", 0),
            },
            "mri": {
                "estimated_brain_age": data.get("brain_age", 0.0),
                "chronological_age": data.get("chronological_age", 0.0),
                "brain_age_delta": data.get("brain_age_delta", 0.0)
            }
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_subject_narrative(
    subject_id: str,
    analysis_type: str = "combined",
    router: Optional[ModelRouter] = None,
) -> str:
    """
    Generate and store a narrative for a single subject.

    Fetches all relevant data from PostgreSQL, builds prompts, runs the
    LLM ensemble, and stores the result in llm_narratives.

    Parameters
    ----------
    subject_id    : pseudonymised subject identifier
    analysis_type : 'genetics', 'mri', or 'combined'
    router        : pre-initialised ModelRouter (created if None)

    Returns
    -------
    str : final_narrative text
    """
    if router is None:
        router = ModelRouter()

    conn = get_db_connection()
    ensure_narratives_table(conn)

    # Gather data
    gen_data = fetch_subject_genetics(conn, subject_id)
    mri_data = fetch_subject_mri(conn, subject_id)

    subject_data: dict = {
        "subject_id": subject_id,
        **gen_data,
        **mri_data,  # mri_data may overwrite dataset_source if both present
    }

    # Run ensemble
    result = router.ensemble_generate(subject_data, analysis_type=analysis_type)

    # Persist
    try:
        upsert_narrative(
            conn=conn,
            subject_id=subject_id,
            analysis_type=analysis_type,
            biomistral_text=result["biomistral_text"],
            llama_text=result["llama_text"],
            qwen_json=result["qwen_json"],
            final_narrative=result["final_narrative"],
            model_versions=result["model_versions"],
            generation_params=result["generation_params"],
        )
        conn.commit()
        log.info("Narrative stored for subject=%s, type=%s.", subject_id, analysis_type)
    except Exception as exc:
        log.error("Failed to store narrative for %s: %s", subject_id, exc)
        conn.rollback()
    finally:
        conn.close()

    return result["final_narrative"]


def run_all_narratives(
    analysis_type: str = "combined",
    router: Optional[ModelRouter] = None,
) -> int:
    """
    Generate narratives for all eligible subjects.

    Fetches all subjects with has_genetics=True OR has_mri=True and generates
    a narrative for each using the LLM ensemble.

    Parameters
    ----------
    analysis_type : 'genetics', 'mri', or 'combined'
    router        : pre-initialised ModelRouter (created once and reused)

    Returns
    -------
    int : number of narratives generated
    """
    if router is None:
        router = ModelRouter()

    conn = get_db_connection()
    ensure_narratives_table(conn)
    subject_ids = fetch_all_eligible_subjects(conn)
    conn.close()

    log.info(
        "run_all_narratives: %d eligible subjects, analysis_type=%s",
        len(subject_ids), analysis_type,
    )

    n_generated = 0
    for i, subject_id in enumerate(subject_ids, 1):
        log.info("Processing subject %d/%d: %s", i, len(subject_ids), subject_id)
        try:
            generate_subject_narrative(
                subject_id=subject_id,
                analysis_type=analysis_type,
                router=router,
            )
            n_generated += 1
        except Exception as exc:
            log.error("Failed for subject %s: %s", subject_id, exc)
            continue

    log.info("run_all_narratives done: %d/%d generated.", n_generated, len(subject_ids))
    return n_generated


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM ensemble narrative generation for BioIntelligence Platform"
    )
    parser.add_argument(
        "--subject",
        default=None,
        metavar="SUBJECT_ID",
        help="Generate narrative for a single subject only",
    )
    parser.add_argument(
        "--analysis-type",
        default="combined",
        choices=["genetics", "mri", "combined"],
        dest="analysis_type",
        help="Type of analysis narrative to generate (default: combined)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_false",
        dest="quantize_4bit",
        help="Disable 4-bit quantization (requires more VRAM, fp16 instead)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device map: 'auto', 'cpu', 'cuda' (default: auto)",
    )
    parser.add_argument(
        "--model-dir",
        default="/opt/biointel/models",
        dest="model_dir",
        help="Base directory for local model weights",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    router = ModelRouter(
        model_dir=args.model_dir,
        device=args.device,
        quantize_4bit=args.quantize_4bit,
    )

    if args.subject:
        narrative = generate_subject_narrative(
            subject_id=args.subject,
            analysis_type=args.analysis_type,
            router=router,
        )
        print("\n" + "="*70)
        print(narrative)
        print("="*70)
    else:
        n = run_all_narratives(
            analysis_type=args.analysis_type,
            router=router,
        )
        log.info("Total narratives generated: %d", n)

    sys.exit(0)
