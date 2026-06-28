"""
superset_api.py -- Superset REST API client for BioIntelligence Platform

Creates/updates all 5 dashboards after pipeline runs.
Uses Superset REST API v1 (no paid service, no external APIs).

Dashboards:
  1. Ancestry x Brain   -- PC1/PC2 scatter + PC1-10 vs brain region heatmap
  2. Variants           -- per-chromosome bar, gnomAD AF distribution, top pathogenic table
  3. DNA Models         -- DNABERT-2/HyenaDNA embedding PCA cluster + pred_score distribution
  4. Pipeline Status    -- pipeline_runs timeline + success/failure pie
  5. Brain Age          -- predicted vs actual scatter + brain_age_delta histogram

Usage:
    python superset_api.py --db-url postgresql://user:pass@host/dbname
    python superset_api.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("superset_api")

# ---------------------------------------------------------------------------
# Defaults (overridable via env vars or CLI)
# ---------------------------------------------------------------------------
SUPERSET_BASE_URL: str = os.environ.get("SUPERSET_URL", "http://localhost:8088")
SUPERSET_USERNAME: str = os.environ.get("SUPERSET_USER", "admin")
SUPERSET_PASSWORD: str = os.environ.get("SUPERSET_PASS", "admin")

DEFAULT_SCHEMA: str = "public"
REQUEST_TIMEOUT: int = 30  # seconds

# ---------------------------------------------------------------------------
# Dashboard / chart configuration
# ---------------------------------------------------------------------------

DASHBOARD_CONFIGS: Dict[str, dict] = {
    # ------------------------------------------------------------------
    # Dashboard 1: Ancestry x Brain
    # ------------------------------------------------------------------
    "ancestry_brain": {
        "title": "Ancestry x Brain — PC Scatter & Region Heatmap",
        "slug": "ancestry-brain",
        "charts": [
            {
                "slice_name": "Ancestry PC1 vs PC2 (colored by total brain volume)",
                "viz_type": "scatter",
                "description": (
                    "Scatter of ancestry PC1 (x) vs PC2 (y) coloured by total_brain_volume. "
                    "PCs are continuous mathematical scores capturing genetic similarity -- "
                    "NOT racial categories."
                ),
                "query_context": {
                    "datasource": {"type": "table"},
                    "queries": [
                        {
                            "columns": ["pc1", "pc2"],
                            "metrics": ["total_brain_volume"],
                            "filters": [],
                            "row_limit": 50000,
                        }
                    ],
                },
                "params": json.dumps({
                    "x_axis": "pc1",
                    "y_axis": "pc2",
                    "color_metric": "total_brain_volume",
                    "x_axis_label": "PC1 (ancestry dimension 1)",
                    "y_axis_label": "PC2 (ancestry dimension 2)",
                    "rich_tooltip": True,
                    "show_legend": True,
                    "opacity": 0.7,
                }),
                "table": "subject_pcs_brain_joined",
            },
            {
                "slice_name": "PC1-10 vs Top 10 Brain Regions — Correlation Heatmap",
                "viz_type": "heatmap",
                "description": (
                    "Pearson correlation heatmap: ancestry PCs 1-10 (rows) vs top 10 brain "
                    "regions by inter-subject variance (columns). Effect sizes below |r|<0.1 "
                    "are negligible."
                ),
                "params": json.dumps({
                    "x_axis": "region_name",
                    "y_axis": "pc_name",
                    "metric": "pearson_r",
                    "linear_color_scheme": "rdbu",
                    "xscale_interval": 1,
                    "yscale_interval": 1,
                    "canvas_image_rendering": "pixelated",
                    "normalize_across": "heatmap",
                    "left_margin": 100,
                }),
                "table": "pc_brain_correlation_matrix",
            },
        ],
    },

    # ------------------------------------------------------------------
    # Dashboard 2: Variants
    # ------------------------------------------------------------------
    "variants": {
        "title": "Variants — Chromosome Distribution, AF, Pathogenic Table",
        "slug": "variants-overview",
        "charts": [
            {
                "slice_name": "Variant Count by Chromosome",
                "viz_type": "bar",
                "description": "Number of variants per chromosome. Ordered 1-22, X, Y, MT.",
                "params": json.dumps({
                    "metrics": ["COUNT(*)"],
                    "groupby": ["chrom"],
                    "color_scheme": "supersetColors",
                    "show_legend": True,
                    "rich_tooltip": True,
                    "x_axis_label": "Chromosome",
                    "y_axis_label": "Variant count",
                    "bar_stacked": False,
                }),
                "table": "variants",
            },
            {
                "slice_name": "gnomAD Allele Frequency Distribution",
                "viz_type": "histogram",
                "description": (
                    "Distribution of gnomad_af (log10 scale) across all variants. "
                    "Rare variants (AF<0.01) are common in research cohorts."
                ),
                "params": json.dumps({
                    "column": "gnomad_af",
                    "bins": 50,
                    "x_axis_label": "gnomAD AF (log10)",
                    "y_axis_label": "Variant count",
                    "log_scale": True,
                    "cumulative": False,
                }),
                "table": "variants",
            },
            {
                "slice_name": "Top Pathogenic Variants (ClinVar)",
                "viz_type": "table",
                "description": (
                    "Table of variants with ClinVar significance = Pathogenic or Likely pathogenic, "
                    "ordered by gnomad_af ascending (rarest first)."
                ),
                "params": json.dumps({
                    "columns": [
                        "variant_id", "gene_symbol", "chrom", "pos",
                        "ref", "alt", "clinvar_sig", "gnomad_af",
                    ],
                    "metrics": [],
                    "filters": [
                        {
                            "col": "clinvar_sig",
                            "op": "IN",
                            "val": ["Pathogenic", "Likely pathogenic"],
                        }
                    ],
                    "order_by": [["gnomad_af", True]],
                    "row_limit": 100,
                    "include_search": True,
                    "page_length": 25,
                }),
                "table": "variants",
            },
        ],
    },

    # ------------------------------------------------------------------
    # Dashboard 3: DNA Models
    # ------------------------------------------------------------------
    "dna_models": {
        "title": "DNA Models — DNABERT-2 & HyenaDNA Embedding Analysis",
        "slug": "dna-models",
        "charts": [
            {
                "slice_name": "Embedding Cluster PCA (DNABERT-2 vs HyenaDNA)",
                "viz_type": "scatter",
                "description": (
                    "PCA projection of DNA model embeddings. Shape = model_name, "
                    "colour = pred_label (pathogenic/benign). "
                    "Clustering indicates learned sequence feature separation."
                ),
                "params": json.dumps({
                    "x_axis": "pca_dim1",
                    "y_axis": "pca_dim2",
                    "color_metric": "pred_label",
                    "entity": "model_name",
                    "x_axis_label": "PCA dim 1",
                    "y_axis_label": "PCA dim 2",
                    "rich_tooltip": True,
                    "show_legend": True,
                    "opacity": 0.8,
                }),
                "table": "dna_model_predictions_pca",
            },
            {
                "slice_name": "Prediction Score Distribution by Model",
                "viz_type": "box_plot",
                "description": (
                    "Distribution of pred_score [0,1] (pathogenic probability via cosine "
                    "similarity) split by model_name and pred_label."
                ),
                "params": json.dumps({
                    "x": "model_name",
                    "columns": ["pred_score"],
                    "groupby": ["pred_label"],
                    "color_scheme": "supersetColors",
                    "x_axis_label": "Model",
                    "y_axis_label": "Prediction score",
                    "whisker_options": "Tukey",
                }),
                "table": "dna_model_predictions",
            },
        ],
    },

    # ------------------------------------------------------------------
    # Dashboard 4: Pipeline Status
    # ------------------------------------------------------------------
    "pipeline_status": {
        "title": "Pipeline Status — Run History & Success Rate",
        "slug": "pipeline-status",
        "charts": [
            {
                "slice_name": "Pipeline Runs Timeline",
                "viz_type": "line",
                "description": (
                    "Timeline of pipeline_runs: number of runs per day coloured by "
                    "pipeline_name. Hover for duration and record counts."
                ),
                "params": json.dumps({
                    "x_axis": "started_at",
                    "metrics": ["COUNT(*)"],
                    "groupby": ["pipeline_name"],
                    "time_grain_sqla": "P1D",
                    "rich_tooltip": True,
                    "show_legend": True,
                    "x_axis_label": "Date",
                    "y_axis_label": "Runs per day",
                }),
                "table": "pipeline_runs",
            },
            {
                "slice_name": "Pipeline Success / Failure Ratio",
                "viz_type": "pie",
                "description": (
                    "Pie chart of pipeline run outcomes: success vs failure vs running. "
                    "Breakdown by status field."
                ),
                "params": json.dumps({
                    "metric": "COUNT(*)",
                    "groupby": ["status"],
                    "color_scheme": "supersetColors",
                    "show_legend": True,
                    "show_labels": True,
                    "labels_outside": True,
                    "label_type": "key_percent",
                    "donut": True,
                    "innerRadius": 40,
                }),
                "table": "pipeline_runs",
            },
        ],
    },

    # ------------------------------------------------------------------
    # Dashboard 5: Brain Age
    # ------------------------------------------------------------------
    "brain_age": {
        "title": "Brain Age — Predicted vs Actual & Delta Distribution",
        "slug": "brain-age",
        "charts": [
            {
                "slice_name": "Predicted vs Chronological Age (with regression line)",
                "viz_type": "scatter",
                "description": (
                    "Scatter of predicted_brain_age (y) vs chronological_age (x). "
                    "Points coloured by brain_age_delta. Identity line shown. "
                    "RESEARCH USE ONLY -- not a clinical diagnostic tool."
                ),
                "params": json.dumps({
                    "x_axis": "chronological_age",
                    "y_axis": "predicted_brain_age",
                    "color_metric": "brain_age_delta",
                    "regression_line": True,
                    "x_axis_label": "Chronological age (years)",
                    "y_axis_label": "Predicted brain age (years)",
                    "rich_tooltip": True,
                    "show_legend": True,
                    "opacity": 0.65,
                }),
                "table": "brain_age_predictions",
            },
            {
                "slice_name": "Brain Age Delta Distribution",
                "viz_type": "histogram",
                "description": (
                    "Histogram of brain_age_delta = predicted_brain_age - chronological_age. "
                    "Positive delta = accelerated ageing (research interpretation only). "
                    "95% CI band shown."
                ),
                "params": json.dumps({
                    "column": "brain_age_delta",
                    "bins": 40,
                    "x_axis_label": "Brain age delta (years)",
                    "y_axis_label": "Subject count",
                    "show_ci": True,
                    "ci_color": "#ff7f0e",
                }),
                "table": "brain_age_predictions",
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# SupersetClient
# ---------------------------------------------------------------------------

class SupersetClient:
    """
    REST API client for Apache Superset v1.

    Manages JWT authentication, CSRF token rotation, dataset/chart/dashboard
    creation and idempotent updates.

    Parameters
    ----------
    base_url : Superset base URL, e.g. 'http://localhost:8088'
    username : Superset admin username
    password : Superset admin password
    """

    def __init__(
        self,
        base_url: str = SUPERSET_BASE_URL,
        username: str = SUPERSET_USERNAME,
        password: str = SUPERSET_PASSWORD,
    ) -> None:
        self.base_url   = base_url.rstrip("/")
        self.username   = username
        self.password   = password
        self._jwt_token: Optional[str] = None
        self._csrf_token: Optional[str] = None
        self._session   = requests.Session()
        self._session.timeout = REQUEST_TIMEOUT

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def login(self) -> None:
        """
        Authenticate with Superset and obtain a JWT bearer token.

        Calls POST /api/v1/security/login. Stores the access_token internally.
        Raises requests.HTTPError on failure.
        """
        url  = f"{self.base_url}/api/v1/security/login"
        body = {
            "username": self.username,
            "password": self.password,
            "provider": "db",
            "refresh":  True,
        }
        log.info("Authenticating with Superset at %s ...", url)
        resp = self._session.post(url, json=body, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        self._jwt_token = data["access_token"]
        log.info("Superset login successful.")

    def get_csrf_token(self) -> str:
        """
        Fetch a fresh CSRF token from Superset.

        Calls GET /api/v1/security/csrf_token/ with the JWT bearer header.
        Stores and returns the csrf_token string.
        """
        if self._jwt_token is None:
            self.login()
        url  = f"{self.base_url}/api/v1/security/csrf_token/"
        resp = self._session.get(url, headers=self._headers(csrf=False), timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        self._csrf_token = resp.json()["result"]
        log.debug("CSRF token refreshed.")
        return self._csrf_token

    def _headers(self, csrf: bool = True) -> dict:
        """
        Return HTTP headers for authenticated Superset API calls.

        Parameters
        ----------
        csrf : whether to include the X-CSRFToken header

        Returns
        -------
        dict of header key-value pairs
        """
        if self._jwt_token is None:
            self.login()
        headers: dict = {
            "Authorization": f"Bearer {self._jwt_token}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }
        if csrf:
            if self._csrf_token is None:
                self.get_csrf_token()
            headers["X-CSRFToken"] = self._csrf_token
        return headers

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        """Authenticated GET against a Superset API path."""
        resp = self._session.get(
            f"{self.base_url}{path}",
            headers=self._headers(csrf=False),
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        """Authenticated POST against a Superset API path."""
        self.get_csrf_token()  # always refresh CSRF before mutating calls
        resp = self._session.post(
            f"{self.base_url}{path}",
            headers=self._headers(csrf=True),
            json=body,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def _put(self, path: str, body: dict) -> dict:
        """Authenticated PUT against a Superset API path."""
        self.get_csrf_token()
        resp = self._session.put(
            f"{self.base_url}{path}",
            headers=self._headers(csrf=True),
            json=body,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str) -> None:
        """Authenticated DELETE against a Superset API path."""
        self.get_csrf_token()
        resp = self._session.delete(
            f"{self.base_url}{path}",
            headers=self._headers(csrf=True),
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    def _find_database_id(self, sqlalchemy_uri: str) -> Optional[int]:
        """Return the Superset database_id for a given SQLAlchemy URI, or None."""
        data = self._get("/api/v1/database/", params={"q": '{"page_size": 100}'})
        for db in data.get("result", []):
            if db.get("sqlalchemy_uri", "") == sqlalchemy_uri:
                return int(db["id"])
        return None

    def _create_database(self, sqlalchemy_uri: str) -> int:
        """
        Register a new database connection in Superset.

        Returns the database_id.
        """
        body = {
            "database_name": "BioIntelligence PostgreSQL",
            "sqlalchemy_uri": sqlalchemy_uri,
            "expose_in_sqllab": True,
            "allow_run_async": True,
        }
        resp = self._post("/api/v1/database/", body)
        db_id = int(resp["id"])
        log.info("Database registered with id=%d.", db_id)
        return db_id

    def _get_or_create_database(self, sqlalchemy_uri: str) -> int:
        """Idempotent: get existing or create new Superset database record."""
        db_id = self._find_database_id(sqlalchemy_uri)
        if db_id is not None:
            log.info("Database already registered: id=%d.", db_id)
            return db_id
        return self._create_database(sqlalchemy_uri)

    def create_or_update_dataset(
        self,
        sqlalchemy_uri: str,
        table_name: str,
        schema: str = DEFAULT_SCHEMA,
    ) -> int:
        """
        Create a virtual Superset dataset for a given table.

        Checks if a dataset with the same table_name exists; updates if so,
        creates otherwise.

        Parameters
        ----------
        sqlalchemy_uri : SQLAlchemy connection string for the source database
        table_name     : name of the source table / view
        schema         : PostgreSQL schema (default 'public')

        Returns
        -------
        int : Superset dataset ID
        """
        database_id = self._get_or_create_database(sqlalchemy_uri)

        # Check if dataset already exists
        existing_id = self._find_dataset_id(table_name, schema)
        if existing_id is not None:
            log.info("Dataset '%s' already exists (id=%d); updating ...", table_name, existing_id)
            self._put(
                f"/api/v1/dataset/{existing_id}",
                {
                    "database":   database_id,
                    "table_name": table_name,
                    "schema":     schema,
                },
            )
            return existing_id

        body = {
            "database":   database_id,
            "table_name": table_name,
            "schema":     schema,
        }
        resp = self._post("/api/v1/dataset/", body)
        dataset_id = int(resp["id"])
        log.info("Dataset '%s' created with id=%d.", table_name, dataset_id)
        return dataset_id

    def _find_dataset_id(self, table_name: str, schema: str) -> Optional[int]:
        """Return dataset_id for given table+schema, or None if not found."""
        try:
            q = json.dumps({"filters": [{"col": "table_name", "opr": "eq", "value": table_name}]})
            data = self._get("/api/v1/dataset/", params={"q": q})
            for ds in data.get("result", []):
                if ds.get("schema") == schema or schema == DEFAULT_SCHEMA:
                    return int(ds["id"])
        except Exception as exc:
            log.warning("_find_dataset_id failed for '%s': %s", table_name, exc)
        return None

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------

    def _find_chart_id(self, slice_name: str) -> Optional[int]:
        """Return chart id for given slice_name, or None."""
        try:
            q = json.dumps({"filters": [{"col": "slice_name", "opr": "chart_all_text", "value": slice_name}]})
            data = self._get("/api/v1/chart/", params={"q": q})
            for chart in data.get("result", []):
                if chart.get("slice_name") == slice_name:
                    return int(chart["id"])
        except Exception as exc:
            log.warning("_find_chart_id failed for '%s': %s", slice_name, exc)
        return None

    def create_chart(self, chart_config: dict) -> int:
        """
        Create a new Superset chart.

        Parameters
        ----------
        chart_config : dict with keys: slice_name, viz_type, datasource_id,
                       datasource_type, params, description

        Returns
        -------
        int : chart (slice) ID
        """
        resp = self._post("/api/v1/chart/", chart_config)
        chart_id = int(resp["id"])
        log.info("Chart '%s' created (id=%d).", chart_config.get("slice_name"), chart_id)
        return chart_id

    def update_chart(self, chart_id: int, chart_config: dict) -> None:
        """
        Update an existing Superset chart.

        Parameters
        ----------
        chart_id     : existing chart ID
        chart_config : partial or full chart config dict
        """
        self._put(f"/api/v1/chart/{chart_id}", chart_config)
        log.info("Chart id=%d updated.", chart_id)

    def create_or_update_chart(
        self,
        chart_def: dict,
        dataset_id: int,
    ) -> int:
        """
        Idempotent chart creation.

        Parameters
        ----------
        chart_def  : chart definition dict from DASHBOARD_CONFIGS
        dataset_id : datasource dataset ID

        Returns
        -------
        int : chart ID
        """
        slice_name = chart_def["slice_name"]
        existing_id = self._find_chart_id(slice_name)

        chart_payload = {
            "slice_name":      slice_name,
            "viz_type":        chart_def["viz_type"],
            "datasource_id":   dataset_id,
            "datasource_type": "table",
            "params":          chart_def.get("params", "{}"),
            "description":     chart_def.get("description", ""),
        }

        if existing_id is not None:
            self.update_chart(existing_id, chart_payload)
            return existing_id
        return self.create_chart(chart_payload)

    # ------------------------------------------------------------------
    # Dashboards
    # ------------------------------------------------------------------

    def _build_layout(self, chart_ids: List[int]) -> dict:
        """
        Build a simple grid layout JSON for a Superset dashboard.

        Creates a single-column layout with one chart per row.

        Parameters
        ----------
        chart_ids : ordered list of chart IDs to include

        Returns
        -------
        dict : Superset dashboard layout specification
        """
        layout: dict = {
            "DASHBOARD_VERSION_KEY": "v2",
            "ROOT_ID":              {"type": "ROOT",   "id": "ROOT_ID",    "children": ["GRID_ID"]},
            "GRID_ID":              {"type": "GRID",   "id": "GRID_ID",    "children": []},
        }
        for i, chart_id in enumerate(chart_ids):
            row_id  = f"ROW-{i}"
            col_id  = f"COLUMN-{i}"
            item_id = f"CHART-{chart_id}"

            layout["GRID_ID"]["children"].append(row_id)
            layout[row_id] = {
                "type":     "ROW",
                "id":       row_id,
                "children": [col_id],
                "meta":     {"background": "BACKGROUND_TRANSPARENT"},
            }
            layout[col_id] = {
                "type":     "CHART",
                "id":       col_id,
                "children": [],
                "meta":     {
                    "chartId": chart_id,
                    "width":   12,
                    "height":  50,
                },
            }

        return layout

    def create_dashboard(
        self,
        title: str,
        slug: str,
        chart_ids: List[int],
    ) -> int:
        """
        Create a new Superset dashboard with the given charts.

        Parameters
        ----------
        title     : dashboard display title
        slug      : unique URL slug
        chart_ids : list of chart IDs to include

        Returns
        -------
        int : dashboard ID
        """
        layout = self._build_layout(chart_ids)
        body = {
            "dashboard_title":  title,
            "slug":             slug,
            "position_json":    json.dumps(layout),
            "published":        True,
        }
        resp = self._post("/api/v1/dashboard/", body)
        dash_id = int(resp["id"])
        log.info("Dashboard '%s' created (id=%d).", title, dash_id)
        return dash_id

    def update_dashboard(
        self,
        dashboard_id: int,
        chart_ids: List[int],
        title: Optional[str] = None,
        slug: Optional[str] = None,
    ) -> None:
        """
        Refresh an existing dashboard with updated chart list and layout.

        Parameters
        ----------
        dashboard_id : existing Superset dashboard ID
        chart_ids    : updated list of chart IDs
        title        : optional new title (unchanged if None)
        slug         : optional new slug (unchanged if None)
        """
        layout = self._build_layout(chart_ids)
        body: dict = {"position_json": json.dumps(layout)}
        if title:
            body["dashboard_title"] = title
        if slug:
            body["slug"] = slug
        self._put(f"/api/v1/dashboard/{dashboard_id}", body)
        log.info("Dashboard id=%d updated.", dashboard_id)

    def _find_dashboard_id(self, slug: str) -> Optional[int]:
        """Return dashboard ID by slug, or None."""
        try:
            q = json.dumps({"filters": [{"col": "slug", "opr": "eq", "value": slug}]})
            data = self._get("/api/v1/dashboard/", params={"q": q})
            for dash in data.get("result", []):
                if dash.get("slug") == slug:
                    return int(dash["id"])
        except Exception as exc:
            log.warning("_find_dashboard_id failed for slug '%s': %s", slug, exc)
        return None

    def create_or_update_dashboard(
        self,
        title: str,
        slug: str,
        chart_ids: List[int],
    ) -> int:
        """
        Idempotent dashboard creation/update.

        Parameters
        ----------
        title     : dashboard title
        slug      : unique URL slug
        chart_ids : chart IDs to include

        Returns
        -------
        int : dashboard ID
        """
        existing_id = self._find_dashboard_id(slug)
        if existing_id is not None:
            log.info("Dashboard slug='%s' already exists (id=%d); updating ...", slug, existing_id)
            self.update_dashboard(existing_id, chart_ids, title=title, slug=slug)
            return existing_id
        return self.create_dashboard(title, slug, chart_ids)

    def refresh_all_dashboards(self) -> None:
        """
        Trigger cache invalidation for all dashboards in Superset.

        Calls PUT /api/v1/dashboard/{id}/cache_dashboard_screenshot for each.
        Errors on individual dashboards are logged but do not abort the loop.
        """
        log.info("Refreshing cache for all dashboards ...")
        try:
            data = self._get("/api/v1/dashboard/", params={"q": '{"page_size": 200}'})
            dashboards = data.get("result", [])
        except Exception as exc:
            log.error("Could not list dashboards for cache refresh: %s", exc)
            return

        for dash in dashboards:
            dash_id = dash.get("id")
            try:
                self.get_csrf_token()
                resp = self._session.put(
                    f"{self.base_url}/api/v1/dashboard/{dash_id}/cache_dashboard_screenshot",
                    headers=self._headers(csrf=True),
                    timeout=REQUEST_TIMEOUT,
                )
                if resp.status_code not in (200, 202):
                    log.warning(
                        "Cache refresh for dashboard %d returned %d.",
                        dash_id, resp.status_code,
                    )
                else:
                    log.info("Cache refreshed for dashboard id=%d.", dash_id)
            except Exception as exc:
                log.warning("Cache refresh error for dashboard %d: %s", dash_id, exc)


# ---------------------------------------------------------------------------
# Main setup function
# ---------------------------------------------------------------------------

def setup_all_dashboards(
    db_url: str,
    base_url: str = SUPERSET_BASE_URL,
    username: str = SUPERSET_USERNAME,
    password: str = SUPERSET_PASSWORD,
) -> None:
    """
    Idempotent setup of all 5 BioIntelligence dashboards in Superset.

    Steps:
      1. Login to Superset.
      2. Register the PostgreSQL database connection (idempotent).
      3. For each dashboard config: create/update datasets, charts, dashboard.
      4. Trigger cache refresh.

    Safe to call multiple times -- will update existing resources rather
    than create duplicates.

    Parameters
    ----------
    db_url   : SQLAlchemy URI for the BioIntelligence PostgreSQL database
    base_url : Superset base URL
    username : Superset admin username
    password : Superset admin password
    """
    client = SupersetClient(base_url=base_url, username=username, password=password)
    client.login()
    client.get_csrf_token()

    log.info("Setting up all 5 BioIntelligence dashboards ...")

    for dash_key, dash_cfg in DASHBOARD_CONFIGS.items():
        log.info("--- Setting up dashboard: %s ---", dash_cfg["title"])
        chart_ids: List[int] = []

        for chart_def in dash_cfg["charts"]:
            table_name = chart_def.get("table", "variants")
            try:
                dataset_id = client.create_or_update_dataset(
                    sqlalchemy_uri=db_url,
                    table_name=table_name,
                    schema=DEFAULT_SCHEMA,
                )
            except Exception as exc:
                log.error(
                    "Failed to create dataset '%s': %s. Skipping chart '%s'.",
                    table_name, exc, chart_def["slice_name"],
                )
                continue

            try:
                chart_id = client.create_or_update_chart(chart_def, dataset_id)
                chart_ids.append(chart_id)
            except Exception as exc:
                log.error(
                    "Failed to create chart '%s': %s",
                    chart_def["slice_name"], exc,
                )
                continue

        if chart_ids:
            try:
                client.create_or_update_dashboard(
                    title=dash_cfg["title"],
                    slug=dash_cfg["slug"],
                    chart_ids=chart_ids,
                )
            except Exception as exc:
                log.error(
                    "Failed to create/update dashboard '%s': %s",
                    dash_cfg["title"], exc,
                )
        else:
            log.warning(
                "No charts created for dashboard '%s' -- skipping dashboard creation.",
                dash_cfg["title"],
            )

        # Brief pause to avoid overwhelming Superset
        time.sleep(0.5)

    # Refresh caches for all dashboards
    client.refresh_all_dashboards()
    log.info("All dashboards setup complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BioIntelligence Superset dashboard setup (REST API v1)"
    )
    parser.add_argument(
        "--db-url",
        required=True,
        dest="db_url",
        metavar="SQLALCHEMY_URI",
        help=(
            "SQLAlchemy URI for the BioIntelligence database, e.g. "
            "postgresql://biointel:biointel@localhost/biointel"
        ),
    )
    parser.add_argument(
        "--superset-url",
        default=SUPERSET_BASE_URL,
        dest="superset_url",
        help=f"Superset base URL (default: {SUPERSET_BASE_URL})",
    )
    parser.add_argument(
        "--username",
        default=SUPERSET_USERNAME,
        help=f"Superset admin username (default: {SUPERSET_USERNAME})",
    )
    parser.add_argument(
        "--password",
        default=SUPERSET_PASSWORD,
        help="Superset admin password",
    )
    parser.add_argument(
        "--refresh-only",
        action="store_true",
        dest="refresh_only",
        help="Only refresh dashboard caches; do not create/update anything",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.refresh_only:
        client = SupersetClient(
            base_url=args.superset_url,
            username=args.username,
            password=args.password,
        )
        client.login()
        client.refresh_all_dashboards()
    else:
        setup_all_dashboards(
            db_url=args.db_url,
            base_url=args.superset_url,
            username=args.username,
            password=args.password,
        )

    sys.exit(0)
