# =============================================================================
# dag_utils.py — Shared helpers for all BioIntelligence Airflow DAGs
#
# DB connection: host=postgres, port=5432, db=biointel, user=biointel
# Airflow connection id: biointel_postgres
# =============================================================================

import os
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import psycopg2
from psycopg2.extras import execute_values, RealDictCursor

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection configuration
# ---------------------------------------------------------------------------
_DB_HOST = os.getenv("BIOINTEL_DB_HOST", "postgres")
_DB_PORT = int(os.getenv("BIOINTEL_DB_PORT", "5432"))
_DB_NAME = os.getenv("BIOINTEL_DB_NAME", "biointel")
_DB_USER = os.getenv("BIOINTEL_DB_USER", "biointel")
_DB_PASS = os.getenv("BIOINTEL_DB_PASSWORD", "biointel")

# Fallback: parse from SQLAlchemy connection string if the env var is set
# (Airflow sets AIRFLOW__DATABASE__SQL_ALCHEMY_CONN for *its* metadata DB,
#  but we keep our own dedicated vars above for the biointel data DB.)
_SQLALCHEMY_CONN = os.getenv("AIRFLOW__DATABASE__SQL_ALCHEMY_CONN", "")


def _parse_dsn_from_sqlalchemy(conn_str: str) -> Optional[str]:
    """
    Convert a SQLAlchemy DSN such as:
        postgresql+psycopg2://user:pass@host:port/dbname
    to a psycopg2-compatible DSN string.
    Returns None if the string is empty or cannot be parsed.
    """
    if not conn_str:
        return None
    try:
        dsn = conn_str.replace("postgresql+psycopg2://", "postgresql://")
        # psycopg2.connect() accepts libpq-style URIs that start with postgresql://
        return dsn
    except Exception:
        return None


def get_pg_conn() -> psycopg2.extensions.connection:
    """
    Return a raw psycopg2 connection to the biointel database.

    Connection priority:
    1. Individual BIOINTEL_DB_* environment variables (recommended)
    2. Fallback to parsing AIRFLOW__DATABASE__SQL_ALCHEMY_CONN (dev convenience)
    3. Hard-coded defaults (local / docker-compose)

    The caller is responsible for closing the connection (or use it as a
    context manager with a 'with' block).
    """
    dsn_from_env = _parse_dsn_from_sqlalchemy(_SQLALCHEMY_CONN)

    # Prefer explicit env vars; fall back to parsed SQLAlchemy DSN
    if _DB_PASS and _DB_USER:
        conn = psycopg2.connect(
            host=_DB_HOST,
            port=_DB_PORT,
            dbname=_DB_NAME,
            user=_DB_USER,
            password=_DB_PASS,
            connect_timeout=10,
            application_name="biointel_airflow",
        )
    elif dsn_from_env:
        conn = psycopg2.connect(dsn_from_env, connect_timeout=10)
    else:
        raise RuntimeError(
            "Cannot build a database connection: set BIOINTEL_DB_PASSWORD "
            "or AIRFLOW__DATABASE__SQL_ALCHEMY_CONN."
        )

    conn.autocommit = False
    log.debug(
        "Opened psycopg2 connection to %s:%s/%s as %s",
        _DB_HOST, _DB_PORT, _DB_NAME, _DB_USER,
    )
    return conn


# ---------------------------------------------------------------------------
# Pipeline run lifecycle helpers
# ---------------------------------------------------------------------------

def log_pipeline_start(dag_id: str, config: Optional[Dict[str, Any]] = None) -> str:
    """
    Insert a new row into pipeline_runs with status='running'.

    Parameters
    ----------
    dag_id : str
        The Airflow DAG identifier.
    config : dict, optional
        Arbitrary JSONB config/context to store alongside the run.

    Returns
    -------
    str
        The UUID run_id of the newly created record (as a string).
    """
    run_id = str(uuid.uuid4())
    sql = """
        INSERT INTO pipeline_runs (run_id, dag_id, run_type, status, started_at, config)
        VALUES (%s, %s, 'airflow', 'running', NOW(), %s)
        ON CONFLICT (run_id) DO NOTHING
        RETURNING run_id;
    """
    config_json = json.dumps(config) if config else None
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (run_id, dag_id, config_json))
            result = cur.fetchone()
            if result:
                run_id = str(result[0])
        conn.commit()
        log.info("pipeline_runs: started run_id=%s for dag_id=%s", run_id, dag_id)
    except Exception:
        conn.rollback()
        log.exception("Failed to log pipeline start for dag_id=%s", dag_id)
        raise
    finally:
        conn.close()
    return run_id


def log_pipeline_complete(run_id: str, n_subjects: int = 0) -> None:
    """
    Mark a pipeline_runs row as successfully completed.

    Parameters
    ----------
    run_id : str
        The UUID returned by log_pipeline_start().
    n_subjects : int
        Number of subjects processed in this run.
    """
    sql = """
        UPDATE pipeline_runs
        SET    status               = 'success',
               completed_at        = NOW(),
               n_subjects_processed = %s
        WHERE  run_id = %s;
    """
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (n_subjects, run_id))
        conn.commit()
        log.info(
            "pipeline_runs: completed run_id=%s (n_subjects=%d)",
            run_id, n_subjects,
        )
    except Exception:
        conn.rollback()
        log.exception("Failed to log pipeline completion for run_id=%s", run_id)
        raise
    finally:
        conn.close()


def log_pipeline_failed(run_id: str, error_message: str) -> None:
    """
    Mark a pipeline_runs row as failed and record the error message.

    Parameters
    ----------
    run_id : str
        The UUID returned by log_pipeline_start().
    error_message : str
        Human-readable description of what went wrong.
    """
    sql = """
        UPDATE pipeline_runs
        SET    status        = 'failed',
               completed_at  = NOW(),
               error_message = %s
        WHERE  run_id = %s;
    """
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (error_message[:4096], run_id))
        conn.commit()
        log.error(
            "pipeline_runs: failed run_id=%s — %s",
            run_id, error_message[:200],
        )
    except Exception:
        conn.rollback()
        log.exception("Failed to log pipeline failure for run_id=%s", run_id)
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Generic bulk-insert helper
# ---------------------------------------------------------------------------

def bulk_insert(
    conn: psycopg2.extensions.connection,
    table: str,
    columns: Sequence[str],
    rows: List[Tuple[Any, ...]],
    on_conflict: str = "DO NOTHING",
    page_size: int = 1000,
) -> int:
    """
    Generic bulk insert using psycopg2's execute_values for high throughput.

    Parameters
    ----------
    conn : psycopg2 connection
        An *open* connection (autocommit=False).  The caller must commit/rollback.
    table : str
        Target table name (optionally schema-qualified, e.g. 'public.variants').
    columns : sequence of str
        Column names in the same order as the data tuples.
    rows : list of tuple
        Data rows to insert.
    on_conflict : str
        Conflict resolution clause appended after ON CONFLICT, e.g.
        'DO NOTHING' or '(subject_id, variant_id) DO UPDATE SET gq = EXCLUDED.gq'.
    page_size : int
        Number of rows per VALUES batch (default 1000).

    Returns
    -------
    int
        Number of rows in the ``rows`` list (not necessarily inserted, since
        ON CONFLICT … DO NOTHING silently skips duplicates).

    Raises
    ------
    ValueError
        If ``rows`` is empty or ``columns`` is empty.
    psycopg2.DatabaseError
        Propagated from the database driver on SQL errors.
    """
    if not columns:
        raise ValueError("columns must not be empty")
    if not rows:
        log.debug("bulk_insert: no rows to insert into %s", table)
        return 0

    col_list = ", ".join(columns)
    sql = (
        f"INSERT INTO {table} ({col_list}) "
        f"VALUES %s "
        f"ON CONFLICT {on_conflict}"
    )

    log.debug(
        "bulk_insert: inserting %d rows into %s (page_size=%d)",
        len(rows), table, page_size,
    )
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=page_size)

    return len(rows)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def fetch_all(
    conn: psycopg2.extensions.connection,
    sql: str,
    params: Optional[Tuple[Any, ...]] = None,
) -> List[Dict[str, Any]]:
    """
    Execute a SELECT and return a list of dicts (column name → value).

    Parameters
    ----------
    conn : psycopg2 connection
        An open connection.
    sql : str
        The SQL query to execute.
    params : tuple, optional
        Query parameters to pass to cur.execute().

    Returns
    -------
    list of dict
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def fetch_one(
    conn: psycopg2.extensions.connection,
    sql: str,
    params: Optional[Tuple[Any, ...]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Execute a SELECT and return the first row as a dict, or None.
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None


def execute_sql(
    conn: psycopg2.extensions.connection,
    sql: str,
    params: Optional[Tuple[Any, ...]] = None,
) -> int:
    """
    Execute a non-SELECT statement (INSERT / UPDATE / DELETE).

    Returns
    -------
    int
        cur.rowcount after execution.
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.rowcount


# ---------------------------------------------------------------------------
# Airflow XCom convenience
# ---------------------------------------------------------------------------

def push_run_id(context: Dict[str, Any], run_id: str) -> None:
    """Push run_id to XCom so downstream tasks can retrieve it."""
    context["ti"].xcom_push(key="run_id", value=run_id)


def pull_run_id(context: Dict[str, Any]) -> Optional[str]:
    """Pull run_id from XCom pushed by an upstream task."""
    return context["ti"].xcom_pull(key="run_id")
