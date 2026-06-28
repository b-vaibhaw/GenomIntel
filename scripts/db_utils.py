"""
db_utils.py — SQLite database utility layer for BioIntelligence Platform
Provides SQLite connection context manager, SQLAlchemy SQLite engine,
and high-performance SQLite bulk INSERT/UPSERT operations.
"""

import os
import sqlite3
import math
import logging
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Iterator, Tuple
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

logger = logging.getLogger("db_utils")

DB_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "biointel.db")

class SqliteStdDev:
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

def get_db_url() -> str:
    """Return SQLite connection URI."""
    return f"sqlite:///{DB_FILE}"

def get_engine(pool_size=5):
    """Return SQLAlchemy engine for SQLite."""
    # SQLite doesn't support pool_size argument directly like Postgres.
    # We use create_engine with default pool or static pool.
    return create_engine(get_db_url(), connect_args={"timeout": 30})

@contextmanager
def get_session() -> Iterator[Session]:
    """Yield a new SQLAlchemy Session."""
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error("Session transaction failed, rolled back: %s", e)
        raise e
    finally:
        session.close()

class ContextCursor:
    def __init__(self, cur):
        self.cur = cur
    def __enter__(self):
        return self
    def __exit__(self, exc_type, val, tb):
        self.cur.close()
    def execute(self, sql, params=None):
        import re
        sql_clean = sql.replace("%s", "?")
        sql_clean = re.sub(r'(?i)uuid_generate_v4\(\)', "lower(hex(randomblob(4)) || '-' || hex(randomblob(2)) || '-4' || substr(hex(randomblob(2)),2,3) || '-' || substr('89ab',abs(random() % 4) + 1, 1) || substr(hex(randomblob(2)),2,3) || '-' || hex(randomblob(6)))", sql_clean)
        # Replace NOW() / now() with CURRENT_TIMESTAMP
        sql_clean = re.sub(r'(?i)\bnow\(\)', "CURRENT_TIMESTAMP", sql_clean)
        # Replace SERIAL PRIMARY KEY / serial primary key with INTEGER PRIMARY KEY AUTOINCREMENT
        sql_clean = re.sub(r'(?i)\bserial\s+primary\s+key\b', "INTEGER PRIMARY KEY AUTOINCREMENT", sql_clean)
        if params is not None:
            if isinstance(params, dict):
                sql_clean = re.sub(r'%\((\w+)\)s', r':\1', sql_clean)
            self.cur.execute(sql_clean, params)
        else:
            statements = [s.strip() for s in sql_clean.split(";") if s.strip()]
            if len(statements) > 1:
                for stmt in statements:
                    self.cur.execute(stmt)
            else:
                self.cur.execute(sql_clean)
        return self
    def executemany(self, sql, seq_of_parameters):
        import re
        sql_clean = sql.replace("%s", "?")
        # Replace NOW() / now() with CURRENT_TIMESTAMP
        sql_clean = re.sub(r'(?i)\bnow\(\)', "CURRENT_TIMESTAMP", sql_clean)
        # Replace SERIAL PRIMARY KEY / serial primary key with INTEGER PRIMARY KEY AUTOINCREMENT
        sql_clean = re.sub(r'(?i)\bserial\s+primary\s+key\b', "INTEGER PRIMARY KEY AUTOINCREMENT", sql_clean)
        self.cur.executemany(sql_clean, seq_of_parameters)
        return self
    def __getattr__(self, name):
        return getattr(self.cur, name)
    def __iter__(self):
        return iter(self.cur)
    def __next__(self):
        return next(self.cur)

class ContextConnection(sqlite3.Connection):
    def cursor(self, *args, **kwargs):
        return ContextCursor(super().cursor(*args, **kwargs))

@contextmanager
def get_pg_conn() -> Iterator[sqlite3.Connection]:
    """
    Yields a raw sqlite3 Connection.
    Named 'get_pg_conn' to act as a drop-in replacement for PostgreSQL connections.
    """
    conn = sqlite3.connect(DB_FILE, timeout=30.0, factory=ContextConnection)
    # Enable registering custom functions
    conn.create_aggregate("STDDEV", 1, SqliteStdDev)
    # Return dictionary or tuple rows based on standard cursor
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error("SQLite transaction failed, rolled back: %s", e)
        raise e
    finally:
        conn.close()

def get_pool():
    """Dummy pool manager for compatibility."""
    return None

def bulk_insert(
    conn: sqlite3.Connection,
    table: str,
    columns: List[str],
    rows: List[List[Any] | Tuple[Any, ...]],
    chunk_size: int = 5000,
    on_conflict: str = "DO NOTHING"
) -> int:
    """
    Perform a high-performance bulk insert using SQLite syntax.
    """
    if not rows:
        return 0

    col_names = ", ".join(columns)
    placeholders = ", ".join(["?"] * len(columns))
    
    conflict_clause = ""
    if on_conflict == "DO NOTHING":
        conflict_clause = "OR IGNORE"
    
    sql = f"INSERT {conflict_clause} INTO {table} ({col_names}) VALUES ({placeholders})"
    
    cur = conn.cursor()
    total_inserted = 0
    
    # Process in chunks
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i : i + chunk_size]
        cur.executemany(sql, chunk)
        total_inserted += cur.rowcount
        
    conn.commit()
    logger.info("Bulk insert: %d rows inserted into %s", total_inserted, table)
    return total_inserted

def bulk_upsert(
    conn: sqlite3.Connection,
    table: str,
    columns: List[str],
    conflict_cols: List[str],
    rows: List[List[Any] | Tuple[Any, ...]]
) -> int:
    """
    Perform a bulk insert with ON CONFLICT UPDATE in SQLite.
    """
    if not rows:
        return 0

    col_names = ", ".join(columns)
    placeholders = ", ".join(["?"] * len(columns))
    
    conflict_target = ", ".join(conflict_cols)
    
    # Exclude conflict columns from update list
    update_cols = [c for c in columns if c not in conflict_cols]
    update_expr = ", ".join([f"{c}=excluded.{c}" for c in update_cols])
    
    sql = f"""
        INSERT INTO {table} ({col_names}) 
        VALUES ({placeholders}) 
        ON CONFLICT({conflict_target}) DO UPDATE SET {update_expr}
    """
    
    cur = conn.cursor()
    total_upserted = 0
    
    for i in range(0, len(rows), 5000):
        chunk = rows[i : i + 5000]
        cur.executemany(sql, chunk)
        total_upserted += cur.rowcount
        
    conn.commit()
    logger.info("Bulk upsert: %d rows processed in %s", total_upserted, table)
    return total_upserted

def query_to_df(sql: str, params: Optional[dict | tuple | list] = None) -> pd.DataFrame:
    """Execute SQL query and return a pandas DataFrame."""
    # Convert PostgreSQL SQL syntax to SQLite if needed
    # Standard query conversions:
    sql_clean = sql.replace("pg_trgm", "") # skip pg_trgm indicators
    
    with get_pg_conn() as conn:
        # Convert param dictionaries to standard tuple or pass dict directly (SQLite supports named parameters with :)
        if isinstance(params, dict):
            # SQLite uses :key, PostgreSQL uses %(key)s. Let's translate %(key)s to :key.
            sql_clean = re_replace_pg_params(sql_clean)
            df = pd.read_sql_query(sql_clean, conn, params=params)
        elif params is not None:
            df = pd.read_sql_query(sql_clean, conn, params=params)
        else:
            df = pd.read_sql_query(sql_clean, conn)
    return df

def re_replace_pg_params(sql: str) -> str:
    """Helper to replace PostgreSQL-style %(name)s parameters with SQLite-style :name."""
    import re
    return re.sub(r'%\((\w+)\)s', r':\1', sql)

def run_health_check() -> dict:
    """Test connection viability."""
    try:
        with get_pg_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1;")
            cur.fetchone()
        return {"status": "healthy", "database": "sqlite"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
