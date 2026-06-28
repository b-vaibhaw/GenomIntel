# psycopg2 mock package for SQLite compatibility
import os
import sqlite3
import re
import math

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

class MockCursor:
    def __init__(self, sqlite_cursor, as_dict=False):
        self.cur = sqlite_cursor
        self.as_dict = as_dict

    def execute(self, sql, params=None):
        sql_clean = self._translate_sql(sql)
        if params is not None:
            # If params is a dict, translate to SQLite style dict params
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
        sql_clean = self._translate_sql(sql)
        self.cur.executemany(sql_clean, seq_of_parameters)
        return self

    def fetchone(self):
        row = self.cur.fetchone()
        if row is None:
            return None
        if self.as_dict:
            return {col[0]: row[idx] for idx, col in enumerate(self.cur.description)}
        return row

    def fetchall(self):
        rows = self.cur.fetchall()
        if self.as_dict:
            return [{col[0]: row[idx] for idx, col in enumerate(self.cur.description)} for row in rows]
        return rows

    @property
    def rowcount(self):
        return self.cur.rowcount

    @property
    def lastrowid(self):
        return self.cur.lastrowid

    def close(self):
        self.cur.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _translate_sql(self, sql):
        # Translate PostgreSQL-specific constructs to SQLite
        sql = sql.replace("postgresql+psycopg2://", "sqlite:///")
        sql = sql.replace("pg_trgm", "") # skip pg_trgm indicators
        # Replace %s placeholders with SQLite ?
        sql = sql.replace("%s", "?")
        # Replace uuid_generate_v4() with a random string generator or literal
        sql = re.sub(r'(?i)uuid_generate_v4\(\)', "lower(hex(randomblob(4)) || '-' || hex(randomblob(2)) || '-4' || substr(hex(randomblob(2)),2,3) || '-' || substr('89ab',abs(random() % 4) + 1, 1) || substr(hex(randomblob(2)),2,3) || '-' || hex(randomblob(6)))", sql)
        # Replace NOW() / now() with CURRENT_TIMESTAMP
        sql = re.sub(r'(?i)\bnow\(\)', "CURRENT_TIMESTAMP", sql)
        # Replace SERIAL PRIMARY KEY / serial primary key with INTEGER PRIMARY KEY AUTOINCREMENT
        sql = re.sub(r'(?i)\bserial\s+primary\s+key\b', "INTEGER PRIMARY KEY AUTOINCREMENT", sql)
        return sql

class MockConnection:
    def __init__(self, sqlite_conn):
        self.conn = sqlite_conn
        # Register custom aggregates
        self.conn.create_aggregate("STDDEV", 1, SqliteStdDev)

    def cursor(self, cursor_factory=None):
        as_dict = (cursor_factory is not None)
        return MockCursor(self.conn.cursor(), as_dict=as_dict)

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.rollback()
        else:
            self.commit()
        self.close()

def connect(dsn=None, **kwargs):
    # Retrieve DB file path
    db_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "biointel.db")
    # If the call is from a subdirectory (like scripts/ or dags/), go up accordingly
    if not os.path.exists(db_file):
        db_file = os.path.abspath("biointel.db")
    
    conn = sqlite3.connect(db_file, timeout=30.0)
    return MockConnection(conn)

class DatabaseError(Exception):
    pass

class extensions:
    class connection:
        pass
