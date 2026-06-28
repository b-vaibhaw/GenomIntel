# psycopg2.extras mock for SQLite
import sqlite3

class RealDictCursor:
    pass

def execute_batch(cur, sql, argslist, page_size=500):
    """Fallback batch execution using SQLite's executemany."""
    # execute_batch runs executemany directly
    cur.executemany(sql, argslist)

def execute_values(cur, sql, argslist, template=None, page_size=500):
    """
    Simulate psycopg2's execute_values by translating the values block into SQLite format.
    execute_values is used in dag_utils.py.
    The sql is usually of form: "INSERT INTO table (c1, c2) VALUES %s"
    We translate the SQL to executemany form: "INSERT INTO table (c1, c2) VALUES (?, ?)"
    """
    import re
    # Translate VALUES %s or VALUES %s ON CONFLICT to appropriate SQLite statements
    sql_clean = sql.replace("%s", "")
    
    # Extract columns list to find out how many placeholders to generate
    # Example: "INSERT INTO pipeline_runs (run_id, dag_id) VALUES %s"
    cols_match = re.search(r'\((.*?)\)\s*VALUES', sql_clean, re.IGNORECASE)
    if cols_match:
        n_cols = len(cols_match.group(1).split(","))
    else:
        # fallback to the number of elements in the first row
        n_cols = len(argslist[0]) if argslist else 0
        
    placeholders = ", ".join(["?"] * n_cols)
    
    # If the SQL already contains VALUES, let's substitute, otherwise append
    if "VALUES" in sql_clean.upper():
        sql_final = re.sub(r'VALUES\s*.*', f'VALUES ({placeholders})', sql_clean, flags=re.IGNORECASE)
    else:
        sql_final = f"{sql_clean} VALUES ({placeholders})"
        
    cur.executemany(sql_final, argslist)
